import io
import subprocess
from typing import Optional

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


_TARGET_SR = 16000
_N_MFCC = 20
_MATCH_THRESHOLD = 0.85
_RUNNING_AVG_ALPHA = 0.7  # weight on stored fingerprint when blending in new
_TRIM_TOP_DB = 20
_MIN_VOICED_SECONDS = 0.3


def _decode_to_pcm(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Decode arbitrary audio bytes (webm/opus/wav/etc.) to mono 16kHz PCM via ffmpeg."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
             "-f", "wav", "-ar", str(_TARGET_SR), "-ac", "1", "pipe:1"],
            input=audio_bytes,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        y, _ = librosa.load(io.BytesIO(proc.stdout), sr=_TARGET_SR, mono=True)
        return y
    except Exception:
        return None


def _fingerprint(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    return mfcc.mean(axis=1)


class SpeakerService:
    def __init__(self) -> None:
        # {conv_id: {speaker_id: fingerprint_vector}}
        self._fingerprints: dict[str, dict[str, np.ndarray]] = {}

    def identify_speaker(self, conv_id: str, audio_bytes: bytes) -> tuple[str, float]:
        y = _decode_to_pcm(audio_bytes)
        if y is None or len(y) == 0:
            print("[diarize] decode failed, using speaker_unknown")
            return "speaker_unknown", 0.0

        y_trim, _ = librosa.effects.trim(y, top_db=_TRIM_TOP_DB)
        if len(y_trim) < _MIN_VOICED_SECONDS * _TARGET_SR:
            print("[diarize] decode failed, using speaker_unknown")
            return "speaker_unknown", 0.0

        fp = _fingerprint(y_trim, _TARGET_SR)
        conv_fps = self._fingerprints.setdefault(conv_id, {})

        if not conv_fps:
            speaker_id = "speaker_1"
            conv_fps[speaker_id] = fp
            print(f"[diarize] speaker={speaker_id} similarity=1.00")
            return speaker_id, 1.0

        # Cosine similarity against each known speaker in this conv
        existing_ids = list(conv_fps.keys())
        existing_vecs = np.stack([conv_fps[sid] for sid in existing_ids])
        sims = cosine_similarity(fp.reshape(1, -1), existing_vecs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim > _MATCH_THRESHOLD:
            speaker_id = existing_ids[best_idx]
            conv_fps[speaker_id] = (
                _RUNNING_AVG_ALPHA * conv_fps[speaker_id]
                + (1.0 - _RUNNING_AVG_ALPHA) * fp
            )
            print(f"[diarize] speaker={speaker_id} similarity={best_sim:.2f}")
            return speaker_id, best_sim

        speaker_id = f"speaker_{len(conv_fps) + 1}"
        conv_fps[speaker_id] = fp
        print(f"[diarize] speaker={speaker_id} similarity={best_sim:.2f}")
        return speaker_id, best_sim


speaker_service = SpeakerService()
