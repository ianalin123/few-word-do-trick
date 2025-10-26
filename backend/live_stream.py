"""
Live EEG streaming from Muse headset with real-time emotion classification
Connects to Muse via LSL and sends predictions via WebSocket
"""

import asyncio
import websockets
import json
import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy import signal
from collections import deque
import sys

# Import preprocessing parameters from preprocess.py
ORIGINAL_SFREQ = 256  # Hz (Muse sampling rate)
TARGET_SFREQ = 128    # Hz (downsampled)
WINDOW_SIZE = 2.0     # seconds
WINDOW_OVERLAP = 0.5  # 50% overlap (1 second step)

# Filter parameters
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 60.0
FILTER_ORDER = 5

# Frequency bands
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Muse channel names (standard order)
CHANNEL_NAMES = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']


class LiveEEGProcessor:
    """Real-time EEG processing and feature extraction"""

    def __init__(self, buffer_duration=3.0):
        """
        Initialize live EEG processor

        Args:
            buffer_duration: How many seconds of data to keep in buffer
        """
        self.buffer_duration = buffer_duration
        self.buffer_size = int(buffer_duration * ORIGINAL_SFREQ)

        # Create circular buffer for each channel
        self.n_channels = len(CHANNEL_NAMES)
        self.buffer = deque(maxlen=self.buffer_size)

        # Processing flags
        self.last_process_time = 0
        self.process_interval = (1 - WINDOW_OVERLAP) * WINDOW_SIZE  # 1 second

    def add_sample(self, sample):
        """Add a new EEG sample to the buffer"""
        self.buffer.append(sample)

    def preprocess(self, data):
        """Apply preprocessing pipeline to raw EEG data"""
        # 1. Bandpass filter (1-45 Hz)
        nyquist = ORIGINAL_SFREQ / 2
        low = BANDPASS_LOW / nyquist
        high = BANDPASS_HIGH / nyquist

        sos_bandpass = signal.butter(FILTER_ORDER, [low, high], btype='band', output='sos')
        data_filtered = signal.sosfiltfilt(sos_bandpass, data, axis=1)

        # 2. Notch filter (60 Hz - power line noise)
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, Q, fs=ORIGINAL_SFREQ)
        data_notched = signal.filtfilt(b_notch, a_notch, data_filtered, axis=1)

        # 3. Downsample to 128 Hz
        downsample_factor = int(ORIGINAL_SFREQ / TARGET_SFREQ)
        data_downsampled = signal.decimate(data_notched, downsample_factor, axis=1)

        return data_downsampled

    def compute_band_power(self, data, sfreq=TARGET_SFREQ):
        """Compute power in each frequency band"""
        band_powers = {band: [] for band in FREQ_BANDS.keys()}

        for ch_idx in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch_idx, :], fs=sfreq, nperseg=min(256, data.shape[1]))

            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name].append(band_power)

        return band_powers

    def compute_differential_entropy(self, data, sfreq=TARGET_SFREQ):
        """Compute differential entropy for each frequency band"""
        de_values = {band: [] for band in FREQ_BANDS.keys()}

        for ch_idx in range(data.shape[0]):
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                nyquist = sfreq / 2
                low = low_freq / nyquist
                high = high_freq / nyquist

                sos = signal.butter(4, [low, high], btype='band', output='sos')
                band_signal = signal.sosfiltfilt(sos, data[ch_idx, :])

                variance = np.var(band_signal)
                if variance > 0:
                    de = 0.5 * np.log(2 * np.pi * np.e * variance)
                else:
                    de = 0.0

                de_values[band_name].append(de)

        return de_values

    def extract_features(self, segment, sfreq=TARGET_SFREQ):
        """Extract all 52 features from a 2-second segment"""
        features = []

        # 1. Band Power (20 features: 4 bands × 5 channels)
        band_powers = self.compute_band_power(segment, sfreq)
        for band_name in ['theta', 'alpha', 'beta', 'gamma']:
            for ch_idx in range(len(CHANNEL_NAMES)):
                features.append(band_powers[band_name][ch_idx])

        # 2. Differential Entropy (20 features: 4 bands × 5 channels)
        de_values = self.compute_differential_entropy(segment, sfreq)
        for band_name in ['theta', 'alpha', 'beta', 'gamma']:
            for ch_idx in range(len(CHANNEL_NAMES)):
                features.append(de_values[band_name][ch_idx])

        # 3. Asymmetry Features (2 features)
        # Frontal Alpha Asymmetry: log(AF8_alpha) - log(AF7_alpha)
        af7_idx = 1  # AF7
        af8_idx = 2  # AF8
        af7_alpha = band_powers['alpha'][af7_idx]
        af8_alpha = band_powers['alpha'][af8_idx]
        frontal_asymmetry = np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10)
        features.append(frontal_asymmetry)

        # Temporal Alpha Asymmetry: log(TP10_alpha) - log(TP9_alpha)
        tp9_idx = 0  # TP9
        tp10_idx = 3  # TP10
        tp9_alpha = band_powers['alpha'][tp9_idx]
        tp10_alpha = band_powers['alpha'][tp10_idx]
        temporal_asymmetry = np.log(tp10_alpha + 1e-10) - np.log(tp9_alpha + 1e-10)
        features.append(temporal_asymmetry)

        # 4. Power Ratios (10 features: 2 ratios × 5 channels)
        for ch_idx in range(len(CHANNEL_NAMES)):
            # Beta/Alpha ratio
            beta_power = band_powers['beta'][ch_idx]
            alpha_power = band_powers['alpha'][ch_idx]
            features.append(beta_power / (alpha_power + 1e-10))

            # Gamma/Theta ratio
            gamma_power = band_powers['gamma'][ch_idx]
            theta_power = band_powers['theta'][ch_idx]
            features.append(gamma_power / (theta_power + 1e-10))

        return features

    def process_buffer(self):
        """Process the current buffer and extract features"""
        if len(self.buffer) < int(WINDOW_SIZE * ORIGINAL_SFREQ):
            return None  # Not enough data yet

        # Get the last 2 seconds of data
        window_samples = int(WINDOW_SIZE * ORIGINAL_SFREQ)
        data = np.array(list(self.buffer)[-window_samples:])  # Shape: (samples, channels)
        data = data.T  # Shape: (channels, samples)

        # Preprocess
        data_preprocessed = self.preprocess(data)

        # Extract features
        features = self.extract_features(data_preprocessed)

        return features


async def connect_to_websocket(uri="ws://localhost:8000/ws"):
    """Connect to the FastAPI WebSocket server"""
    try:
        websocket = await websockets.connect(uri)
        print(f"✓ Connected to WebSocket: {uri}")
        return websocket
    except Exception as e:
        print(f"✗ Failed to connect to WebSocket: {e}")
        return None


async def main():
    """Main streaming loop"""
    print("="*60)
    print("LIVE EEG EMOTION CLASSIFICATION")
    print("="*60)

    # 1. Connect to Muse via LSL
    print("\n[1/3] Connecting to Muse headset...")
    print("Searching for EEG stream...")

    try:
        print("Looking for EEG streams (timeout: 5s)...")
        streams = resolve_byprop('type', 'EEG', timeout=5.0)

        if not streams:
            print("✗ No EEG stream found!")
            print("\nPlease ensure:")
            print("  1. Your Muse headset is on")
            print("  2. muselsl is streaming: muselsl stream")
            print("\nTip: Run 'muselsl list' to check if your Muse is detected")
            return

        print(f"✓ Found {len(streams)} EEG stream(s)")
        inlet = StreamInlet(streams[0])
        print(f"✓ Connected to: {streams[0].name()}")

    except Exception as e:
        print(f"✗ Error connecting to LSL: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if muselsl is installed: pip install muselsl")
        print("  2. Try running: muselsl list")
        print("  3. Then run: muselsl stream")
        return

    # 2. Connect to WebSocket server
    print("\n[2/3] Connecting to prediction server...")
    websocket = await connect_to_websocket()
    if not websocket:
        print("\nPlease ensure the backend server is running:")
        print("  cd backend && python app.py")
        return

    # Wait for welcome message
    try:
        welcome = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        msg = json.loads(welcome)
        print(f"✓ {msg.get('message', 'Connected')}")
    except:
        pass

    # 3. Start streaming
    print("\n[3/3] Starting live classification...")
    print("="*60)
    print("\nWearing your Muse? Your emotions are being classified!\n")

    processor = LiveEEGProcessor()
    sample_count = 0
    last_prediction_time = 0

    try:
        while True:
            # Get EEG sample from Muse
            sample, timestamp = inlet.pull_sample(timeout=1.0)

            if sample:
                # Add to buffer
                processor.add_sample(sample)
                sample_count += 1

                # Process every second (50% overlap)
                import time
                current_time = time.time()

                if current_time - last_prediction_time >= processor.process_interval:
                    # Extract features
                    features = processor.process_buffer()

                    if features is not None:
                        # Send to WebSocket for prediction
                        message = {
                            "type": "predict",
                            "features": features
                        }

                        await websocket.send(json.dumps(message))

                        # Wait for prediction
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            result = json.loads(response)

                            if result.get('type') == 'prediction':
                                emotion = result['emotion']
                                confidence = result['confidence'] * 100

                                # Print prediction
                                emoji = "😊" if emotion == "happy" else "😢"
                                print(f"  {emoji} {emotion.upper():8s} | Confidence: {confidence:5.1f}%")

                        except asyncio.TimeoutError:
                            print("  ⏱ Prediction timeout")
                        except Exception as e:
                            print(f"  ✗ Error: {e}")

                        last_prediction_time = current_time

            # Small delay to prevent CPU overload
            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Streaming stopped by user")
        print(f"Total samples processed: {sample_count}")
        print("="*60)

    finally:
        if websocket:
            await websocket.close()


if __name__ == "__main__":
    asyncio.run(main())
