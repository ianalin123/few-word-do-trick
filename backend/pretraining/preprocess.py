"""
EEG Emotion Recognition - Preprocessing & Feature Extraction (FIXED)
Processes raw EEG data and extracts features for emotion classification.
NOW TRACKS TRIAL ID TO PREVENT DATA LEAKAGE!
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')


# Processing parameters
ORIGINAL_SFREQ = 256  # Hz (Muse sampling rate)
TARGET_SFREQ = 128    # Hz (downsampled)
WINDOW_SIZE = 2.0     # seconds
WINDOW_OVERLAP = 0.5  # 50% overlap (1 second step)

# Filter parameters
BANDPASS_LOW = 1.0    # Hz
BANDPASS_HIGH = 45.0  # Hz
NOTCH_FREQ = 60.0     # Hz (power line noise)
FILTER_ORDER = 5

# Frequency bands (standard neuroscience definitions)
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Default channel names (Muse standard)
DEFAULT_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']


class EEGPreprocessor:
    """Handles EEG preprocessing and feature extraction."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.features_list = []
        self.channel_names = None

    def load_trial(self, filepath):
        """
        Load a single trial CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            tuple: (data array, emotion label, channel names)
        """
        try:
            df = pd.read_csv(filepath)

            # Extract emotion from filename (e.g., "lemon_trial1.csv" -> "lemon")
            filename = os.path.basename(filepath)
            emotion = filename.split('_')[0]

            # Get channel names (all columns except 'timestamp')
            channel_names = [col for col in df.columns if col != 'timestamp']

            # Extract EEG channels
            eeg_data = df[channel_names].values.T  # Shape: (n_channels, n_samples)

            # Store channel names if first trial
            if self.channel_names is None:
                self.channel_names = channel_names
                print(f"  Detected {len(channel_names)} channels: {', '.join(channel_names)}")

            return eeg_data, emotion

        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None, None

    def preprocess(self, data, sfreq=ORIGINAL_SFREQ):
        """
        Apply preprocessing pipeline to raw EEG data.

        Steps:
        1. Bandpass filter (1-45 Hz)
        2. Notch filter (60 Hz)
        3. Downsample to 128 Hz

        Args:
            data: EEG data array (channels × samples)
            sfreq: Sampling frequency

        Returns:
            Preprocessed data array
        """
        # 1. Bandpass filter (1-45 Hz)
        nyquist = sfreq / 2
        low = BANDPASS_LOW / nyquist
        high = BANDPASS_HIGH / nyquist

        sos_bandpass = signal.butter(FILTER_ORDER, [low, high], btype='band', output='sos')
        data_filtered = signal.sosfiltfilt(sos_bandpass, data, axis=1)

        # 2. Notch filter (60 Hz - power line noise)
        Q = 30.0  # Quality factor
        b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, Q, fs=sfreq)
        data_notched = signal.filtfilt(b_notch, a_notch, data_filtered, axis=1)

        # 3. Downsample to 128 Hz
        downsample_factor = int(sfreq / TARGET_SFREQ)
        data_downsampled = signal.decimate(data_notched, downsample_factor, axis=1)

        return data_downsampled

    def segment_data(self, data, sfreq=TARGET_SFREQ):
        """
        Segment data into 2-second windows with 50% overlap.

        Args:
            data: Preprocessed EEG data (channels × samples)
            sfreq: Sampling frequency

        Returns:
            List of data segments
        """
        window_samples = int(WINDOW_SIZE * sfreq)
        step_samples = int((1 - WINDOW_OVERLAP) * window_samples)

        segments = []
        n_samples = data.shape[1]

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            segment = data[:, start:end]
            segments.append(segment)

        return segments

    def compute_band_power(self, data, sfreq=TARGET_SFREQ):
        """
        Compute power in each frequency band using Welch's method.

        Args:
            data: EEG segment (channels × samples)
            sfreq: Sampling frequency

        Returns:
            Dictionary of band powers per channel
        """
        band_powers = {band: [] for band in FREQ_BANDS.keys()}

        for ch_idx in range(data.shape[0]):
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(data[ch_idx, :], fs=sfreq, nperseg=min(256, data.shape[1]))

            # Extract power in each band
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name].append(band_power)

        return band_powers

    def compute_differential_entropy(self, data, sfreq=TARGET_SFREQ):
        """
        Compute differential entropy for each frequency band.

        DE = 0.5 * log(2 * π * e * variance)

        Args:
            data: EEG segment (channels × samples)
            sfreq: Sampling frequency

        Returns:
            Dictionary of DE values per channel
        """
        de_values = {band: [] for band in FREQ_BANDS.keys()}

        for ch_idx in range(data.shape[0]):
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                # Bandpass filter for specific band
                nyquist = sfreq / 2
                low = low_freq / nyquist
                high = high_freq / nyquist

                sos = signal.butter(4, [low, high], btype='band', output='sos')
                band_signal = signal.sosfiltfilt(sos, data[ch_idx, :])

                # Compute differential entropy
                variance = np.var(band_signal)
                if variance > 0:
                    de = 0.5 * np.log(2 * np.pi * np.e * variance)
                else:
                    de = 0.0

                de_values[band_name].append(de)

        return de_values

    def extract_features(self, segment, sfreq=TARGET_SFREQ):
        """
        Extract all features from a single 2-second segment.

        Features:
        - Band power (4 bands × N channels)
        - Differential entropy (4 bands × N channels)
        - Asymmetry features (2 features: frontal, temporal - if applicable)
        - Power ratios (2 ratios × N channels)

        Args:
            segment: EEG data segment (N channels × samples)
            sfreq: Sampling frequency

        Returns:
            Dictionary of features
        """
        features = {}

        # 1. Band Power
        band_powers = self.compute_band_power(segment, sfreq)
        for band_name in FREQ_BANDS.keys():
            for ch_idx, ch_name in enumerate(self.channel_names):
                features[f'{ch_name}_power_{band_name}'] = band_powers[band_name][ch_idx]

        # 2. Differential Entropy
        de_values = self.compute_differential_entropy(segment, sfreq)
        for band_name in FREQ_BANDS.keys():
            for ch_idx, ch_name in enumerate(self.channel_names):
                features[f'{ch_name}_de_{band_name}'] = de_values[band_name][ch_idx]

        # 3. Asymmetry Features (only if channels are present)
        # Frontal Alpha Asymmetry: log(AF8_alpha) - log(AF7_alpha) [STANDARD METHOD]
        if 'AF7' in self.channel_names and 'AF8' in self.channel_names:
            af7_idx = self.channel_names.index('AF7')
            af8_idx = self.channel_names.index('AF8')
            af7_alpha = band_powers['alpha'][af7_idx]
            af8_alpha = band_powers['alpha'][af8_idx]
            # Use log ratio (standard in neuroscience literature)
            features['frontal_alpha_asymmetry'] = np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10)

        # Temporal Alpha Asymmetry: log(TP10_alpha) - log(TP9_alpha) [STANDARD METHOD]
        if 'TP9' in self.channel_names and 'TP10' in self.channel_names:
            tp9_idx = self.channel_names.index('TP9')
            tp10_idx = self.channel_names.index('TP10')
            tp9_alpha = band_powers['alpha'][tp9_idx]
            tp10_alpha = band_powers['alpha'][tp10_idx]
            # Use log ratio (standard in neuroscience literature)
            features['temporal_alpha_asymmetry'] = np.log(tp10_alpha + 1e-10) - np.log(tp9_alpha + 1e-10)

        # 4. Power Ratios
        for ch_idx, ch_name in enumerate(self.channel_names):
            # Beta/Alpha ratio
            beta_power = band_powers['beta'][ch_idx]
            alpha_power = band_powers['alpha'][ch_idx]
            features[f'{ch_name}_beta_alpha_ratio'] = beta_power / (alpha_power + 1e-10)

            # Gamma/Theta ratio
            gamma_power = band_powers['gamma'][ch_idx]
            theta_power = band_powers['theta'][ch_idx]
            features[f'{ch_name}_gamma_theta_ratio'] = gamma_power / (theta_power + 1e-10)

        return features

    def process_trial(self, filepath):
        """
        Process a single trial file.
        FIXED: Now tracks trial_id for each window!

        Args:
            filepath: Path to trial CSV file

        Returns:
            List of feature dictionaries (one per window)
        """
        # Load data
        data, emotion = self.load_trial(filepath)

        if data is None:
            return []

        # Extract trial ID from filename (e.g., "happy_trial1.csv" -> "happy_trial1")
        filename = os.path.basename(filepath)
        trial_id = filename.replace('.csv', '')

        print(f"  Processing: {filename}")
        print(f"    Trial ID: {trial_id}")
        print(f"    Raw shape: {data.shape}")

        # Preprocess
        data_preprocessed = self.preprocess(data)
        print(f"    Preprocessed shape: {data_preprocessed.shape}")

        # Segment into windows
        segments = self.segment_data(data_preprocessed)
        print(f"    Number of windows: {len(segments)}")

        # Extract features from each window
        trial_features = []
        for segment in segments:
            features = self.extract_features(segment)
            features['emotion'] = emotion
            features['trial_id'] = trial_id  # ⭐ ADD TRIAL ID
            trial_features.append(features)

        return trial_features

    def process_all_trials(self, data_dir='data'):
        """
        Process all trial files in the data directory.

        Args:
            data_dir: Directory containing trial CSV files

        Returns:
            DataFrame with all extracted features
        """
        if not os.path.exists(data_dir):
            print(f"✗ Data directory '{data_dir}' not found!")
            return None

        # Get all CSV files
        csv_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.csv')
        ])

        if not csv_files:
            print(f"✗ No CSV files found in '{data_dir}'")
            return None

        print(f"\nFound {len(csv_files)} trial files")
        print("="*60)

        all_features = []

        for i, filepath in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}]")
            trial_features = self.process_trial(filepath)
            all_features.extend(trial_features)

        print("\n" + "="*60)
        print(f"✓ Processing complete!")
        print(f"  Total windows: {len(all_features)}")

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Move emotion and trial_id columns to end
        feature_cols = [c for c in df.columns if c not in ['emotion', 'trial_id']]
        df = df[feature_cols + ['trial_id', 'emotion']]

        return df


def main():
    """Main preprocessing workflow."""
    print("="*60)
    print("EEG EMOTION RECOGNITION - PREPROCESSING & FEATURE EXTRACTION")
    print("="*60)

    # Initialize preprocessor
    preprocessor = EEGPreprocessor()

    # Process all trials
    df_features = preprocessor.process_all_trials()

    if df_features is not None:
        # Save to CSV
        output_file = 'features.csv'
        df_features.to_csv(output_file, index=False)

        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"✓ Features saved to: {output_file}")
        print(f"  Shape: {df_features.shape[0]} windows × {df_features.shape[1]} features")
        print(f"  Feature columns: {df_features.shape[1] - 2} (+ trial_id + emotion)")

        # Show emotion distribution
        print("\nEmotion distribution:")
        emotion_counts = df_features['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"  {emotion:12s}: {count:4d} windows")

        # Show trial distribution
        print("\nTrials per emotion:")
        for emotion in df_features['emotion'].unique():
            emotion_trials = df_features[df_features['emotion'] == emotion]['trial_id'].nunique()
            print(f"  {emotion:12s}: {emotion_trials} trials")

        # Show sample of features
        print("\nSample features (first 5 columns + trial_id + emotion):")
        print(df_features.iloc[:3, [0,1,2,3,4,-2,-1]])

        print("\n" + "="*60)
        print("Next steps:")
        print("  1. Run training script to train model with GroupKFold")
        print("  2. GroupKFold will use trial_id to prevent data leakage!")
        print("="*60)

    else:
        print("\n✗ Feature extraction failed. Please check:")
        print("  1. Data files exist in 'data/' directory")
        print("  2. CSV files have correct format (timestamp, TP9, AF7, AF8, TP10)")


if __name__ == "__main__":
    main()