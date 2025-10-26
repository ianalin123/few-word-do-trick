"""
EEG Emotion Recognition - IMPROVED Data Recording Script (Phase 1)
Records 60-second EEG trials with LIVE FFT visualization for quality control.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
import os
from datetime import datetime


# Emotion definitions with STRONG induction prompts
EMOTIONS = {
    'irritation': {
        'description': 'Biting into a lemon (irritation/anger)',
        'prompt': '''
😤 IRRITATION - Anger/Annoyance
═══════════════════════════════
Close your eyes. Imagine:
- Biting into a VERY sour lemon
- The sour taste flooding your mouth
- Your face scrunching up in disgust
- Feeling annoyed and irritated

TIP: Actually make the facial expression!
Think of something that recently frustrated you.
        '''
    },
    'happy': {
        'description': 'Warm cozy blanket on cold day (happy/comfort)',
        'prompt': '''
😊 HAPPY - Joy/Comfort
══════════════════════
Close your eyes. Imagine:
- Wrapped in a warm, soft blanket
- Feeling cozy and safe
- A smile naturally forming on your face
- Pure contentment and happiness

TIP: Actually smile! Recall a happy memory.
        '''
    },
    'sadness': {
        'description': 'Feeling down and melancholic (sadness/sorrow)',
        'prompt': '''
😔 SADNESS - Melancholy/Sorrow
═══════════════════════════════
Close your eyes. Imagine:
- A heavy feeling in your chest
- Thinking of a loss or disappointment
- Tears welling up in your eyes
- Feeling down and hopeless

TIP: Think of something sad. Let yourself feel it.
        '''
    },
    'neutral': {
        'description': 'Standing outside on windy night (calm/neutral)',
        'prompt': '''
😐 NEUTRAL - Calm/Baseline
══════════════════════════
Close your eyes. Imagine:
- Standing outside on a quiet, windy evening
- Looking at distant city lights
- Gentle breeze on your face
- Feeling peaceful and centered

TIP: Take slow, deep breaths. Relax all muscles.
        '''
    }
}

# Recording parameters
RECORDING_DURATION = 60  # seconds (FIXED: was 30, now 60)
SAMPLING_RATE = 256  # Hz (Muse default)
FFT_WINDOW = 512  # samples for FFT
FFT_UPDATE_RATE = 4  # Hz (update every 0.25 seconds)


class EEGRecorder:
    """Handles EEG data recording from Muse headset with live FFT visualization."""

    def __init__(self):
        """Initialize the EEG recorder."""
        self.inlet = None
        self.data_buffer = []
        self.timestamps = []
        self.channel_names = []
        self.n_channels = 0

    def connect_to_muse(self):
        """Connect to Muse EEG stream via LSL."""
        print("Searching for Muse EEG stream...")
        print("Make sure you've run 'muselsl stream' in another terminal first!")

        try:
            streams = resolve_byprop('type', 'EEG', timeout=10)
            if not streams:
                raise RuntimeError("No EEG stream found! Make sure Muse is streaming.")

            self.inlet = StreamInlet(streams[0])
            print(f"✓ Connected to Muse EEG stream")

            # Get channel names from stream info
            info = self.inlet.info()
            self.n_channels = info.channel_count()

            # Extract all channel names
            self.channel_names = []
            ch = info.desc().child('channels').child('channel')
            for i in range(self.n_channels):
                ch_name = ch.child_value('label')
                self.channel_names.append(ch_name)
                ch = ch.next_sibling()

            print(f"  Channels ({self.n_channels}): {', '.join(self.channel_names)}")
            print(f"  Sampling rate: {info.nominal_srate()} Hz")

            return True

        except Exception as e:
            print(f"✗ Error connecting to Muse: {e}")
            return False

    def compute_psd(self, data, fs=SAMPLING_RATE):
        """
        Compute Power Spectral Density for live visualization.
        
        Args:
            data: EEG data (channels × samples)
            fs: Sampling rate
            
        Returns:
            freqs, psd (averaged across channels)
        """
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(data, fs=fs, nperseg=min(FFT_WINDOW, data.shape[1]), axis=1)
        
        # Average across channels for display
        psd_mean = np.mean(psd, axis=0)
        
        return freqs, psd_mean

    def countdown(self, emotion_name):
        """Show countdown before recording starts."""
        print(f"\n{'='*60}")
        print(EMOTIONS[emotion_name]['prompt'])
        print(f"{'='*60}")
        input("Press ENTER when ready to start recording...")

        for i in range(3, 0, -1):
            print(f"\n  {i}...")
            time.sleep(1)
        print("\n  🔴 RECORDING! Maintain the emotional state...\n")

    def record_trial(self, emotion, trial_num):
        """
        Record a single 60-second trial with LIVE FFT visualization.

        Args:
            emotion: Emotion name (e.g., 'lemon')
            trial_num: Trial number

        Returns:
            DataFrame with recorded EEG data
        """
        # Reset buffers
        self.data_buffer = []
        self.timestamps = []

        # Show countdown with strong prompt
        self.countdown(emotion)

        # Setup live plot with 2 subplots: Raw EEG + Live FFT/PSD
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Top subplot: Raw EEG (last 5 seconds)
        ax_raw = fig.add_subplot(gs[0])
        ax_raw.set_title(f"Recording: {emotion.upper()} - Trial {trial_num} | Raw EEG (last 5 sec)", 
                         fontsize=12, fontweight='bold')
        ax_raw.set_ylabel('Amplitude (µV)')
        ax_raw.set_xlabel('Time (s)')
        ax_raw.set_ylim(-100, 100)
        ax_raw.grid(True, alpha=0.3)
        
        # Create lines for each channel
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        raw_lines = []
        for i, ch_name in enumerate(self.channel_names):
            line, = ax_raw.plot([], [], lw=0.8, label=ch_name, color=colors[i % len(colors)], alpha=0.7)
            raw_lines.append(line)
        ax_raw.legend(loc='upper right', fontsize=8)

        # Bottom subplot: Live PSD (averaged across channels)
        ax_psd = fig.add_subplot(gs[1])
        ax_psd.set_title('Live Power Spectral Density (Band Powers)', fontsize=12, fontweight='bold')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power')
        ax_psd.set_xlim(1, 45)
        ax_psd.grid(True, alpha=0.3)
        
        # Add frequency band markers
        bands = {'Theta\n4-8Hz': (4, 8), 'Alpha\n8-13Hz': (8, 13), 
                 'Beta\n13-30Hz': (13, 30), 'Gamma\n30-45Hz': (30, 45)}
        for i, (label, (low, high)) in enumerate(bands.items()):
            ax_psd.axvspan(low, high, alpha=0.1, color=colors[i])
            ax_psd.text((low+high)/2, ax_psd.get_ylim()[1]*0.9, label, 
                       ha='center', fontsize=8, fontweight='bold')
        
        psd_line, = ax_psd.plot([], [], 'k-', lw=2, label='Power Spectrum')
        ax_psd.legend(loc='upper right')

        plt.ion()
        plt.show()

        # Record data
        target_samples = int(RECORDING_DURATION * SAMPLING_RATE)  # 60s × 256Hz = 15,360 samples
        sample_count = 0
        last_update = time.time()
        last_progress = 0
        
        # Flush any buffered samples from before recording started
        print("🔄 Flushing old buffered samples...")
        self.inlet.flush()
        time.sleep(0.5)  # Brief pause to ensure clean start

        print(f"🔴 Recording {target_samples} samples (~{RECORDING_DURATION} seconds)...")
        print("   Watch the FFT to ensure good brain signal!\n")
        
        start_wall_time = time.time()
        first_timestamp = None

        try:
            while sample_count < target_samples:
                # Pull sample from stream with short timeout for responsiveness
                sample, lsl_timestamp = self.inlet.pull_sample(timeout=0.1)

                if sample:
                    # Record first timestamp as reference
                    if first_timestamp is None:
                        first_timestamp = lsl_timestamp
                    
                    # Use REAL LSL timestamp (relative to start)
                    relative_time = lsl_timestamp - first_timestamp
                    
                    self.data_buffer.append(sample)
                    self.timestamps.append(relative_time)
                    sample_count += 1

                    # Update plot at specified rate
                    if time.time() - last_update >= 1.0 / FFT_UPDATE_RATE:
                        last_update = time.time()
                        
                        data_array = np.array(self.data_buffer)
                        time_array = np.array(self.timestamps)

                        # Update RAW EEG plot (last 5 seconds)
                        window_start = max(0, len(time_array) - int(5 * SAMPLING_RATE))
                        time_window = time_array[window_start:]
                        data_window = data_array[window_start:]
                        
                        for i, line in enumerate(raw_lines):
                            if len(time_window) > 0:
                                line.set_data(time_window, data_window[:, i])
                        
                        if len(time_window) > 0:
                            ax_raw.set_xlim(time_window[0], time_window[-1] + 0.1)

                        # Update PSD plot (last 2 seconds of data)
                        if len(data_array) >= FFT_WINDOW:
                            recent_data = data_array[-int(2*SAMPLING_RATE):].T  # Shape: (channels, samples)
                            freqs, psd = self.compute_psd(recent_data)
                            
                            # Limit to 1-45 Hz
                            freq_mask = (freqs >= 1) & (freqs <= 45)
                            psd_line.set_data(freqs[freq_mask], psd[freq_mask])
                            ax_psd.set_ylim(0, np.max(psd[freq_mask]) * 1.1)

                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    # Show progress every 10 seconds worth of samples
                    if relative_time >= last_progress + 10 and last_progress < 60:
                        last_progress = int(relative_time / 10) * 10
                        remaining = RECORDING_DURATION - relative_time
                        progress_pct = (sample_count / target_samples) * 100
                        print(f"  ⏱️  {sample_count}/{target_samples} samples ({progress_pct:.1f}%) - ~{relative_time:.1f}s elapsed")
                else:
                    # No sample received - check if stream is still alive
                    elapsed_wall = time.time() - start_wall_time
                    if elapsed_wall > RECORDING_DURATION + 10:  # Give 10 extra seconds grace period
                        print(f"⚠️  Warning: No samples for too long. Stream may be disconnected.")
                        break

        except KeyboardInterrupt:
            print("\n✗ Recording interrupted by user")
            plt.close()
            return None

        plt.close()
        print(f"\n✓ Recording complete! Collected {sample_count} samples")

        # Create DataFrame
        df = pd.DataFrame(
            self.data_buffer,
            columns=self.channel_names
        )
        df.insert(0, 'timestamp', self.timestamps)

        return df

    def save_trial(self, df, emotion, trial_num):
        """Save trial data to CSV."""
        filename = f"data/{emotion}_trial{trial_num}.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Saved to {filename}")
        print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} columns")


def main():
    """Main recording workflow."""
    print("="*60)
    print("EEG EMOTION RECOGNITION - IMPROVED DATA RECORDING")
    print("="*60)

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Initialize recorder
    recorder = EEGRecorder()

    # Connect to Muse
    if not recorder.connect_to_muse():
        print("\n✗ Failed to connect. Please:")
        print("  1. Turn on your Muse headset")
        print("  2. Run 'muselsl stream' in another terminal")
        print("  3. Wait for connection, then run this script again")
        return

    print(f"\n{'='*60}")
    print("EMOTIONS TO RECORD:")
    for i, (emotion, info) in enumerate(EMOTIONS.items(), 1):
        print(f"  {i}. {emotion:12s} - {info['description']}")
    print(f"{'='*60}\n")
    print("⚠️  IMPORTANT TIPS:")
    print("   • Each trial = 60 seconds (1 minute)")
    print("   • Record 10 trials per emotion (10 min per emotion)")
    print("   • Watch the live FFT - alpha should be high when calm")
    print("   • Take breaks between emotions to reset your state")
    print("   • Really FEEL the emotions, don't just think about them!")
    print(f"{'='*60}\n")

    # Recording loop
    try:
        while True:
            print("\nOptions:")
            print("  1. Record a trial")
            print("  2. Exit")

            choice = input("\nEnter choice (1-2): ").strip()

            if choice == '2':
                print("Exiting...")
                break

            elif choice == '1':
                # Select emotion
                print("\nSelect emotion:")
                emotion_list = list(EMOTIONS.keys())
                for i, emotion in enumerate(emotion_list, 1):
                    print(f"  {i}. {emotion} - {EMOTIONS[emotion]['description']}")

                emotion_choice = input("Enter emotion number (1-4): ").strip()

                try:
                    emotion_idx = int(emotion_choice) - 1
                    if emotion_idx < 0 or emotion_idx >= len(emotion_list):
                        print("✗ Invalid emotion number")
                        continue

                    emotion = emotion_list[emotion_idx]

                except ValueError:
                    print("✗ Invalid input")
                    continue

                # Get trial number
                trial_num = input(f"Enter trial number for {emotion} (1-10): ").strip()

                try:
                    trial_num = int(trial_num)
                    if trial_num < 1:
                        print("✗ Trial number must be positive")
                        continue
                except ValueError:
                    print("✗ Invalid trial number")
                    continue

                # Check if file exists
                filename = f"data/{emotion}_trial{trial_num}.csv"
                if os.path.exists(filename):
                    overwrite = input(f"⚠️  {filename} exists. Overwrite? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        continue

                # Record trial
                df = recorder.record_trial(emotion, trial_num)

                if df is not None:
                    recorder.save_trial(df, emotion, trial_num)
                    print("\n✓ Trial recorded successfully!")

            else:
                print("✗ Invalid choice")

    except KeyboardInterrupt:
        print("\n\nRecording session ended.")

    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)

    # Show recorded trials
    if os.path.exists('data'):
        files = sorted([f for f in os.listdir('data') if f.endswith('.csv')])
        if files:
            print(f"Recorded trials: {len(files)}")
            for emotion in EMOTIONS.keys():
                emotion_files = [f for f in files if f.startswith(emotion)]
                print(f"  {emotion:12s}: {len(emotion_files)} trials")
            
            print(f"\nTarget: 40 trials (10 per emotion)")
            print(f"Progress: {len(files)}/40 ({len(files)/40*100:.0f}%)")
        else:
            print("No trials recorded yet.")

    print("\nNext steps:")
    print("  1. Record remaining trials to reach 40 total")
    print("  2. Run 'python preprocess.py' to extract features")
    print("  3. Run 'python visualize_pca.py' to check separability")
    print("="*60)


if __name__ == "__main__":
    main()