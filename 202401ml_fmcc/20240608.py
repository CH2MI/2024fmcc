import matplotlib.pyplot as plt
import librosa
import numpy as np

sample_rate = 16000

def plot_signal_and_transformations(path):
    # Load audio signal
    audio, sr = librosa.load(path, sr=sample_rate)
    
    # Pre-emphasis
    pre_emphasized = librosa.effects.preemphasis(audio, coef=0.97)
    
    # Windowing (Hamming)
    windowed = audio * np.hamming(len(audio))
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    axs[0].plot(audio)
    axs[0].set_title('Original Signal')
    axs[0].set_ylabel('Amplitude')
    
    axs[1].plot(pre_emphasized)
    axs[1].set_title('Pre-emphasized Signal')
    axs[1].set_ylabel('Amplitude')
    
    axs[2].plot(windowed)
    axs[2].set_title('Hamming Windowed Signal')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlabel('Sample')
    
    plt.tight_layout()
    plt.show()

# Test the function with a sample path
# Replace 'sample_audio_path.wav' with an actual audio file path to test
plot_signal_and_transformations('sample_audio_path.wav')
