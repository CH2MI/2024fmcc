import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import os

def plot_comparative_spectrogram(original_file_path, sr=22050, hop_length=512):
    # 오디오 파일 로드
    original_audio, sr = librosa.load(original_file_path, sr=sr)
    
    # 노이즈 감소 적용
    reduced_audio = nr.reduce_noise(y=original_audio, sr=sr)
    
    # STFT 계산 (Short-Time Fourier Transform)
    original_stft = librosa.stft(original_audio, hop_length=hop_length)
    reduced_stft = librosa.stft(reduced_audio, hop_length=hop_length)
    
    original_stft_db = librosa.amplitude_to_db(np.abs(original_stft), ref=np.max)
    reduced_stft_db = librosa.amplitude_to_db(np.abs(reduced_stft), ref=np.max)
    
    # 스펙트로그램 시각화
    plt.figure(figsize=(12, 10))
    
    # 원본 오디오 스펙트로그램
    plt.subplot(2, 1, 1)
    librosa.display.specshow(original_stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Audio Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # 노이즈 제거된 오디오 스펙트로그램
    plt.subplot(2, 1, 2)
    librosa.display.specshow(reduced_stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noise Reduced Audio Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()

def plot_waveforms(original_file_path, sr=22050):
    # 오디오 파일 로드
    original_audio, sr = librosa.load(original_file_path, sr=sr)
    
    # 노이즈 감소 적용
    reduced_audio = nr.reduce_noise(y=original_audio, sr=sr)
    
    # 시간축 생성
    time_original = np.linspace(0, len(original_audio) / sr, len(original_audio))
    time_reduced = np.linspace(0, len(reduced_audio) / sr, len(reduced_audio))
    
    # 파형 시각화
    plt.figure(figsize=(12, 8))
    
    # 원본 오디오 파형
    plt.subplot(2, 1, 1)
    plt.plot(time_original, original_audio)
    plt.title('Original Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # 노이즈 제거된 오디오 파형
    plt.subplot(2, 1, 2)
    plt.plot(time_reduced, reduced_audio)
    plt.title('Noise Reduced Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

with open('fmcc_train.ctl', 'r') as f:
    path = f.readline().strip()
    path = 'raw16k/train/' + path + '.wav'
    plot_comparative_spectrogram(path, 16000)
    plot_waveforms(path, 16000)