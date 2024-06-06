import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import noisereduce as nr
import joblib

sample_rate = 16000
dtype = np.int16

max_length = 0

def Read_Path(f):
    return 

# Spectral Subtraction을 사용한 노이즈 제거 함수
def spectral_subtraction(audio, noise_threshold=1.0):
    stft_audio = librosa.stft(audio)  # Short-time Fourier Transform (STFT) 계산
    magnitude_audio = np.abs(stft_audio)  # 스펙트럼의 크기 계산
    phase_audio = np.angle(stft_audio)  # 스펙트럼의 위상 계산

    # 노이즈 스펙트럼 추정
    noise = np.mean(magnitude_audio[:, :int(magnitude_audio.shape[1] * 0.1)], axis=1)  # 처음 10% 프레임을 노이즈로 가정
    noise_mag = np.repeat(np.expand_dims(noise, axis=1), magnitude_audio.shape[1], axis=1)

    # 노이즈 제거
    magnitude_clean = np.maximum(magnitude_audio - noise_mag, noise_threshold)

    # ISTFT 계산
    stft_clean = magnitude_clean * np.exp(1.0j * phase_audio)
    audio_clean = librosa.istft(stft_clean)
    return audio_clean

def MFCC(path):
     # 음성 신호 불러오기
    sr=16000
    audio, sr = librosa.load(path, sr=sr)
    y = librosa.effects.preemphasis(audio, coef=0.97)
    y_clean = spectral_subtraction(y)
    # STFT 계산 (Short-Time Fourier Transform)
    stft = librosa.stft(y_clean)
    magnitude = np.abs(stft)  # Magnitude spectrum

    # Mel 필터 뱅크 적용
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
    mel_spectrogram = np.dot(mel_filter_bank, magnitude)

    # 로그 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr,n_mfcc=75, n_fft=8192, hop_length=2048, dct_type=3, lifter=23)
    return mfcc

def wav2mfcc(type, cnt, max_length):
    data = []
    with open('fmcc_' + type + '.ctl', 'r') as f:
        for i in range (cnt):
            path = f.readline().strip()
            path = 'raw16k/' + type + '/' + path + '.wav'
            mfcc = MFCC(path).T
            data.append(mfcc)
            if len(mfcc) > max_length:
                max_length = len(mfcc)
    return data, max_length

def MakeFeatrue(padding):
    feature = []
    for i in range(len(padding)):   
        mfcc = padding[i].T
        mfcc = StandardScaler().fit_transform(mfcc)
        # 델타와 델타-델타 계산
        delta_mfccs = librosa.feature.delta(mfcc)
        delta2_mfccs = librosa.feature.delta(mfcc, order=2)   

        F = np.concatenate([mfcc, delta_mfccs, delta2_mfccs], axis=0)
        feature.append(F.T)
    feature1 = np.array(feature)
    flatted = feature1.reshape(feature1.shape[0], -1)
    return flatted

train, max_length = wav2mfcc('train', 8000, max_length)

train_padding = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in train]

feature = MakeFeatrue(train_padding)
print(feature.shape)

labels = np.concatenate((np.zeros(4000), np.ones(4000)), axis=0)

# Train LDA
lda = LinearDiscriminantAnalysis(n_components=1)

feature_lda = lda.fit_transform(feature, labels)

gmm20240601 = GaussianMixture(n_components=2, covariance_type='full', max_iter=200, random_state=0)
gmm20240601.fit(feature_lda)
gmm20240601_labels = gmm20240601.predict(feature_lda)

from scipy.stats import mode
from sklearn.metrics import accuracy_score

def map_labels(true_labels, cluster_labels):
    labels = np.zeros_like(cluster_labels)
    for i in range(2):  # 두 개의 클러스터
        mask = (cluster_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

mapped_labels = map_labels(labels, gmm20240601_labels)

# 정확도 확인
accuracy = accuracy_score(labels, mapped_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(feature_lda[labels == 0], np.zeros_like(feature_lda[labels == 0]) - 0.1, c='r', label='Female (True)', alpha=0.5)
plt.scatter(feature_lda[labels == 1], np.ones_like(feature_lda[labels == 1]) + 0.1, c='b', label='Male (True)', alpha=0.5)
plt.scatter(feature_lda[mapped_labels == 0], np.zeros_like(feature_lda[mapped_labels == 0]) - 0.3, marker='x', c='r', label='Female (Pred)', alpha=0.5)
plt.scatter(feature_lda[mapped_labels == 1], np.ones_like(feature_lda[mapped_labels == 1]) + 0.3, marker='x', c='b', label='Male (Pred)', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])
plt.legend()
plt.title('LDA-transformed Feature Visualization with GMM Clustering')
plt.show()

test, max_length = wav2mfcc('test', 1000, max_length)

test_padding = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in test]

feature = MakeFeatrue(test_padding)
print(feature.shape)
feature_lda = lda.transform(feature)
labels = np.concatenate((np.zeros(500), np.ones(500)), axis=0)
gmm20240601_labels = gmm20240601.predict(feature_lda)

mapped_labels = map_labels(labels, gmm20240601_labels)

# 정확도 확인
accuracy = accuracy_score(labels, mapped_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(feature_lda[labels == 0], np.zeros_like(feature_lda[labels == 0]) - 0.1, c='r', label='Female (True)', alpha=0.5)
plt.scatter(feature_lda[labels == 1], np.ones_like(feature_lda[labels == 1]) + 0.1, c='b', label='Male (True)', alpha=0.5)
plt.scatter(feature_lda[mapped_labels == 0], np.zeros_like(feature_lda[mapped_labels == 0]) - 0.3, marker='x', c='r', label='Female (Pred)', alpha=0.5)
plt.scatter(feature_lda[mapped_labels == 1], np.ones_like(feature_lda[mapped_labels == 1]) + 0.3, marker='x', c='b', label='Male (Pred)', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])
plt.legend()
plt.title('LDA-transformed Feature Visualization with GMM Clustering')
plt.show()
