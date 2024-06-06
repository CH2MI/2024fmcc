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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

sample_rate = 16000
window_size = 400
hop_length = 160
mfcc_length = 0
energy_length = 0
# Spectral Subtraction을 사용한 노이즈 제거 함수
def spectral_subtraction(audio, noise_threshold=1.0):
    stft_audio = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)  # Short-time Fourier Transform (STFT) 계산
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
    audio, sr = librosa.load(path, sr=sample_rate)
    y = librosa.effects.preemphasis(audio, coef=0.97)
    y_clean = spectral_subtraction(y)
    # STFT 계산 (Short-Time Fourier Transform)
    stft = librosa.stft(y_clean, n_fft=window_size, hop_length=hop_length)
    magnitude = np.abs(stft)  # Magnitude spectrum

    # Mel 필터 뱅크 적용
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=window_size, n_mels=128)
    mel_spectrogram = np.dot(mel_filter_bank, magnitude)

    # 로그 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    
    mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13, n_fft=window_size, hop_length=hop_length, dct_type=3, lifter=23)
    return mfcc

def wav2mfcc(type, cnt, mfcc_length):
    mfccs = []
    with open('fmcc_' + type + '.ctl', 'r') as f:
        for i in range (cnt):
            path = f.readline().strip()
            path = 'raw16k/' + type + '/' + path + '.wav'
            mfcc = MFCC(path).T
            mfccs.append(mfcc)
            if len(mfcc) > mfcc_length:
                mfcc_length = len(mfcc)
    return mfccs, mfcc_length

def pad_features(mfccs, mfcc_length):
    padded_mfccs = [np.pad(m, ((0, mfcc_length - m.shape[0]), (0, 0)), mode='constant') for m in mfccs]
    return padded_mfccs

def extract_features(padded_mfccs,):
    features = []
    for mfcc in padded_mfccs:
        delta_mfcc = librosa.feature.delta(mfcc.T).T
        double_delta_mfcc = librosa.feature.delta(mfcc.T, order=2).T
        combined_features = np.hstack((mfcc, delta_mfcc, double_delta_mfcc))
        features.append(combined_features)
    return np.array(features)

def standardize_feature(feature):
    # Flatten the features for each sample
    flattened_features = [f.flatten() for f in feature]
    # Convert to numpy array for further processing
    features_array = np.array(flattened_features)
    print(features_array.shape)
    train_data = np.reshape(feature, (len(feature) * len(feature[0]), len(feature[0][0])))
    print(train_data.shape)
    # Standardize the features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features_array)
    return features_standardized

mfccs, mfcc_length = wav2mfcc('train', 8000, mfcc_length)
print('mfcc 추출 완료')
padded_mfccs = pad_features(mfccs, mfcc_length)
print('padding 완료')
features = extract_features(padded_mfccs)
print('특징 추출 완료')
features_standardized = standardize_feature(features)


# Apply LDA
labels = np.concatenate((np.zeros(4000), np.ones(4000)))  # Assuming binary labels for the example
lda = LinearDiscriminantAnalysis(n_components=1)
# # Cross-validation
# scores = cross_val_score(lda, features_standardized, labels, cv=5)  # 5-fold cross-validation
# print(f'Cross-validation scores: {scores}')
# print(f'Mean cross-validation score: {np.mean(scores) * 100:.2f}%')

# Fit LDA on the entire dataset
features_lda = lda.fit_transform(features_standardized, labels)
print('LDA 완료')

gmm20240606 = GaussianMixture(n_components=2, covariance_type='full', max_iter=200, random_state=0)
gmm20240606.fit(features_lda)
gmm20240606_labels = gmm20240606.predict(features_lda)

from scipy.stats import mode
from sklearn.metrics import accuracy_score

def map_labels(true_labels, cluster_labels):
    labels = np.zeros_like(cluster_labels)
    for i in range(2):  # 두 개의 클러스터
        mask = (cluster_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

mapped_labels = map_labels(labels, gmm20240606_labels)

# 정확도 확인
accuracy = accuracy_score(labels, mapped_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(features_lda[labels == 0], np.zeros_like(features_lda[labels == 0]) - 0.1, c='r', label='Female (True)', alpha=0.5)
plt.scatter(features_lda[labels == 1], np.ones_like(features_lda[labels == 1]) + 0.1, c='b', label='Male (True)', alpha=0.5)
plt.scatter(features_lda[mapped_labels == 0], np.zeros_like(features_lda[mapped_labels == 0]) - 0.3, marker='x', c='r', label='Female (Pred)', alpha=0.5)
plt.scatter(features_lda[mapped_labels == 1], np.ones_like(features_lda[mapped_labels == 1]) + 0.3, marker='x', c='b', label='Male (Pred)', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])
plt.legend()
plt.title('LDA-transformed Feature Visualization with GMM Clustering')
plt.show()

mfccs, mfcc_length = wav2mfcc('test', 1000, mfcc_length)
padded_mfccs = pad_features(mfccs, mfcc_length)
features = extract_features(padded_mfccs)
features_standardized = standardize_feature(features)

labels = np.concatenate((np.zeros(500), np.ones(500))) 
features_lda = lda.transform(features_standardized)

gmm20240606_labels = gmm20240606.predict(features_lda)

from scipy.stats import mode
from sklearn.metrics import accuracy_score

def map_labels(true_labels, cluster_labels):
    labels = np.zeros_like(cluster_labels)
    for i in range(2):  # 두 개의 클러스터
        mask = (cluster_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

mapped_labels = map_labels(labels, gmm20240606_labels)

# 정확도 확인
accuracy = accuracy_score(labels, mapped_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(features_lda[labels == 0], np.zeros_like(features_lda[labels == 0]) - 0.1, c='r', label='Female (True)', alpha=0.5)
plt.scatter(features_lda[labels == 1], np.ones_like(features_lda[labels == 1]) + 0.1, c='b', label='Male (True)', alpha=0.5)
plt.scatter(features_lda[mapped_labels == 0], np.zeros_like(features_lda[mapped_labels == 0]) - 0.3, marker='x', c='r', label='Female (Pred)', alpha=0.5)
plt.scatter(features_lda[mapped_labels == 1], np.ones_like(features_lda[mapped_labels == 1]) + 0.3, marker='x', c='b', label='Male (Pred)', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])
plt.legend()
plt.title('LDA-transformed Feature Visualization with GMM Clustering')
plt.show()