import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import noisereduce as nr
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sample_rate = 16000
window_size = 8192
hop_length = 2048
mfcc_length = 0
energy_length = 0
n_mfcc = 66
# Spectral Subtraction을 사용한 노이즈 제거 함수
def spectral_subtraction(audio, noise_threshold=1.0):
    stft_audio = librosa.stft(audio, n_fft=400, hop_length=160)  # Short-time Fourier Transform (STFT) 계산
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
    # y_clean = spectral_subtraction(y)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length, fmin=0, fmax=800)   
    return StandardScaler().fit_transform(mfcc)

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
    padded_mfccs = np.array([np.pad(m, ((0, mfcc_length - m.shape[0]), (0, 0)), mode='constant') for m in mfccs])
    flatted_mfccs = np.reshape(padded_mfccs, (padded_mfccs.shape[0], -1))
    return flatted_mfccs

mfccs, mfcc_length = wav2mfcc('train', 8000, mfcc_length)
print('mfcc 추출 완료')

mfccs = pad_features(mfccs, mfcc_length)
print('전처리 완료')

# Apply LDA
labels = np.concatenate((np.zeros(4000), np.ones(4000)))  # Assuming binary labels for the example
lda = LinearDiscriminantAnalysis(n_components=1)
lda_featrue = lda.fit_transform(mfccs, labels) 
print("LDA 완료")

gmm_feml = GaussianMixture(n_components=3, covariance_type='full', max_iter=300, random_state=0)
gmm_male = GaussianMixture(n_components=3, covariance_type='full', max_iter=300, random_state=0)

feml_feature = lda_featrue[0:4000]
male_feature = lda_featrue[4000:8000]

gmm_feml.fit(feml_feature)
print('feml 학습 완료')
gmm_male.fit(male_feature)
print('male 학습 완료')
test_ret = []

for i in lda_featrue:
    i = i.reshape(1, -1)  # 2D 배열로 변환
    log_likelihood_male = gmm_male.score(i)
    log_likelihood_feml = gmm_feml.score(i)

    # 성별 예측
    if log_likelihood_male > log_likelihood_feml:
        test_ret.append(1)
    else:
        test_ret.append(0)

with open('fmcc_train_result_20240606.txt', 'w') as f:
    for i in test_ret:
        f.write(f'{i}\n')

#####################################################################################

mfccs, mfcc_length = wav2mfcc('test', 1000, mfcc_length)
print('mfcc 추출 완료')

mfccs = pad_features(mfccs, mfcc_length)
print('전처리 완료')

lda_featrue = lda.transform(mfccs)
print("LDA 완료")

test_ret = []

for i in lda_featrue:
    i = i.reshape(1, -1)  # 2D 배열로 변환
    log_likelihood_male = gmm_male.score(i)
    log_likelihood_feml = gmm_feml.score(i)

    # 성별 예측
    if log_likelihood_male > log_likelihood_feml:
        test_ret.append(1)
    else:
        test_ret.append(0)

with open('fmcc_test_result_20240606.txt', 'w') as f:
    for i in test_ret:
        f.write(f'{i}\n')

################################################333

from scipy.stats import mode
from sklearn.metrics import accuracy_score

def map_labels(true_labels, cluster_labels):
    labels = np.zeros_like(cluster_labels)
    for i in range(2):  # 두 개의 클러스터
        mask = (cluster_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

labels = np.concatenate((np.zeros(500), np.ones(500)), axis=0)
mapped_labels = map_labels(labels, np.array(test_ret))

# 정확도 확인
accuracy = accuracy_score(labels, mapped_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(lda_featrue[labels == 0], np.zeros_like(lda_featrue[labels == 0]) - 0.1, c='r', label='Female (True)', alpha=0.5)
plt.scatter(lda_featrue[labels == 1], np.ones_like(lda_featrue[labels == 1]) + 0.1, c='b', label='Male (True)', alpha=0.5)
plt.scatter(lda_featrue[mapped_labels == 0], np.zeros_like(lda_featrue[mapped_labels == 0]) - 0.3, marker='x', c='r', label='Female (Pred)', alpha=0.5)
plt.scatter(lda_featrue[mapped_labels == 1], np.ones_like(lda_featrue[mapped_labels == 1]) + 0.3, marker='x', c='b', label='Male (Pred)', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])
plt.legend()
plt.title('LDA-transformed Feature Visualization with GMM Clustering')
plt.show()
