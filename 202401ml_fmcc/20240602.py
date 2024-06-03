import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

sample_rate = 16000
dtype = np.int16

male = []
female = []
max_length = 0

def Read_Path(f):
    return f.readline().strip()

def MFCC(path):
     # 음성 신호 불러오기
    sr=16000
    audio, sr = librosa.load(path, sr=sr)
    n_fft = 8192  # 원하는 FFT 윈도우 크기
    hop_length=4096
    n_mfcc=75
    fmin=0
    fmax=8000
    top_db=90

    # 스케일 조정
    max_amp = np.max(np.abs(audio))
    scale_factor = 1.0 / max_amp
    scaled_audio = audio * scale_factor
     
    # 스펙트로그램 평활화
    norm_audio = librosa.util.normalize(scaled_audio)
    
    # RMS 정규화
    rms = librosa.feature.rms(y=norm_audio)
    normalized_audio = norm_audio / (np.max(rms) + 1e-8)
  
    # MFCC 추출
    mfcc = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, fmin=fmin, fmax=fmax)
    
    return mfcc

# 특성 선택 함수
def select_features(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

with open('fmcc_train.ctl', 'r') as f:
    for i in range (8000):
        path = Read_Path(f)
        gender = path[0]
        path = 'raw16k/train/' + path + '.wav'
        temp = MFCC(path).T
        if gender == 'F':
            female.append(temp)
            max_length = max([max_length, len(temp)])
        else:
            male.append(temp)
            max_length = max([max_length, len(temp)])

female_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in female]
male_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in male]

# Combine features and labels
feature = np.concatenate((female_feature, male_feature), axis=0)
labels = np.concatenate((np.zeros(len(female_feature)), np.ones(len(male_feature))), axis=0)

# Flatten and scale features
feature_flatted = feature.reshape(feature.shape[0], -1)
feature_scaled = StandardScaler().fit_transform(feature_flatted)

# Select features
#feature_selected = select_features(feature_flatted, labels)

# Train LDA
lda = LinearDiscriminantAnalysis(n_components=1)
feature_lda = lda.fit_transform(feature_scaled, labels)

# # Visualize LDA-transformed features
# plt.figure(figsize=(10, 6))
# plt.scatter(feature_lda[labels == 0], np.zeros_like(feature_lda[labels == 0]), c='r', label='Female', alpha=0.5)
# plt.scatter(feature_lda[labels == 1], np.ones_like(feature_lda[labels == 1]), c='b', label='Male', alpha=0.5)
# plt.xlabel('LDA Feature 1')
# plt.yticks([])  # y축 눈금을 제거합니다.
# plt.legend()
# plt.title('LDA-transformed Feature Visualization')
# plt.show()

gmm20240601 = GaussianMixture(n_components=2, covariance_type='diag', max_iter=200, random_state=0)
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