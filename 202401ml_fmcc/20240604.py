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

male = []
female = []
max_length = 0

train = []
train_delta = []
train_deltadelta = []
def Read_Path(f):
    return f.readline().strip()

def MFCC(path):
     # 음성 신호 불러오기
    sr=16000
    audio, sr = librosa.load(path, sr=sr)
    y = librosa.effects.preemphasis(audio, coef=0.97)

    # STFT 계산 (Short-Time Fourier Transform)
    stft = librosa.stft(y)
    magnitude = np.abs(stft)  # Magnitude spectrum

    # Mel 필터 뱅크 적용
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
    mel_spectrogram = np.dot(mel_filter_bank, magnitude)

    # 로그 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram**2)

    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=39, hop_length=128, n_fft=512, fmin=0, fmax=8000)

    # 델타와 델타-델타 계산
    delta_mfccs = librosa.feature.delta(mfcc)
    delta2_mfccs = librosa.feature.delta(mfcc, order=2)

    train_delta.append(delta_mfccs.T)
    train_deltadelta.append(delta2_mfccs.T)

    return mfcc

with open('fmcc_train.ctl', 'r') as f:
    for i in range (8000):
        path = Read_Path(f)
        gender = path[0]
        path = 'raw16k/train/' + path + '.wav'
        mfcc = MFCC(path).T
        train.append(mfcc)
        if len(mfcc) > max_length:
            max_length = len(mfcc)

mfcc_padding = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in train]


feature = []
for i in range(8000):   
    mfcc = mfcc_padding[i].T

    # 델타와 델타-델타 계산
    delta_mfccs = librosa.feature.delta(mfcc)
    delta2_mfccs = librosa.feature.delta(mfcc, order=2)   

    F = np.concatenate([mfcc, delta_mfccs, delta2_mfccs], axis=0)
    feature.append(F.T)

feature1 = np.array(feature)
flatted = feature1.reshape(feature1.shape[0], -1)



gmm20240604feml = BayesianGaussianMixture(n_components=1, covariance_type='diag', max_iter=200, n_init=5, random_state=0)
gmm20240604male = BayesianGaussianMixture(n_components=1, covariance_type='diag', max_iter=200, n_init=5, random_state=0)


female = flatted[0:4000]
male = flatted[4001:8000]

gmm20240604feml.fit(female)
gmm20240604male.fit(male)

test_ret = []

log_likelihood_male = gmm20240604male.score_samples(flatted)
log_likelihood_female = gmm20240604feml.score_samples(flatted)

for i in range(8000):
    # 성별 예측
    if log_likelihood_male[i] > log_likelihood_female[i]:
        test_ret.append(1)
    else:
        test_ret.append(0)

with open('fmcc_train_result_20240604.txt', 'w') as f:
    for i in test_ret:
        f.write(f'{i}\n')
