import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import product

# 환경 변수 설정
os.environ["OMP_NUM_THREADS"] = "16"

sample_rate = 16000
mfcc_length = 0
n_mfcc = 76
window_size = 8192
hop_length = 512

def MFCC(path, window_size, hop_length, n_mfcc):
    # 음성 신호 불러오기
    audio, sr = librosa.load(path, sr=sample_rate)
    y = librosa.effects.preemphasis(audio, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length, fmin=0, fmax=800)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, n_fft=window_size, hop_length=hop_length)
    rmse = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)
    feature = np.concatenate([rmse, mfcc, spectral_contrast], axis=0)
    return StandardScaler().fit_transform(feature)

def wav2mfcc(type, cnt, mfcc_length, window_size, hop_length, n_mfcc):
    mfccs = []
    with open('202401ml_fmcc/fmcc_' + type + '.ctl', 'r') as f:
        for i in range(cnt):
            if i % (cnt / 10) == 0:
                print(f'{(i / cnt * 100):.2f}%...')
            path = f.readline().strip()
            path = '202401ml_fmcc/raw16k/' + type + '/' + path + '.wav'
            mfcc = MFCC(path, window_size, hop_length, n_mfcc).T
            mfccs.append(mfcc)
            if len(mfcc) > mfcc_length:
                mfcc_length = len(mfcc)
    return mfccs, mfcc_length

def pad_features(mfccs, mfcc_length):
    padded_mfccs = np.array([np.pad(m, ((0, mfcc_length - m.shape[0]), (0, 0)), mode='constant') for m in mfccs])
    flatted_mfccs = np.reshape(padded_mfccs, (padded_mfccs.shape[0], -1))
    return flatted_mfccs

best_accuracy = 0
best_params = None

mfcc_length = 0
mfccs, mfcc_length = wav2mfcc('train', 8000, mfcc_length, window_size, hop_length, n_mfcc)
print('mfcc 추출 완료')

mfccs = pad_features(mfccs, mfcc_length)
print('전처리 완료')

labels = np.concatenate((np.zeros(4000), np.ones(4000))) 
lda = LinearDiscriminantAnalysis(n_components=1)
lda_feature = lda.fit_transform(mfccs, labels) 
print("LDA 완료")

gmm_feml = GaussianMixture(n_components=3, covariance_type='full', max_iter=200, random_state=0)
gmm_male = GaussianMixture(n_components=3, covariance_type='full', max_iter=200, random_state=0)

feml_feature = lda_feature[0:4000]
male_feature = lda_feature[4000:8000]

gmm_feml.fit(feml_feature)
print('feml 학습 완료')
gmm_male.fit(male_feature)
print('male 학습 완료')
test_ret = []

for i in lda_feature:
    i = i.reshape(1, -1)
    log_likelihood_male = gmm_male.score(i)
    log_likelihood_feml = gmm_feml.score(i)

    if log_likelihood_male > log_likelihood_feml:
        test_ret.append(1)
    else:
        test_ret.append(0)

labels = np.concatenate((np.zeros(4000), np.ones(4000)), axis=0)
accuracy = accuracy_score(labels, test_ret)
print(f'Accuracy with window_size={window_size}, hop_length={hop_length}, n_mfcc={n_mfcc}: {accuracy * 100:.2f}%')
