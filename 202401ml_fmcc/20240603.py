import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
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

def Read_Path(f):
    return f.readline().strip()

def MFCC(path):
     # 음성 신호 불러오기
    sr=16000
    audio, sr = librosa.load(path, sr=sr)
    y = librosa.effects.preemphasis(audio, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=75, hop_length=2048, n_fft=8192, fmin=0, fmax=8000)
    return mfcc

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
standardscaler = StandardScaler()
feature_scaled = standardscaler.fit_transform(feature_flatted)

# Select features
#feature_selected = select_features(feature_flatted, labels)

# Train LDA
lda = LinearDiscriminantAnalysis(n_components=1)
feature_lda = lda.fit_transform(feature_scaled, labels)

gmm20240603feml = GaussianMixture(n_components=1, covariance_type='diag', max_iter=200, random_state=0)
gmm20240603male = GaussianMixture(n_components=1, covariance_type='diag', max_iter=200, random_state=0)

gmm20240603feml.fit(feature_lda[0:4000])
gmm20240603male.fit(feature_lda[4000:8000])

test_ret = []

for i in feature_lda:
    i = i.reshape(1, -1)  # 2D 배열로 변환
    log_likelihood_male = gmm20240603male.score(i)
    log_likelihood_female = gmm20240603feml.score(i)

    # 성별 예측
    if log_likelihood_male > log_likelihood_female:
        test_ret.append(1)
    else:
        test_ret.append(0)

with open('fmcc_train_result_20240603.txt', 'w') as f:
    for i in test_ret:
        f.write(i)

ret = []
with open('fmcc_test.ctl', 'r') as f:
    for i in range (1000):
        path = Read_Path(f)
        path = 'raw16k/test/' + path + '.wav'
        temp = MFCC(path).T
        ret.append(temp)

ret_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in ret]

ret_nparray = np.array(ret_feature)

# print(ret_feature)
# print(ret_nparray)
# Flatten and scale features
ret_flatted = ret_nparray.reshape(ret_nparray.shape[0], -1)
ret_scaled = standardscaler.transform(ret_flatted)

ret_lda = lda.transform(ret_scaled)


train_ret = []
for i in ret_lda:
    i = i.reshape(1, -1)  # 2D 배열로 변환
    log_likelihood_male = gmm20240603male.score(i)
    log_likelihood_female = gmm20240603feml.score(i)

    # 성별 예측
    if log_likelihood_male > log_likelihood_female:
        train_ret.append(1)
    else:
        train_ret.append(0)

with open('fmcc_test_result_20240603.txt', 'w') as f:
    for i in train_ret:
        f.write(i)