import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

sample_rate = 16000
dtype = np.int16

male = []
female = []
max_length = 0

def Read_Path(f):
    return f.readline().strip()

def MFCC(path):
    data = np.fromfile(path, dtype=dtype)
    audio = data.astype(np.float32) / np.iinfo(dtype).max
    audio = librosa.effects.preemphasis(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    return mfcc

f = open('fmcc_train.ctl', 'r')
for i in range (8000):
    path = Read_Path(f)
    gender = path[0]
    path = 'raw16k/train/' + path + '.raw'
    temp = MFCC(path).T
    if gender == 'F':
        female.append(temp)
        max_length = max([max_length, len(temp)])
    else:
        male.append(temp)
        max_length = max([max_length, len(temp)])


f.close()

# 모든 벡터를 같은 길이로 맞춘다.
male_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in male]
female_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in female]

# 두 벡터를 결합한다.
feature = np.concatenate((female_feature, male_feature), axis=0)

z = np.zeros(4000)
o = np.ones(4000)

label = np.concatenate((z, o), axis=0)

feature_flatted = feature.reshape(feature.shape[0], -1)
print(feature_flatted)

feature_scaled = StandardScaler().fit_transform(feature_flatted)
print(feature_scaled)
# LDA 변환
lda = LinearDiscriminantAnalysis(n_components=1)
feature_lda = lda.fit_transform(feature_scaled, label)
print(feature_lda)
# LDA 변환된 특징 시각화
plt.figure(figsize=(10, 6))
plt.scatter(feature_lda[label == 0], np.zeros_like(feature_lda[label == 0]), c='r', label='Female', alpha=0.5)
plt.scatter(feature_lda[label == 1], np.ones_like(feature_lda[label == 1]), c='b', label='Male', alpha=0.5)
plt.xlabel('LDA Feature 1')
plt.yticks([])  # y축 눈금을 제거합니다.
plt.legend()
plt.title('LDA-transformed Feature Visualization')
plt.show()


gmm20240531 = GaussianMixture(n_components=2, covariance_type='full', max_iter=200, random_state=42)
gmm20240531.fit(feature_lda)
gmm20240531_labels = gmm20240531.predict(feature_lda)

with open('fmcc_train_result_20240531', 'w') as f1:
    for i in gmm20240531_labels:
        if i == 0:
            f1.write("feml\n")
        else:
            f1.write("male\n")

ret =[]
paths = []

with open('fmcc_train.ctl', 'r') as f2:
    for i in range(1000):
        path = Read_Path(f2)
        path = 'raw16k/train/' + path + '.raw'
        paths.append(path)
        temp = MFCC(path).T
        ret.append(temp)


ret_feature =  [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in ret]
ret_nparray = np.array(ret_feature)
ret_flatted = ret_nparray.reshape(ret_nparray.shape[0], -1)
ret_scaled = StandardScaler().fit_transform(ret_flatted)
ret_lda = lda.transform(ret_scaled)

# print(ret_lda)

ret_label = gmm20240531.predict(ret_lda)
# print(ret_label)

with open('fmcc_test_result_20240531.txt', 'w') as f1:
    for i in range(1000):
        j = ret_label[i]
        if j == 0:
            f1.write(f'{paths[i]} feml\n') 
        else:
            f1.write(f'{paths[i]} male\n')