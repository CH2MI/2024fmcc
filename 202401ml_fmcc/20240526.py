import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

sample_rate = 16000
dtype = np.int16
feature = []

def Read_Path(f):
    return f.readline().strip()

def MFCC(path):
    # .raw 파일 읽기
    data = np.fromfile(path, dtype=dtype)

    # 데이터를 librosa로 변환 (정규화 필요)
    audio = data.astype(np.float32) / np.iinfo(dtype).max

    # librosa를 사용하여 처리
    # 예: 오디오의 멜-스펙트로그램 계산
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

    # 멜-스펙트로그램을 로그 스케일로 변환
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=20)

    return mfcc
    

f = open('202401ml_fmcc/fmcc_train.ctl', 'r')
for i in range (8000):
    path = Read_Path(f)
    path = '202401ml_fmcc/raw16k/train/' + path + '.raw'
    temp = MFCC(path).T
    feature.append(temp)

feature = np.vstack(feature)
print("Total feature shape:", feature.shape)
gmm20240526 = GaussianMixture(n_components=2, covariance_type='spherical', max_iter=200)
gmm20240526.fit(feature)
gmm20240526_labels = gmm20240526.predict(feature)

# fig = plt.figure(figsize=(8,8))
# fig.set_facecolor('white')
# plt.scatter(feature[:,0], feature[:,1], c=gmm20240526_labels)
# plt.show()

f1 = open('fmcc_train_result', 'w') 
print(len(gmm20240526_labels))
for i in gmm20240526_labels:
    if i == 0:
        f1.write("male\n")
    else:
        f1.write("feml\n")



f.close()
f1.close()

# f = open('fmcc_test.ctl', 'r')

# TTTTT = []

# for i in range (1000):
#     path = Read_Path(f)
#     path = 'raw16k/test/' + path + '.raw'

#     temp = MFCC(path).T
#     data = np.mean(temp, axis=0)
#     TTTTT.append(data)

# f.close()

# TTTTT = np.stack(TTTTT)
# labelssss = gmm20240526.predict(TTTTT)

# f = open('fmcc_test.ctl', 'r')
# f2 = open('fmcc_result.txt', 'w')

# for i in range (1000):
#     path = Read_Path(f)
#     path = 'raw16k/test/' + path + '.raw'
#     c = 'feml'
#     if labelssss[i] == 0:
#         c = 'male'
#     f2.write(path + ' ' + c + '\n')
# f.close()
# f2.close()

