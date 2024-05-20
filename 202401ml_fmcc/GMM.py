import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

sample_rate = 16000
dtype = np.int16

male = []
female = []

def Read_Path(f):
    return f.readline().strip()

def MFCC(path, gender):
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

    librosa.display.specshow(mfcc, sample_rate)

    if gender == 'M':
        male.append(mfcc.T)
    else:
        female.append(mfcc.T)

    

    


f = open('fmcc_train.ctl', 'r')

for i in range (8000):
    path = Read_Path(f)
    if i != 0: break
    gender = path[0]
    print(gender)
    path = 'raw16k/train/' + path + '.raw'
    MFCC(path, gender)


male = np.vstack(male)
female = np.vstack(female)

# GMM 모델 생성 및 학습
n_components = 16  # GMM의 컴포넌트 수
gmm_male = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200, random_state=42)
gmm_female = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200, random_state=42)

# 각 성별 데이터로 GMM 학습
gmm_male.fit(male)
gmm_female.fit(female)

joblib.dump(gmm_male, 'gmm_male_model.pkl')
joblib.dump(gmm_female, 'gmm_female_model.pkl')


f.close()

