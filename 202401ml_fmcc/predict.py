import numpy as np
import joblib
import librosa

def Read_Path(f):
    return f.readline().strip()

def predict_gender(audio_path, sr=16000, dtype=np.int16):
    # GMM 모델 로드
    gmm_male = joblib.load('gmm_male_model.pkl')
    gmm_female = joblib.load('gmm_female_model.pkl')
    
    # 새로운 오디오 파일 읽기
    data = np.fromfile(audio_path, dtype=dtype)
    audio = data.astype(np.float32) / np.iinfo(dtype).max
    
    # MFCC 특징 추출
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features = mfcc.T
    
    # 각 GMM 모델에 대한 로그-우도 계산
    log_likelihood_male = gmm_male.score(features)
    log_likelihood_female = gmm_female.score(features)
    
    # 성별 예측
    if log_likelihood_male > log_likelihood_female:
        return 'male'
    else:
        return 'feml'


f = open('fmcc_test.ctl', 'r')
f2 = open('fmcc_result.txt', 'w')


for i in range(1000):
    path1 = Read_Path(f)
    path = 'raw16k/test/' + path1 + '.raw'
    result = path1 + ' ' + predict_gender(path) + '\n'
    f2.write(result)
    
f.close()
f2.close()
    