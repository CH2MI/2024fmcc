import librosa
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import wave

sample_rate = 16000
window_size = 10000
hop_length = 2000
mfcc_length = 0 
n_mfcc = 76 
save_path =[]

# raw파일을 wav 파일로 변환한다.
def raw2wav(path):
    sample_width = 2  
    sample_rate = 16000 
    num_channels = 1  

    with open(path + ".raw", 'rb') as raw_file:
        raw_data = raw_file.read()
    
    with wave.open(path + ".wav", 'wb') as wav_file:
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.setnchannels(num_channels)
        wav_file.writeframes(raw_data)

# 음성 특징을 추출하고, 결합한다.
def MFCC(path):
    audio, sr = librosa.load(path, sr=sample_rate)
    y = librosa.effects.preemphasis(audio, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length, fmin=0, fmax=800)   
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, n_fft=window_size, hop_length=hop_length)
    rmse = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)
    feature = np.concatenate([rmse, mfcc, spectral_contrast], axis=0)
    return StandardScaler().fit_transform(feature)
 
# 경로를 읽어서 raw2wav를 한 뒤 특징을 추출해 반환한다. 
def wav2mfcc(type, cnt, mfcc_length):
    mfccs = []
    with open('fmcc_' + type + '.ctl', 'r') as f:
        for i in range (cnt):
            path = f.readline().strip()
            path = 'raw16k/' + type + '/' + path
            raw2wav(path)
            save_path.append(path + '.raw')
            mfcc = MFCC(path + '.wav').T
            mfccs.append(mfcc)
            if len(mfcc) > mfcc_length:
                mfcc_length = len(mfcc)
    return mfccs, mfcc_length

# 특징을 패딩하고 2차원으로 만든다.
def pad_features(mfccs, mfcc_length):
    padded_mfccs = np.array([np.pad(m, ((0, mfcc_length - m.shape[0]), (0, 0)), mode='constant') for m in mfccs])
    flatted_mfccs = np.reshape(padded_mfccs, (padded_mfccs.shape[0], -1))
    return flatted_mfccs

# 모델을 읽어온다.
loaded_models_dict = joblib.load('동해둘_models.pkl')

lda = loaded_models_dict['lda']
gmm_feml = loaded_models_dict['gmm_feml']
gmm_male = loaded_models_dict['gmm_male']
mfcc_length = loaded_models_dict['mfcc_length']

mfccs, mfcc_length = wav2mfcc('eval', 1000, mfcc_length)
print('mfcc 추출 완료')

mfccs = pad_features(mfccs, mfcc_length)
print('전처리 완료')

# LDA를 이용해 두 그룹간 거리를 최대화 한다.
lda_feature = lda.transform(mfccs)
print("LDA 완료")

with open('동해둘_eval_results.txt', 'w', newline='\n') as f:
    for path, i in zip(save_path, lda_feature):
        i = i.reshape(1, -1)
        log_likelihood_male = gmm_male.score(i)
        log_likelihood_feml = gmm_feml.score(i)

        # 성별 예측
        if log_likelihood_male > log_likelihood_feml:
            f.write(path + ' male\n')
        else:
            f.write(path + ' feml\n')
            
        