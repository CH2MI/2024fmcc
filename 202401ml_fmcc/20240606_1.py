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


sample_rate = 16000
window_size = 400
hop_length = 160
mfcc_length = 0
energy_length = 0
n_mfcc = 75
# Spectral Subtraction을 사용한 노이즈 제거 함수
def spectral_subtraction(audio, noise_threshold=1.0):
    stft_audio = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)  # Short-time Fourier Transform (STFT) 계산
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
    y_clean = spectral_subtraction(y)
    # STFT 계산 (Short-Time Fourier Transform)
    stft = librosa.stft(y_clean, n_fft=window_size, hop_length=hop_length)
    magnitude = np.abs(stft)  # Magnitude spectrum

    # Mel 필터 뱅크 적용
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=window_size, n_mels=128)
    mel_spectrogram = np.dot(mel_filter_bank, magnitude)

    # 로그 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length, dct_type=3, lifter=23)
    return mfcc

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
    padded_mfccs = [np.pad(m, ((0, mfcc_length - m.shape[0]), (0, 0)), mode='constant') for m in mfccs]
    return padded_mfccs

def extract_features(padded_mfccs,):
    features = []
    for mfcc in padded_mfccs:
        scaler = StandardScaler()   
        mfcc_standardized = scaler.fit_transform(mfcc.T)
        delta_mfcc = librosa.feature.delta(mfcc_standardized)
        double_delta_mfcc = librosa.feature.delta(mfcc_standardized, order=2)
        combined_features = np.concatenate([mfcc_standardized, delta_mfcc, double_delta_mfcc], axis=0)
        features.append(combined_features.T)
    return np.array(features)

mfccs, mfcc_length = wav2mfcc('train', 8000, mfcc_length)
print('mfcc 추출 완료')

padded_mfccs = pad_features(mfccs, mfcc_length)
print('padding 완료')

features = extract_features(padded_mfccs)
print('특징 추출 완료')

feml = features[0:4000]
male = features[4000:8000]

feature_feml = np.reshape(feml, (feml.shape[0] * feml.shape[1], feml.shape[2]))
feature_male = np.reshape(male, (male.shape[0] * male.shape[1], male.shape[2]))
print('특징 생성 완료')
gmm_feml = GaussianMixture(n_components=3, max_iter=200, random_state=0)
gmm_male = GaussianMixture(n_components=3, max_iter=200, random_state=0)

gmm_feml.fit(feature_feml)  
print('feml 특징 학습 완료')
gmm_male.fit(feature_male)
print('male 특징 학습 완료')

result = []
with open('20240606_1_train_data.txt', 'w') as f:

    for i in features:
        
        likelihood_feml = gmm_feml.score(i)
        likelihood_male = gmm_male.score(i)

        f.write(f'{i} :\nfeml :\n{likelihood_feml}\nmale :\n{likelihood_male}\n\n')

        if likelihood_feml > likelihood_male:
            result.append(0)
        else:
            result.append(1)

with open('20240606_1_train_result.txt', 'w') as f:
    for i in result:
        f.write(f'{i}\n')