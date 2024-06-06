import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# 상수 정의
sample_rate = 16000
dtype = np.int16
n_mfcc = 20
n_components_lda = 1
n_components_gmm = 2
max_iter_gmm = 200
random_state_gmm = 42
num_train_samples = 8000
num_test_samples = 1000

def read_path(file):
    return file.readline().strip()

def extract_mfcc(path, sample_rate, dtype, n_mfcc):
    data = np.fromfile(path, dtype=dtype)
    audio = data.astype(np.float32) / np.iinfo(dtype).max
    audio = librosa.effects.preemphasis(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc

def pad_features(features, max_length):
    return [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in features]

def process_data(file_path, base_path, num_samples):
    male, female = [], []
    max_length = 0
    with open(file_path, 'r') as file:
        for _ in range(num_samples):
            path = read_path(file)
            gender = path[0]
            full_path = os.path.join(base_path, f'{path}.raw')
            mfcc = extract_mfcc(full_path, sample_rate, dtype, n_mfcc).T
            if gender == 'F':
                female.append(mfcc)
            else:
                male.append(mfcc)
            max_length = max(max_length, len(mfcc))
    return male, female, max_length

def main():
    # 학습 데이터 처리
    base_path = '202401ml_fmcc/raw16k/train'
    train_file_path = '202401ml_fmcc/fmcc_train.ctl'
    
    male, female, max_length = process_data(train_file_path, base_path, num_train_samples)
    
    female_feature = pad_features(female, max_length)
    male_feature = pad_features(male, max_length)
    
    features = np.concatenate((female_feature, male_feature), axis=0)
    labels = np.concatenate((np.zeros(len(female_feature)), np.ones(len(male_feature))), axis=0)
    
    features_flattened = features.reshape(features.shape[0], -1)
    features_scaled = StandardScaler().fit_transform(features_flattened)
    
    # LDA 변환
    lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
    features_lda = lda.fit_transform(features_scaled, labels)
    
    # GMM 클러스터링
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type='full', max_iter=max_iter_gmm, random_state=random_state_gmm)
    gmm.fit(features_lda)
    gmm_labels = gmm.predict(features_lda)
    
    # Accuracy 계산
    accuracy = accuracy_score(labels, gmm_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # LDA 변환된 특징 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(features_lda[labels == 0], np.zeros_like(features_lda[labels == 0]), c='r', label='Female', alpha=0.5)
    plt.scatter(features_lda[labels == 1], np.ones_like(features_lda[labels == 1]), c='b', label='Male', alpha=0.5)
    plt.xlabel('LDA Feature 1')
    plt.yticks([])
    plt.legend()
    plt.title('LDA-transformed Feature Visualization')
    plt.show()
    
    # 학습 결과 저장
    with open('fmcc_train_result_20240531.txt', 'w') as f:
        for label in gmm_labels:
            f.write('feml\n' if label == 0 else 'male\n')
    
    # 테스트 데이터 처리 및 예측
    test_paths = []
    test_features = []
    
    with open(train_file_path, 'r') as file:
        for _ in range(num_test_samples):
            path = read_path(file)
            full_path = os.path.join(base_path, f'{path}.raw')
            test_paths.append(full_path)
            mfcc = extract_mfcc(full_path, sample_rate, dtype, n_mfcc).T
            test_features.append(mfcc)
    
    test_feature_padded = pad_features(test_features, max_length)
    test_features_flattened = np.array(test_feature_padded).reshape(len(test_feature_padded), -1)
    test_features_scaled = StandardScaler().fit_transform(test_features_flattened)
    test_features_lda = lda.transform(test_features_scaled)
    
    test_labels = gmm.predict(test_features_lda)
    
    # 테스트 결과 저장
    with open('fmcc_test_result_20240531.txt', 'w') as f:
        for i, path in enumerate(test_paths):
            f.write(f'{path} feml\n' if test_labels[i] == 0 else f'{path} male\n')

if __name__ == "__main__":
    main()
