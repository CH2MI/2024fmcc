import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

sample_rate = 16000

male = []
female = []
max_length = 0

def Read_Path(f):
    return f.readline().strip()

def extract_features(path):
    audio, sr = librosa.load(path, sr=sample_rate)
    max_amp = np.max(np.abs(audio))
    scaled_audio = audio / max_amp

    # MFCC
    mfcc = librosa.feature.mfcc(y=scaled_audio, sr=sr, n_mfcc=20).T
    mel_spec = librosa.feature.melspectrogram(y=scaled_audio, sr=sr).T
    chroma = librosa.feature.chroma_stft(y=scaled_audio, sr=sr).T

    max_len = max(mfcc.shape[0], mel_spec.shape[0], chroma.shape[0])
    mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')
    mel_spec = np.pad(mel_spec, ((0, max_len - mel_spec.shape[0]), (0, 0)), mode='constant')
    chroma = np.pad(chroma, ((0, max_len - chroma.shape[0]), (0, 0)), mode='constant')
    
    features = np.hstack([mfcc, mel_spec, chroma])
    return features

with open('202401ml_fmcc/fmcc_train.ctl', 'r') as f:
    for i in range(4000):  # 데이터 양 줄이기 (예: 4000개로 축소)
        path = Read_Path(f)
        gender = path[0]
        path = '202401ml_fmcc/raw16k/train/' + path + '.wav'
        temp = extract_features(path)
        if gender == 'F':
            female.append(temp)
            max_length = max(max_length, temp.shape[0])
        else:
            male.append(temp)
            max_length = max(max_length, temp.shape[0])

female_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in female]
male_feature = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant') for arr in male]

feature = np.concatenate((female_feature, male_feature), axis=0)
labels = np.concatenate((np.zeros(len(female_feature)), np.ones(len(male_feature))), axis=0)

feature_flatted = feature.reshape(feature.shape[0], -1)
feature_scaled = StandardScaler().fit_transform(feature_flatted)

selector = SelectKBest(score_func=f_classif, k=300)  # 특성 개수 줄이기
feature_selected = selector.fit_transform(feature_scaled, labels)

pca = PCA(n_components=50)  # PCA 차원 축소 (50개로 축소)
feature_pca = pca.fit_transform(feature_selected)

X_train, X_test, y_train, y_test = train_test_split(feature_pca, labels, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluation
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

joblib.dump(best_rf, 'best_rf_model_fast.pkl')

importances = best_rf.feature_importances_  # 최적 모델에서 특성 중요도 추출
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
