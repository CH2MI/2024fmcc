import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import wave

def raw2wav(path):
    sample_width = 2  # 16비트 샘플
    sample_rate = 16000  # 샘플링 속도
    num_channels = 1  # 모노 오디오

    with open(path + ".raw", 'rb') as raw_file:
        raw_data = raw_file.read()
    
    with wave.open(path + ".wav", 'wb') as wav_file:
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.setnchannels(num_channels)

        wav_file.writeframes(raw_data)
    
    return path + ".wav"

def Read_Path(f):
    return f.readline().strip()


f = open('machine_learning/202401ml_fmcc/fmcc_train.ctl', 'r')

# path = Read_Path(f)
# path = 'machine_learning/202401ml_fmcc/raw16k/train/' + path
# path = raw2wav(path)

for i in range (8000):
    path = Read_Path(f)
    gender = path[0]
    path = 'machine_learning/202401ml_fmcc/raw16k/train/' + path
    path = raw2wav(path)
print(path)
y, sr = librosa.load(path)
ft = librosa.stft(y)
db = librosa.amplitude_to_db(np.abs(ft), ref=np.max)
librosa.display.specshow(db, x_axis='time', y_axis='linear')
plt.show()