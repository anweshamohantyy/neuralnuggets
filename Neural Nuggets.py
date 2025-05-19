# Importing Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

# Kaggle authentication
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Download TESS dataset
os.system("kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess")
os.system("unzip toronto-emotional-speech-set-tess.zip")
Tess = os.path.join(os.getcwd(), "TESS Toronto emotional speech set data")

# Sort the emotions in the dataset
emotion = []
path = []
tess_directory_list = os.listdir(Tess)
for dir in tess_directory_list:
    directories = os.listdir(os.path.join(Tess, dir))
    for file in directories:
        part = file.split('.')[0]                   # Splitting the file whenever it sees a period (ex: removing .wav)
        part = part.split('_')[2]                   # Splitting after second underscore to classify the emotion
        if part=='ps':
            emotion.append('surprise')              # label ps as surprise
        else:
            emotion.append(part)
        path.append(os.path.join(Tess, dir, file))  # use os.path.join for correct file paths

# Create a dataframe
tess_df = pd.DataFrame()
tess_df["path"] = path        # creating a column called path
tess_df["emotion"] = emotion  # creating a column called emotion
tess_df.head()

# Create the waveplot function to display the amplitude of an audio signal
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr, color='pink')
    plt.show()

# Create the spectogram function to display the frequency and amplitude over time
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz', cmap='spring')
    plt.colorbar()
    plt.show()

# Emotion Sample (Fear)
emotion = 'fear'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Anger)
emotion = 'angry'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Disgust)
emotion = 'disgust'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Neutral)
emotion = 'neutral'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Happy)
emotion = 'happy'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Sad)
emotion = 'sad'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Emotion Sample (Pleasant Surprise)
emotion = 'surprise'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]
data, sr = librosa.load(path)
waveplot(data, sr, emotion)
spectogram(data, sr, emotion)
Audio(path)

# Feature extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
extract_mfcc(tess_df["path"][0])

# Processes every file in the 'path' column and creates a list of MFCC features for all
X_mfcc = tess_df["path"].apply(lambda x: extract_mfcc(x))
X = [x for x in X_mfcc]
X = np.array(X)
X.shape
X = np.expand_dims(X, -1)
X.shape

# Converts categorical labels into binary vectors
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(tess_df[['emotion']])        # Changes the emotion labels into understandable numbers for model to read
y = y.toarray()
y.shape

# Create Long Short-Term Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Creates neural network model where layers are stacked in order
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

# Loss = Measures how wrong predictions are. Optimizer = Tries to correct mistakes. Metrics = % of correct predictions
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Prints a summary of model with Layer Types, Output Shapes, and Param #
model.summary()

# Trains the model for 50 rounds by checking the accuracy using 20% of the data
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

# Training accuracy and validation accuracy, seeing model's performance
epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the Results
plt.title('Training vs Validation Accuracy')
plt.plot(epochs, acc, label='train accuracy', color='orchid')
plt.plot(epochs, val_acc, label='val accuracy', color='lightblue')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Training and validation loss which measure how wrong the models predictions are
# Visualize loss changing over time
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()