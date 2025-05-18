# Importing Libraries 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import keras
import kaggle
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

# Kaggle authentication
os.system("mkdir ~/.kaggle")
os.system("touch ~/.kaggle/kaggle.json")
api_token = {"username":"sanjanapremkumar","key":"e2acda3c12ec484d98b029a64286a8c6"}
import json
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)
os.system("chmod 600 ~/.kaggle/kaggle.json")

# Download TESS dataset
os.system("kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess")
os.system("unzip /content/toronto-emotional-speech-set-tess.zip")
Tess = r"/content/TESS Toronto emotional speech set data"

# Sort the emotions in the dataset
emotion = []
path = []
tess_directory_list = os.listdir(Tess)
for dir in tess_directory_list:
    directories = os.listdir(Tess +'/'+ dir)
    for file in directories:
        part = file.split('.')[0]                   #Splitting the file whenever it sees a period (ex: removing .wav)
        part = part.split('_')[2]                   #Splitting after second underscore to clasify the emotion 
        if part=='ps':
            emotion.append('surprise')     #label ps as suprise
        else:
            emotion.append(part)
        path.append(Tess + dir + '/' + file)
        
# Create a dataframe
tess_df = pd.DataFrame()
tess_df["path"] = path        #creating a column called path
tess_df["emotion"] = emotion          #creating a column called emotion
tess_df.head()

# Create the waveplot function to display the amplitude of an audio signal
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))              #Creates chart size
    plt.title(emotion, size=20)             #Sets title of plot to the specified emotion   
    librosa.display.waveshow(data, sr=sr, color='pink')     #Adds the waveform to a plot
    plt.show()      #Renders the plot

# Create the spectogram function to display the frequency and amplitude over time
def spectogram(data, sr, emotion):
    x = librosa.stft(data)          #transforms data into time-frequency domain for spectogram use
    xdb = librosa.amplitude_to_db(abs(x))   #transforms data to decibells(dB)
    plt.figure(figsize=(11,4))              #Sets the size of the chart
    plt.title(emotion, size=20)             #Sets the title of the plot to the specified emotion
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz', cmap = 'spring')   #Makes 2D array of decibel-scaled amplitudes
    plt.colorbar()      #Adds a colorbar: Mapping between color intensity and dB values
    plt.show()      #Renders the plot
       
# Emotion Sample (Fear)
emotion = 'fear'
path = np.array(tess_df["path"][tess_df["emotion"] == emotion])[0]   # line gives you the file path to one 'fear' audio sample
data, sr = librosa.load(path)   #Loads the audio signal so it can be visualized or used in processing.
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
    y, sr = librosa.load(filename, duration=3, offset=0.5)      #Loads 3 seconds of audio, starting at 0.5 seconds
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)      #Turns audio into numbers for machine to learn
    return mfcc
extract_mfcc(tess_df["path"][0])    #applies the fuction to the first audio file

# Processes every file in the 'path' column and creates a list of MFCC features for all
X_mfcc = tess_df["path"].apply (lambda x: extract_mfcc(x))      #Applies the extract function to every file path
X_mfcc

X = [x for x in X_mfcc]     #Loops through all MFCC and converts into a regular python list 
X = np.array(X)         #Converts the list into a 2d num array
X.shape             

X = np.expand_dims(X, -1)       #Adds a new dimension to make the shape (2800, 40, 1)
X.shape

# Converts categorical labels into binary vectors
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()  # Each emotion is turned into a vector of 0s and 1s
y = encoder.fit_transform(tess_df[['path']])        #Changes the emotion labels into understandable numbers for model to read
y = y.toarray()
y.shape

# Create Long Short-Term Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Creates neural network model where layers are stacked in order
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),      #256 memory units, gives final summary only
    Dropout(0.2),       #Ignores 20% of memory to prevent overfitting (memorizing data)
    Dense(128, activation='relu'),  #Reduces neurons to 128 (helps model focus on key characteristics). Uses ReLU function (kills weak signals [negative values])
    Dropout(0.2),
    Dense(64, activation='relu'),   #Reduces neurons to 64. Uses ReLU function (kills weak signals [negative values])
    Dropout(0.2),       
    Dense(7, activation='softmax')  #Final layer: 7 emotions. Uses Softmax to convert outputs to probabilities that add up to 100%
])

# Loss = Measures how wrong predictions are. Optimizer = Tries to correct mistakes. Metrics = % of correct predictions
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Prints a summary of model with Layer Types, Output Shapes, and Param #
model.summary()

# Trains the model for 50 rounds by checking the accuracy using 20% of the data 
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)  

#Training accuracy and validation accuracy, seeing model's performance
epochs = list(range(50))  # Sets up the x-axis values for the accuracy plot
acc = history.history['accuracy'] #stores the training accuracy so it can be plotted in a graph
val_acc = history.history['val_accuracy'] #stores the validation accuracy so it can be plotted in a graph

# Plot the Results
plt.title('Training vs Validation Accuracy')
plt.plot(epochs, acc, label='train accuracy', color='orchid')
plt.plot(epochs, val_acc, label='val accuracy', color='lightblue')
plt.xlabel('epochs')    #Adds axis labels
plt.ylabel('accuracy')
plt.legend()            #Adds a legend 
plt.show()              #Displays plot

#Training and validation loss which measure how wrong the models predictions are
# Visualize loss changing over time
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, label='train loss')      #Plots two lines: training loss and validation loss against the number of epochs
plt.plot(epochs, val_loss, label='val loss')       #epochs is a list or array representing the numbers
plt.xlabel('epochs')        #Adds axis labels
plt.ylabel('loss')         
plt.legend()                #Adds a legend 
plt.show()                  #Displays plot
