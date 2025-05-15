# Importing Libraries 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
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
    plt.figure(digsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz', cmap = 'spring')
    plt.colorbar()
    plt.show()
       
    
# Emotion Sample (Fear)
emotion = 'fear'
path np.array(df['path'][df[emotion]])[0]
data,