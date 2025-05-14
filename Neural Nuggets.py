# Step 1: Import Libraries
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