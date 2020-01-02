%reset -f
import pandas as pd
import numpy as np

dt = pd.read_csv(r'C:\Users\jayasans4085\Desktop\HR Exit Interview\Exit Interview Data - 14.10.2019 pseudonimised.csv',
                 low_memory = False,encoding = 'ISO 8859-1')

xpnd = pd.DataFrame(dt['Would you like to expand further on your reasons for leaving?'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def Polarity(x):
    if pd.notna(x):
        sol = analyser.polarity_scores(x)
        return [sol[i] for i in sol]
    else:
        return [np.nan,np.nan,np.nan,np.nan]

for i in ['Negative','Neutral','Positive','Compound']:
    xpnd[i]=0

for i in list(range(0,xpnd.shape[0])):
    print(i)
    xpnd.iloc[i,1:] = Polarity(xpnd.iloc[i,0])

count = sum(xpnd.iloc[:,4]<0)

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
