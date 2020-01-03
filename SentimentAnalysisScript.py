%reset -f
import pandas as pd
import numpy as np

dt = pd.read_csv(r'C:\Users\jayasans4085\Desktop\HR Exit Interview\Exit Interview Data - 14.10.2019 pseudonimised.csv',
                 low_memory = False,encoding = 'ISO 8859-1')

xpnd = pd.DataFrame(dt['Would you like to expand further on your reasons for leaving?'])
xpndt = xpnd.copy()

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

# Load Regular Ex
import re

def Remove_Regex(x):
    if pd.notna(x):
        return re.sub('[,\.!?]', '', x)
    else:
        return np.nan

xpndt.columns  = ['Text']
xpndt.iloc[:,0] = xpndt.iloc[:,0].map(lambda x: Remove_Regex(x))

from wordcloud import WordCloud
import matplotlib as plt
# Join the different processed titles together.
long_string = ','.join(list(xpndt['Text'][~xpndt['Text'].isna()].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(xpndt['Text'][~xpndt['Text'].isna()])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)
