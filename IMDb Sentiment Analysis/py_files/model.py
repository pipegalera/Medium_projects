# @Author: pipegalera
# @Date:   2020-04-15T18:39:19+02:00
# @Last modified by:   pipegalera
# @Last modified time: 2020-04-15T19:30:46+02:00


# Packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os

# Read data
os.chdir("/Users/pipegalera/Documents/GitHub/side_projects/IMDb Sentiment Analysis")
df = pd.read_csv('raw_data/movie_data.csv', encoding = 'utf-8')
df.head()
df.shape

# Transforming the words into feature vectors

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet,'
    'and one and one is two'
    ])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
