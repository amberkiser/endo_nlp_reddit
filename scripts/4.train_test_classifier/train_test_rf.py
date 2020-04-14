import pandas as pd
import pickle
from train_test_classifiers import TrainTestRF

# Load data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
test_data = pd.read_pickle('../../data/test_data.pkl')

with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# for testing
train_data = train_data[:500]


train_test_tfidf = TrainTestRF(train_data, test_data, ngram_range=(1,2), max_df=0.5, min_df=0, vector='tfidf',
                               n_estimators=100, criterion='gini', max_depth=10, stopwords=stopwords,
                               title='TF-IDF - RF')
train_test_tfidf.train_test_save()

train_test_count = TrainTestRF(train_data, test_data, ngram_range=(1,2), max_df=0.5, min_df=0, vector='count',
                               n_estimators=100, criterion='gini', max_depth=10, stopwords=stopwords,
                               title='Count - RF')
train_test_count.train_test_save()
