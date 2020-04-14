import pandas as pd
import pickle
from train_test_classifiers import TrainTestSVM

# Load data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
test_data = pd.read_pickle('../../data/test_data.pkl')

with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# for testing
train_data = train_data[:500]


train_test_tfidf = TrainTestSVM(train_data, test_data, ngram_range=(1,2), max_df=0.5, min_df=0, vector='tfidf',
                                C=0.5, kernel='linear', stopwords=stopwords, title='TF-IDF - SVM')
train_test_tfidf.train_test_save()

train_test_count = TrainTestSVM(train_data, test_data, ngram_range=(1,2), max_df=0.5, min_df=0, vector='count',
                                C=0.5, kernel='linear', stopwords=stopwords, title='Count - SVM')
train_test_count.train_test_save()
