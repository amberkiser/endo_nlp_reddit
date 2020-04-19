import pandas as pd
import pickle
from train_test_classifiers import TrainTestRF


train_data = pd.read_pickle('../../data/train_data.pkl')
test_data = pd.read_pickle('../../data/test_data.pkl')

with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)


rf_count_params = {
    'ngram_range': (1,1),
    'max_df': 1.0,
    'min_df': 0,
    'n_estimators': 300,
    'criterion': 'gini',
    'max_depth': 100
}
train_test_count = TrainTestRF(train_data, test_data, ngram_range=rf_count_params['ngram_range'],
                               max_df=rf_count_params['max_df'], min_df=rf_count_params['min_df'], vector='count',
                               n_estimators=rf_count_params['n_estimators'], criterion=rf_count_params['criterion'],
                               max_depth=rf_count_params['max_depth'], stopwords=stopwords, title='Count - RF')
train_test_count.train_test_save()


rf_tfidf_params = {
    'ngram_range': (1,1),
    'max_df': 0.5,
    'min_df': 0,
    'n_estimators': 300,
    'criterion': 'entropy',
    'max_depth': 100
}
train_test_tfidf = TrainTestRF(train_data, test_data, ngram_range=rf_tfidf_params['ngram_range'],
                               max_df=rf_tfidf_params['max_df'], min_df=rf_tfidf_params['min_df'], vector='tfidf',
                               n_estimators=rf_tfidf_params['n_estimators'], criterion=rf_tfidf_params['criterion'],
                               max_depth=rf_tfidf_params['max_depth'], stopwords=stopwords, title='TF-IDF - RF')
train_test_tfidf.train_test_save()
