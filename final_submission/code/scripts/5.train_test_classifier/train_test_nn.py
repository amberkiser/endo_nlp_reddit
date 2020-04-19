import pandas as pd
import pickle
from train_test_classifiers import TrainTestNN


train_data = pd.read_pickle('../../data/train_data.pkl')
test_data = pd.read_pickle('../../data/test_data.pkl')

with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)


nn_count_params = {
    'ngram_range': (1,2),
    'max_df': 1.0,
    'min_df': 0,
    'h1_nodes': 128,
    'optimizer': 'Adadelta'
}

train_test_count = TrainTestNN(train_data, test_data, ngram_range=nn_count_params['ngram_range'],
                               max_df=nn_count_params['max_df'], min_df=nn_count_params['min_df'], vector='count',
                               h1_nodes=nn_count_params['h1_nodes'], optimizer=nn_count_params['optimizer'],
                               stopwords=stopwords, title='Count - NN')
train_test_count.train_test_save()


nn_tfidf_params = {
    'ngram_range': (1,2),
    'max_df': 0.75,
    'min_df': 0,
    'h1_nodes': 128,
    'optimizer': 'Adadelta'
}

train_test_tfidf = TrainTestNN(train_data, test_data, ngram_range=nn_tfidf_params['ngram_range'],
                               max_df=nn_tfidf_params['max_df'], min_df=nn_tfidf_params['min_df'], vector='tfidf',
                               h1_nodes=nn_tfidf_params['h1_nodes'], optimizer=nn_tfidf_params['optimizer'],
                               stopwords=stopwords, title='TF-IDF - NN')
train_test_tfidf.train_test_save()
