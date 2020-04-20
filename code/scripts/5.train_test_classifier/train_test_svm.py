import pandas as pd
import pickle
from train_test_classifiers import TrainTestSVM


train_data = pd.read_pickle('../../data/train_data.pkl')
test_data = pd.read_pickle('../../data/test_data.pkl')

with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)


svm_count_params = {
    'ngram_range': (1,2),
    'max_df': 0.5,
    'min_df': 0,
    'C': 0.01,
    'kernel': 'linear'
}
train_test_count = TrainTestSVM(train_data, test_data, ngram_range=svm_count_params['ngram_range'],
                                max_df=svm_count_params['max_df'], min_df=svm_count_params['min_df'],
                                vector='count', C=svm_count_params['C'], kernel=svm_count_params['kernel'],
                                stopwords=stopwords, title='Count - SVM')
train_test_count.train_test_save()


svm_tfidf_params = {
    'ngram_range': (1,2),
    'max_df': 0.25,
    'min_df': 0,
    'C': 1.2575,
    'kernel': 'linear'
}
train_test_tfidf = TrainTestSVM(train_data, test_data, ngram_range=svm_tfidf_params['ngram_range'],
                                max_df=svm_tfidf_params['max_df'], min_df=svm_tfidf_params['min_df'],
                                vector='tfidf', C=svm_tfidf_params['C'], kernel=svm_tfidf_params['kernel'],
                                stopwords=stopwords, title='TF-IDF - SVM')
train_test_tfidf.train_test_save()
