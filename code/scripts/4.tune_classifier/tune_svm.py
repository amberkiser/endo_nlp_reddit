import pandas as pd
import pickle
from tune_hyperparameters import TuneSVM


# Load training data and stopwords
train_data = pd.read_pickle('../../../data/train_data.pkl')
with open('../../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

svm_params = {
    'ngram_range':[(1,1),(1,2),(2,2)],
    'max_df':np.linspace(0, 1, 5),
    'min_df':np.linspace(0, 1, 5),
    'C':np.linspace(0.01, 5, 5)
}

tune_psvm = TuneSVM(train_data, 'poly', 3, stopwords, 'psvm')
tune_psvm.tune_parameters(svm_params, 'count')
tune_psvm.tune_parameters(svm_params, 'tfidf')
tune_psvm.save_scores_csv('psvm')

tune_lsvm = TuneSVM(train_data, 'linear', 3, stopwords, 'lsvm')
tune_lsvm.tune_parameters(svm_params, 'count')
tune_lsvm.tune_parameters(svm_params, 'tfidf')
tune_lsvm.save_scores_csv('lsvm')

tune_rsvm = TuneSVM(train_data, 'rbf', 3, stopwords, 'rsvm')
tune_rsvm.tune_parameters(svm_params, 'count')
tune_rsvm.tune_parameters(svm_params, 'tfidf')
tune_rsvm.save_scores_csv('rsvm')
