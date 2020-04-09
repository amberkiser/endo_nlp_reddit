import pandas as pd
import numpy as np
import pickle
from tune_hyperparameters import TuneSVM


# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)


svm_params = {
    'ngram_range':[(2,2)],
    'max_df':np.linspace(0, 1, 5),
    'min_df':np.linspace(0, 1, 5),
    'max_features':[1000],
    'C':np.linspace(0.01, 5, 5)
}


tune_lsvm = TuneSVM(train_data, 'poly', 5, stopwords, 'psvm6')
tune_lsvm.tune_parameters(svm_params, 'count')
tune_lsvm.tune_parameters(svm_params, 'tfidf')
tune_lsvm.save_scores_csv('psvm6')
