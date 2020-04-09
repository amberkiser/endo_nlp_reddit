import pandas as pd
import pickle
import numpy as np
from tune_hyperparameters import TuneRandomForest


# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

rf_params = {
    'ngram_range':[(1,2)],
    'max_df':np.linspace(0, 1, 5),
    'min_df':np.linspace(0, 1, 5),
    'max_features':[2000],
    'n_estimators':[100],
    'criterion':['gini','entropy'],
    'max_depth':[2, 10, 20, 50, 100]
}

tune_rf = TuneRandomForest(train_data, 5, stopwords, 'rf35')
tune_rf.tune_parameters(rf_params, 'count')
tune_rf.tune_parameters(rf_params, 'tfidf')
tune_rf.save_scores_csv('rf35')
