import pandas as pd
import pickle
import numpy as np
from tune_hyperparameters import TuneRandomForest


# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# for testing
train_data = train_data[:500]

rf_params = {
    'ngram_range':[(1,1)],
    'max_df':[0.5],
    'min_df':[0],
    'max_features':[None],
    'n_estimators':[10],
    'criterion':['gini'],
    'max_depth':[2]
}

tune_rf = TuneRandomForest(train_data, 3, stopwords, 'rf')
tune_rf.tune_parameters(rf_params, 'count')
tune_rf.tune_parameters(rf_params, 'tfidf')
tune_rf.save_scores_csv('rf')
