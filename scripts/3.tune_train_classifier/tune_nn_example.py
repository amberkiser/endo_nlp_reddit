import pandas as pd
import pickle
import numpy as np
from tune_hyperparameters import TuneNeuralNet

# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# for testing
train_data = train_data[:500]

nn_params = {
    'ngram_range':[(1,1)],
    'max_df':[0.5],
    'min_df':[0],
    'h1_nodes':[128],
    'optimizer':['Adam']
}

tune_nn = TuneNeuralNet(train_data, 3, stopwords, 'nn')
tune_nn.tune_parameters(nn_params, 'count')
tune_nn.tune_parameters(nn_params, 'tfidf')
tune_nn.save_scores_csv('nn')
