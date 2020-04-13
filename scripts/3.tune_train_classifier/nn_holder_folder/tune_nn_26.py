import pandas as pd
import pickle
import numpy as np
from tune_hyperparameters import TuneNeuralNet


# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

nn_params = {
    'ngram_range':[(1,2)],
    'max_df':np.linspace(0, 1, 5),
    'min_df':np.linspace(0, 1, 5),
    'h1_nodes':[1024],
    'optimizer':['Adadelta']
}

tune_nn = TuneNeuralNet(train_data, 3, stopwords, 'nn26')
tune_nn.tune_parameters(nn_params, 'count')
tune_nn.tune_parameters(nn_params, 'tfidf')
tune_nn.save_scores_csv('nn26')