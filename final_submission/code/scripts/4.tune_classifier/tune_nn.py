import pandas as pd
import pickle
import numpy as np
from keras import backend as K
import tensorflow as tf
import os
from tune_hyperparameters import TuneNeuralNet

# Load training data and stopwords
train_data = pd.read_pickle('../../../data/train_data.pkl')
with open('../../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# This is the number of physical cores
NUM_PARALLEL_EXEC_UNITS = 12
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                                  allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

session = tf.compat.v1.Session(config=config)

K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

nn_params = {
    'ngram_range':[(1,1),(1,2),(2,2)],
    'max_df':np.linspace(0, 1, 5),
    'min_df':np.linspace(0, 1, 5),
    'h1_nodes':[128, 512, 1024, 2048, 3200],
    'optimizer':['Adam','RMSprop','Adadelta']
}

# May need to edit batch_size to a smaller size to lessen memory load.
tune_nn = TuneNeuralNet(train_data, 3, stopwords, 'nn')
tune_nn.tune_parameters(nn_params, 'count')
tune_nn.tune_parameters(nn_params, 'tfidf')
tune_nn.save_scores_csv('nn')
