import pandas as pd
import pickle
from tune_hyperparameters import TuneSVM


# Load training data and stopwords
train_data = pd.read_pickle('../../data/train_data.pkl')
with open('../../data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# for testing
train_data = train_data[:500]

# svm_params = {
#     'ngram_range':[(1,1),(1,2),(2,2)],
#     'max_df':np.linspace(0, 1, 5),
#     'min_df':np.linspace(0, 1, 5),
#     'max_features':[None, 1000, 2000],
#     'C':np.linspace(0.01, 5, 5)
# }

svm_params = {
    'ngram_range':[(1,1)],
    'max_df':[0.5],
    'min_df':[0],
    'max_features':[None],
    'C':[1.0]
}

tune_psvm = TuneSVM(train_data, 'poly', 5, stopwords, 'psvm1')
tune_psvm.tune_parameters(svm_params, 'count')
tune_psvm.tune_parameters(svm_params, 'tfidf')
tune_psvm.save_scores_csv('psvm1')
