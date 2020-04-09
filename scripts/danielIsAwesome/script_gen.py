def makefilio(fileNum):

    line01 = "import pandas as pd"
    line02 = "import pickle"
    line03 = "import numpy as np"
    line04 = "from tune_hyperparameters import TuneRandomForest"
    line05 = ""
    line06 = ""
    line07 = "# Load training data and stopwords"
    line08 = "train_data = pd.read_pickle('../../data/train_data.pkl')"
    line09 = "with open('../../data/stopwords.pkl', 'rb') as f:"
    line10 = "    stopwords = pickle.load(f)"
    line11 = ""
    line12 = "tune_rf = TuneRandomForest(train_data, 5, stopwords, 'rf%d')" %fileNum
    line13 = "tune_rf.tune_parameters(rf_params, 'count')"
    line14 = "tune_rf.tune_parameters(rf_params, 'tfidf')"
    line15 = "tune_rf.save_scores_csv('rf%d')" %fileNum
    line16 = ""

    lines = [line01, line02, line03, line04, line05, line06, line07, line08, line09, line10, line11, line12, line13,
             line14, line15, line16]

    with open("tune_rf_%d.py" % fileNum, "w") as out:
        out.write('\n'.join(lines))


for i in range(1, 46, 1):
    makefilio(i)
