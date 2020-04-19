import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, roc_curve
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class TrainTestSVM(object):
    def __init__(self, train_data, test_data, ngram_range, max_df, min_df, vector, C,
                 kernel, stopwords, title):
        self.train_data = train_data
        self.test_data = test_data
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vector = vector
        self.C = C
        self.kernel = kernel
        self.stopwords = stopwords
        self.title = title
        self.scores = pd.DataFrame()

    def train_test_save(self):
        X_train = self.train_data['text'].values
        y_train = self.train_data['label'].values
        X_test = self.test_data['text'].values
        y_test = self.test_data['label'].values

        if self.vector == 'count':
            vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)
        else:
            vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        dump(X_train_vec, '../../data/%s_train_vec.joblib' % self.title.replace(' ', ''))
        dump(X_test_vec, '../../data/%s_test_vec.joblib' % self.title.replace(' ', ''))

        clf = SVC(C=self.C, kernel=self.kernel, probability=True, gamma='scale')
        clf.fit(X_train_vec, y_train)

        y_train_pred = clf.predict(X_train_vec)
        y_train_prob = clf.predict_proba(X_train_vec)
        y_train_prob = y_train_prob[:, 1]
        train_scores = self.evaluate_results(y_train, y_train_pred, y_train_prob)

        y_test_pred = clf.predict(X_test_vec)
        y_test_prob = clf.predict_proba(X_test_vec)
        y_test_prob = y_test_prob[:, 1]
        test_scores = self.evaluate_results(y_test, y_test_pred, y_test_prob)

        self.scores = self.create_scores_dataframe(train_scores, test_scores)
        self.plot_roc_curve(y_test, y_test_prob)
        self.save_scores_csv()
        dump(clf, '../../results/models/%s_model.joblib' % self.title.replace(' ', ''))
        return None

    def evaluate_results(self, y_true, y_pred, y_prob):
        scores = {}
        scores['ngram_range'] = [self.ngram_range]
        scores['max_df'] = [self.max_df]
        scores['min_df'] = [self.min_df]
        scores['vector'] = [self.vector]
        scores['C'] = [self.C]
        scores['Acc'] = [accuracy_score(y_true, y_pred)]
        scores['recall'] = [recall_score(y_true, y_pred)]
        scores['PPV'] = [precision_score(y_true, y_pred)]
        scores['AUC'] = [roc_auc_score(y_true, y_prob)]

        return scores

    def create_scores_dataframe(self, train_dict, test_dict):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'

        test_df = pd.DataFrame(test_dict)
        test_df['dataset'] = 'test'
        eval_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        return eval_df

    def save_scores_csv(self):
        self.scores.to_csv('../../results/final/%s_scores.csv' % self.title.replace(' ', ''))
        return None

    def plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        roc_name = '../../results/images/%s_ROC_curve.png' % self.title.replace(' ', '')
        plt.figure(figsize=(10, 7))
        plt.plot([0, 1], [0, 1], linestyle='--', color='#D3D3D3')
        plt.plot(fpr, tpr, color='#8B0000')
        plt.suptitle('%s Receiver Operating Characteristic Curve' % self.title, fontsize=20, y=0.96)
        plt.title('AUC = {:.4f}'.format(auc), fontsize=16)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(roc_name, bbox_inches='tight')
        plt.close()
        return None


class TrainTestRF(object):
    def __init__(self, train_data, test_data, ngram_range, max_df, min_df, vector, n_estimators,
                 criterion, max_depth, stopwords, title):
        self.train_data = train_data
        self.test_data = test_data
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vector = vector
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.stopwords = stopwords
        self.title = title
        self.scores = pd.DataFrame()

    def train_test_save(self):
        X_train = self.train_data['text'].values
        y_train = self.train_data['label'].values
        X_test = self.test_data['text'].values
        y_test = self.test_data['label'].values

        if self.vector == 'count':
            vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)
        else:
            vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        dump(X_train_vec, '../../data/%s_train_vec.joblib' % self.title.replace(' ', ''))
        dump(X_test_vec, '../../data/%s_test_vec.joblib' % self.title.replace(' ', ''))

        clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                     max_depth=self.max_depth)
        clf.fit(X_train_vec, y_train)

        y_train_pred = clf.predict(X_train_vec)
        y_train_prob = clf.predict_proba(X_train_vec)
        y_train_prob = y_train_prob[:, 1]
        train_scores = self.evaluate_results(y_train, y_train_pred, y_train_prob)

        y_test_pred = clf.predict(X_test_vec)
        y_test_prob = clf.predict_proba(X_test_vec)
        y_test_prob = y_test_prob[:, 1]
        test_scores = self.evaluate_results(y_test, y_test_pred, y_test_prob)

        self.scores = self.create_scores_dataframe(train_scores, test_scores)
        self.plot_roc_curve(y_test, y_test_prob)
        self.save_scores_csv()
        dump(clf, '../../results/models/%s_model.joblib' % self.title.replace(' ', ''))
        return None

    def evaluate_results(self, y_true, y_pred, y_prob):
        scores = {}
        scores['ngram_range'] = [self.ngram_range]
        scores['max_df'] = [self.max_df]
        scores['min_df'] = [self.min_df]
        scores['vector'] = [self.vector]
        scores['n_estimators'] = [self.n_estimators]
        scores['criterion'] = [self.criterion]
        scores['max_depth'] = [self.max_depth]
        scores['Acc'] = [accuracy_score(y_true, y_pred)]
        scores['recall'] = [recall_score(y_true, y_pred)]
        scores['PPV'] = [precision_score(y_true, y_pred)]
        scores['AUC'] = [roc_auc_score(y_true, y_prob)]

        return scores

    def create_scores_dataframe(self, train_dict, test_dict):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'

        test_df = pd.DataFrame(test_dict)
        test_df['dataset'] = 'test'
        eval_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        return eval_df

    def save_scores_csv(self):
        self.scores.to_csv('../../results/final/%s_scores.csv' % self.title.replace(' ', ''))
        return None

    def plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        roc_name = '../../results/images/%s_ROC_curve.png' % self.title.replace(' ', '')
        plt.figure(figsize=(10, 7))
        plt.plot([0, 1], [0, 1], linestyle='--', color='#D3D3D3')
        plt.plot(fpr, tpr, color='#8B0000')
        plt.suptitle('%s Receiver Operating Characteristic Curve' % self.title, fontsize=20, y=0.96)
        plt.title('AUC = {:.4f}'.format(auc), fontsize=16)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(roc_name, bbox_inches='tight')
        plt.close()
        return None


class TrainTestNN(object):
    def __init__(self, train_data, test_data, ngram_range, max_df, min_df, vector, h1_nodes, optimizer,
                 stopwords, title):
        self.train_data = train_data
        self.test_data = test_data
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vector = vector
        self.h1_nodes = h1_nodes
        self.optimizer = optimizer
        self.stopwords = stopwords
        self.title = title
        self.scores = pd.DataFrame()

    def train_test_save(self):
        X_train = self.train_data['text'].values
        y_train = self.train_data['label'].values
        X_test = self.test_data['text'].values
        y_test = self.test_data['label'].values

        if self.vector == 'count':
            vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)
        else:
            vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         stop_words=self.stopwords)

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        dump(X_train_vec, '../../data/%s_train_vec.joblib' % self.title.replace(' ', ''))
        dump(X_test_vec, '../../data/%s_test_vec.joblib' % self.title.replace(' ', ''))

        n_dim = X_train_vec.shape[1]
        early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=3)

        model = Sequential()
        model.add(Dense(self.h1_nodes, activation='relu', input_dim=n_dim))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        history = model.fit(X_train_vec, y_train, epochs=3000, validation_split=0.2, batch_size=100,
                            callbacks=[early_stopping_monitor], verbose=0)

        y_train_prob = model.predict(X_train_vec).flatten()
        y_train_pred = model.predict_classes(X_train_vec).flatten()
        train_scores = self.evaluate_results(y_train, y_train_pred, y_train_prob)

        y_test_prob = model.predict(X_test_vec).flatten()
        y_test_pred = model.predict_classes(X_test_vec).flatten()
        test_scores = self.evaluate_results(y_test, y_test_pred, y_test_prob)

        self.scores = self.create_scores_dataframe(train_scores, test_scores)
        self.plot_roc_curve(y_test, y_test_prob)
        self.save_scores_csv()
        model.save('../../results/models/%s_model.h5' % self.title.replace(' ', ''))
        return None

    def evaluate_results(self, y_true, y_pred, y_prob):
        scores = {}
        scores['ngram_range'] = [self.ngram_range]
        scores['max_df'] = [self.max_df]
        scores['min_df'] = [self.min_df]
        scores['vector'] = [self.vector]
        scores['h1_nodes'] = [self.h1_nodes]
        scores['optimizer'] = [self.optimizer]
        scores['Acc'] = [accuracy_score(y_true, y_pred)]
        scores['recall'] = [recall_score(y_true, y_pred)]
        scores['PPV'] = [precision_score(y_true, y_pred)]
        scores['AUC'] = [roc_auc_score(y_true, y_prob)]

        return scores

    def create_scores_dataframe(self, train_dict, test_dict):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'

        test_df = pd.DataFrame(test_dict)
        test_df['dataset'] = 'test'
        eval_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        return eval_df

    def save_scores_csv(self):
        self.scores.to_csv('../../results/final/%s_scores.csv' % self.title.replace(' ', ''))
        return None

    def plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        roc_name = '../../results/images/%s_ROC_curve.png' % self.title.replace(' ', '')
        plt.figure(figsize=(10, 7))
        plt.plot([0, 1], [0, 1], linestyle='--', color='#D3D3D3')
        plt.plot(fpr, tpr, color='#8B0000')
        plt.suptitle('%s Receiver Operating Characteristic Curve' % self.title, fontsize=20, y=0.96)
        plt.title('AUC = {:.4f}'.format(auc), fontsize=16)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(roc_name, bbox_inches='tight')
        plt.close()
        return None
