import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class TuneSVM(object):
    def __init__(self, train_data, kernel, cv_num, stopwords, title):
        self.data = train_data
        self.kernel = kernel
        self.stopwords = stopwords
        self.title = title
        self.k_folds = KFold(n_splits=cv_num, shuffle=True)
        self.cv_scores = pd.DataFrame()

    def tune_parameters(self, params, vector):
        ngram_range = params['ngram_range']
        max_df = params['max_df']
        min_df = params['min_df']

        C = params['C']

        for n in ngram_range:
            for mx in max_df:
                for mn in min_df:
                    for c in C:
                        self.run_cv(n, mx, mn, c, vector)
        return None

    def save_scores_csv(self, title):
        self.cv_scores.to_csv('../../results/tuning/%s_tuning.csv' % title)
        return None

    def run_cv(self, ngram_range, max_df, min_df, C, vector):
        fold = 0
        for train_index, val_index in self.k_folds.split(self.data):
            fold += 1
            print(fold)
            X_train = self.data.iloc[train_index]['text'].values
            y_train = self.data.iloc[train_index]['label'].values
            X_val = self.data.iloc[val_index]['text'].values
            y_val = self.data.iloc[val_index]['label'].values

            if vector == 'count':
                vectorizer = CountVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)
            else:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)

            try:
                X_train_vec = vectorizer.fit_transform(X_train)
                X_val_vec = vectorizer.transform(X_val)
            except:
                return None
            else:
                clf = SVC(C=C, kernel=self.kernel, probability=True, gamma='scale')
                clf.fit(X_train_vec, y_train)

                y_train_pred = clf.predict(X_train_vec)
                y_train_prob = clf.predict_proba(X_train_vec)
                y_train_prob = y_train_prob[:, 1]
                train_scores = self.evaluate_cv_results(y_train, y_train_pred, y_train_prob,
                                                        ngram_range, max_df, min_df, C)

                y_val_pred = clf.predict(X_val_vec)
                y_val_prob = clf.predict_proba(X_val_vec)
                y_val_prob = y_val_prob[:, 1]
                val_scores = self.evaluate_cv_results(y_val, y_val_pred, y_val_prob,
                                                      ngram_range, max_df, min_df, C)

                eval_df = self.create_scores_dataframe(train_scores, val_scores, fold, vector)
                self.cv_scores = pd.concat([self.cv_scores, eval_df])
                self.save_scores_csv('temp_%s' % self.title)
        return None

    def evaluate_cv_results(self, y_true, y_pred, y_prob, ngram_range, max_df, min_df, C):
        scores = {'ngram_range': [], 'max_df': [], 'min_df': [], 'C': [],
                  'Acc': [], 'recall': [], 'PPV': [], 'AUC': []}

        scores['ngram_range'].append(ngram_range)
        scores['max_df'].append(max_df)
        scores['min_df'].append(min_df)
        scores['C'].append(C)
        scores['Acc'].append(accuracy_score(y_true, y_pred))
        scores['recall'].append(recall_score(y_true, y_pred))
        scores['PPV'].append(precision_score(y_true, y_pred))
        scores['AUC'].append(roc_auc_score(y_true, y_prob))

        return scores

    def create_scores_dataframe(self, train_dict, val_dict, fold, vector):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'
        train_df['fold'] = fold
        train_df['vector'] = vector

        val_df = pd.DataFrame(val_dict)
        val_df['dataset'] = 'val'
        val_df['fold'] = fold
        val_df['vector'] = vector
        eval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        return eval_df


class TuneRandomForest(object):
    def __init__(self, train_data, cv_num, stopwords, title):
        self.data = train_data
        self.stopwords = stopwords
        self.title = title
        self.k_folds = KFold(n_splits=cv_num, shuffle=True)
        self.cv_scores = pd.DataFrame()

    def tune_parameters(self, params, vector):
        ngram_range = params['ngram_range']
        max_df = params['max_df']
        min_df = params['min_df']
        max_features = params['max_features']

        n_estimators = params['n_estimators']
        criterion = params['criterion']
        max_depth = params['max_depth']

        for n in ngram_range:
            for mx in max_df:
                for mn in min_df:
                    for m in max_features:
                        for nest in n_estimators:
                            for c in criterion:
                                for mxd in max_depth:
                                    self.run_cv(n, mx, mn, m, nest, c, mxd, vector)
        return None

    def save_scores_csv(self, title):
        self.cv_scores.to_csv('../../results/tuning/%s_tuning.csv' % title)
        return None

    def run_cv(self, ngram_range, max_df, min_df, max_features, n_estimators, criterion, max_depth, vector):
        fold = 0
        for train_index, val_index in self.k_folds.split(self.data):
            fold += 1
            print(fold)
            X_train = self.data.iloc[train_index]['text'].values
            y_train = self.data.iloc[train_index]['label'].values
            X_val = self.data.iloc[val_index]['text'].values
            y_val = self.data.iloc[val_index]['label'].values

            if vector == 'count':
                vectorizer = CountVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)
            else:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)

            try:
                X_train_vec = vectorizer.fit_transform(X_train)
                X_val_vec = vectorizer.transform(X_val)
            except:
                return None
            else:
                clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                clf.fit(X_train_vec, y_train)

                y_train_pred = clf.predict(X_train_vec)
                y_train_prob = clf.predict_proba(X_train_vec)
                y_train_prob = y_train_prob[:, 1]
                train_scores = self.evaluate_cv_results(y_train, y_train_pred, y_train_prob,
                                                        ngram_range, max_df, min_df, max_features,
                                                        n_estimators, criterion, max_depth)

                y_val_pred = clf.predict(X_val_vec)
                y_val_prob = clf.predict_proba(X_val_vec)
                y_val_prob = y_val_prob[:, 1]
                val_scores = self.evaluate_cv_results(y_val, y_val_pred, y_val_prob,
                                                      ngram_range, max_df, min_df, max_features,
                                                      n_estimators, criterion, max_depth)

                eval_df = self.create_scores_dataframe(train_scores, val_scores, fold, vector)
                self.cv_scores = pd.concat([self.cv_scores, eval_df])
                self.save_scores_csv('temp_%s' % self.title)
        return None

    def evaluate_cv_results(self, y_true, y_pred, y_prob, ngram_range, max_df, min_df, max_features,
                            n_estimators, criterion, max_depth):
        scores = {'ngram_range': [], 'max_df': [], 'min_df': [], 'max_features': [], 'n_estimators': [],
                  'criterion': [],
                  'max_depth': [], 'Acc': [], 'recall': [], 'PPV': [], 'AUC': []}

        scores['ngram_range'].append(ngram_range)
        scores['max_df'].append(max_df)
        scores['min_df'].append(min_df)
        scores['max_features'].append(max_features)
        scores['n_estimators'].append(n_estimators)
        scores['criterion'].append(criterion)
        scores['max_depth'].append(max_depth)
        scores['Acc'].append(accuracy_score(y_true, y_pred))
        scores['recall'].append(recall_score(y_true, y_pred))
        scores['PPV'].append(precision_score(y_true, y_pred))
        scores['AUC'].append(roc_auc_score(y_true, y_prob))

        return scores

    def create_scores_dataframe(self, train_dict, val_dict, fold, vector):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'
        train_df['fold'] = fold
        train_df['vector'] = vector

        val_df = pd.DataFrame(val_dict)
        val_df['dataset'] = 'val'
        val_df['fold'] = fold
        val_df['vector'] = vector
        eval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        return eval_df


class TuneNeuralNet(object):
    def __init__(self, train_data, cv_num, stopwords, title):
        self.data = train_data
        self.stopwords = stopwords
        self.title = title
        self.k_folds = KFold(n_splits=cv_num, shuffle=True)
        self.cv_scores = pd.DataFrame()

    def tune_parameters(self, params, vector):
        ngram_range = params['ngram_range']
        max_df = params['max_df']
        min_df = params['min_df']

        h1_nodes = params['h1_nodes']
        optimizer = params['optimizer']

        for n in ngram_range:
            for mx in max_df:
                for mn in min_df:
                    for h1 in h1_nodes:
                        for o in optimizer:
                            self.run_cv(n, mx, mn, h1, o, vector)
        return None

    def save_scores_csv(self, title):
        self.cv_scores.to_csv('../../results/tuning/%s_tuning.csv' % title)
        return None

    def run_cv(self, ngram_range, max_df, min_df, h1_nodes, optimizer, vector):
        fold = 0
        for train_index, val_index in self.k_folds.split(self.data):
            fold += 1
            print(fold)
            X_train = self.data.iloc[train_index]['text'].values
            y_train = self.data.iloc[train_index]['label'].values
            X_val = self.data.iloc[val_index]['text'].values
            y_val = self.data.iloc[val_index]['label'].values

            if vector == 'count':
                vectorizer = CountVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)
            else:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                             max_df=max_df,
                                             min_df=min_df,
                                             stop_words=self.stopwords)

            try:
                X_train_vec = vectorizer.fit_transform(X_train)
                X_val_vec = vectorizer.transform(X_val)
            except:
                return None
            else:
                n_dim = X_train_vec.shape[1]
                early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=3)

                model = Sequential()
                model.add(Dense(h1_nodes, activation='relu', input_dim=n_dim))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                history = model.fit(X_train_vec, y_train, epochs=3000, validation_split=0.2, batch_size=100,
                                    callbacks=[early_stopping_monitor], verbose=0)

                y_train_prob = model.predict(X_train_vec).flatten()
                y_train_pred = model.predict_classes(X_train_vec).flatten()
                train_scores = self.evaluate_cv_results(y_train, y_train_pred, y_train_prob,
                                                        ngram_range, max_df, min_df,
                                                        h1_nodes, optimizer)

                y_val_prob = model.predict(X_val_vec).flatten()
                y_val_pred = model.predict_classes(X_val_vec).flatten()
                val_scores = self.evaluate_cv_results(y_val, y_val_pred, y_val_prob,
                                                      ngram_range, max_df, min_df,
                                                      h1_nodes, optimizer)

                eval_df = self.create_scores_dataframe(train_scores, val_scores, fold, vector)
                self.cv_scores = pd.concat([self.cv_scores, eval_df])
                self.save_scores_csv('temp_%s' % self.title)
        return None

    def evaluate_cv_results(self, y_true, y_pred, y_prob, ngram_range, max_df, min_df,
                            h1_nodes, optimizer):
        scores = {'ngram_range': [], 'max_df': [], 'min_df': [], 'h1_nodes': [], 'optimizer': [],
                  'Acc': [], 'recall': [], 'PPV': [], 'AUC': []}

        scores['ngram_range'].append(ngram_range)
        scores['max_df'].append(max_df)
        scores['min_df'].append(min_df)
        scores['h1_nodes'].append(h1_nodes)
        scores['optimizer'].append(optimizer)
        scores['Acc'].append(accuracy_score(y_true, y_pred))
        scores['recall'].append(recall_score(y_true, y_pred))
        scores['PPV'].append(precision_score(y_true, y_pred))
        scores['AUC'].append(roc_auc_score(y_true, y_prob))

        return scores

    def create_scores_dataframe(self, train_dict, val_dict, fold, vector):
        train_df = pd.DataFrame(train_dict)
        train_df['dataset'] = 'train'
        train_df['fold'] = fold
        train_df['vector'] = vector

        val_df = pd.DataFrame(val_dict)
        val_df['dataset'] = 'val'
        val_df['fold'] = fold
        val_df['vector'] = vector
        eval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        return eval_df
