# -*- coding: utf-8 -*-


import lightgbm
import numpy as np
import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TarbularSHAP:
    def __init__(self, X, y, categorical_states=[]):
        self.X = X
        self.y = y
        self.categorical_states = categorical_states
        self.X_enc = X.copy()
        self.encoders = []
        for state in self.categorical_states:
            if self.X_enc[state].dtypes != object:
                self.X_enc[state] = self.X_enc[state].astype(str)
            encoder = LabelEncoder()
            encoder.fit(self.X_enc[state])
            self.encoders.append(encoder)
            self.X_enc[state] = encoder.transform(self.X_enc[state])
        X_train, X_test, y_train, y_test = train_test_split(self.X_enc, self.y, test_size=0.1,
                                                            random_state=42)
        if len(np.unique(self.y)) > 2:
            task_obj = 'multiclass'
            task_metric = 'multi_logloss'
        else:
            task_obj = 'binary'
            task_metric = 'binary_logloss'
        model = lightgbm.LGBMClassifier(objective=task_obj, num_leaves=31, learning_rate=0.05, n_estimators=1000)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=task_metric,
                  early_stopping_rounds=10, categorical_feature=self.categorical_states, verbose=False)
        self.predictions = model.predict(self.X_enc, num_iteration=model.best_iteration_)
        self.report = classification_report(self.y, self.predictions)
        self.explainer = shap.Explainer(model)

    def explain(self, X=None):
        if X is None:
            X = self.X
        X_enc = X.copy()
        for i in range(len(self.categorical_states)):
            encoder = self.encoders[i]
            if X_enc[self.categorical_states[i]].dtypes != object:
                X_enc[self.categorical_states[i]] = X_enc[self.categorical_states[i]].astype(str)
                X[self.categorical_states[i]] = X[self.categorical_states[i]].astype(str)
            X_enc.loc[~X_enc[self.categorical_states[i]].isin(encoder.classes_), self.categorical_states[i]] = 'unknow'
            encoder.classes_ = np.append(encoder.classes_, 'unknow')
            X_enc[self.categorical_states[i]] = encoder.transform(X_enc[self.categorical_states[i]])
        shap_values = self.explainer(X_enc)
        shap_values.display_data = X.values
        return shap_values




