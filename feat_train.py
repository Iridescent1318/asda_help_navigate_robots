import pandas as pd
import numpy as np
import math
import csv
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    x_train_df = pd.read_csv('x_train_feats.csv')
    del x_train_df['series_id']
    y_train_df = pd.read_csv('y_train.csv')['surface']
    folds = StratifiedKFold(n_splits=10, shuffle=True)
    acc_overall = []

    for train, test in folds.split(x_train_df, y_train_df):
        clf_name = ['LogisticRegression', 
                    'AdaBoost', 
                    'DecisionTreeGini',
                    'DecisionTreeEntropy',
                    'GradientTreeBoosting', 
                    'RandomForest',
                    ]

        clfs = [LogisticRegression(solver='newton-cg', max_iter=1000, C=0.5, n_jobs=-1),
                AdaBoostClassifier(n_estimators=500, learning_rate=0.1),
                DecisionTreeClassifier(),
                DecisionTreeClassifier(criterion='entropy'),
                GradientBoostingClassifier(n_estimators=500),
                RandomForestClassifier(n_estimators=500, n_jobs=-1),
                ]

        acc = []

        for c_name, c in zip(clf_name, clfs):
            model = c
            model.fit(x_train_df.iloc[train], y_train_df[train])
            score = model.score(x_train_df.iloc[test], y_train_df[test])
            print("{} accuracy: {:.4f}".format(c_name, score))
            acc.append(score)

        acc_overall.append(acc)

    acc_df = pd.DataFrame()
    for i, c_name in enumerate(clf_name):
        acc_df[c_name] = [x[i] for x in acc_overall]

    print(acc_df.values)
    print(acc_df.describe())
    acc_df.to_csv('accuracy_result.csv', index=True)
