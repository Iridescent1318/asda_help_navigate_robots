import pandas as pd
import numpy as np
import math
import csv
import time
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

USE_PCA = 0
ESTIMATION = 1

# CREDIT TO https://www.kaggle.com/jesucristo/1-smart-robots-most-complete-notebook#Run-Model

def save_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.clf()
    plt.figure(figsize=(14, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix: '+title, size=20)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.savefig('cmatrix_' + title + '.png')

if __name__ == "__main__":
    x_train_df = pd.read_csv('x_train_feats.csv')
    del x_train_df['series_id']
    y_train_df = pd.read_csv('y_train.csv')['surface']
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

    if USE_PCA:
        pca = PCA(n_components=40)
        pca.fit(x_train_df)
        x_train_df = pd.DataFrame(pca.transform(x_train_df))

    if ESTIMATION == 0:
        acc_overall = []
        start = time.perf_counter()
        count = 0
        for train, test in folds.split(x_train_df, y_train_df):
            clf_name = ['LogisticRegression', 
                        'AdaBoost', 
                        'DecisionTree',
                        'GradientBoosting', 
                        'RandomForest',
                        ]

            clfs = [LogisticRegression(solver='newton-cg', max_iter=1000, n_jobs=-1),
                    AdaBoostClassifier(n_estimators=500, learning_rate=0.01),
                    DecisionTreeClassifier(),
                    GradientBoostingClassifier(n_estimators=500),
                    RandomForestClassifier(n_estimators=500, n_jobs=-1),
                    ]

            acc = []

            for c_name, c in zip(clf_name, clfs):
                print("CV {} starts".format(count))
                model = c
                model.fit(x_train_df.iloc[train], y_train_df[train])
                score = model.score(x_train_df.iloc[test], y_train_df[test])
                # plot_confusion_matrix(model.predict(x_train_df.iloc[test]), y_train_df[test], model.classes_)
                print("{} accuracy: {:.4f}".format(c_name, score))
                acc.append(score)

            count += 1
            acc_overall.append(acc)

        end = time.perf_counter() - start
        acc_df = pd.DataFrame()
        for i, c_name in enumerate(clf_name):
            acc_df[c_name] = [x[i] for x in acc_overall]

        print("Total time: {:.2f} sec".format(end))
        print(acc_df.values)
        print(acc_df.describe())
        acc_df.to_csv('accuracy_result.csv', index=True)
    
    else:
        prec = []
        prec_sw = []
        recall = []
        recall_sw = []
        f1 = []
        f1_sw = []
        auc = []
        auc_sw = []

        class_prec = []
        class_recall = []
        class_f1 = []

        roc_fpr = []
        roc_tpr = []

        for train, test in folds.split(x_train_df, y_train_df):
            sw = compute_sample_weight(class_weight='balanced', y=y_train_df[test])
            clf_name = ['LogisticRegression', 
                        'AdaBoost', 
                        'DecisionTree',
                        'GradientBoosting', 
                        'RandomForest',
                        ]

            clfs = [LogisticRegression(solver='newton-cg', max_iter=1000, n_jobs=-1),
                    AdaBoostClassifier(n_estimators=500, learning_rate=0.01),
                    DecisionTreeClassifier(),
                    GradientBoostingClassifier(n_estimators=500),
                    RandomForestClassifier(n_estimators=500, n_jobs=-1),
                    ]
            
            for c_name, c in zip(clf_name, clfs):
                c.fit(x_train_df.iloc[train], y_train_df[train])
                d_y_true = y_train_df[test]
                d_y_true_onehot = label_binarize(d_y_true, c.classes_)
                d_y_pred = c.predict(x_train_df.iloc[test])
                d_y_score = c.predict_proba(x_train_df.iloc[test])
                prec.append(precision_score(d_y_true, d_y_pred, average='weighted'))
                prec_sw.append(precision_score(d_y_true, d_y_pred, average='weighted', sample_weight=sw))
                recall.append(recall_score(d_y_true, d_y_pred, average='weighted'))
                recall_sw.append(recall_score(d_y_true, d_y_pred, average='weighted', sample_weight=sw))
                f1.append(f1_score(d_y_true, d_y_pred, average='weighted'))
                f1_sw.append(f1_score(d_y_true, d_y_pred, average='weighted', sample_weight=sw))
                auc.append(roc_auc_score(d_y_true_onehot, d_y_score, average='weighted'))
                auc_sw.append(roc_auc_score(d_y_true_onehot, d_y_score, average='weighted', sample_weight=sw))
                class_prec.append(precision_score(d_y_true, d_y_pred, average=None))
                class_recall.append(recall_score(d_y_true, d_y_pred, average=None))
                class_f1.append(f1_score(d_y_true, d_y_pred, average=None))
                c_roc_fpr, c_roc_tpr, _ = roc_curve(d_y_true_onehot.ravel(), d_y_score.ravel())
                roc_fpr.append(c_roc_fpr)
                roc_tpr.append(c_roc_tpr)
                save_confusion_matrix(d_y_true, d_y_pred, c.classes_, title=c_name)

            esti_overall = pd.DataFrame()
            esti_overall['Classifier'] = clf_name
            esti_overall['Precision'] = prec
            esti_overall['Recall'] = recall
            esti_overall['f1_score'] = f1
            esti_overall['auc'] = auc
            esti_overall['Precision_sw'] = prec_sw
            esti_overall['Recall_sw'] = recall_sw
            esti_overall['f1_score_sw'] = f1_sw
            esti_overall['auc_sw'] = auc_sw

            esti_overall.to_csv('estimation_overall.csv', index=False)

            esti_classes = pd.DataFrame()
            esti_classes['surface'] = clfs[0].classes_
            for i, c_name in enumerate(clf_name):
                esti_classes[c_name + '_prec'] = class_prec[i]
                esti_classes[c_name + '_recall'] = class_recall[i]
                esti_classes[c_name + '_f1'] = class_f1[i]
            
            esti_classes.to_csv('estimation_class.csv', index=False)

            plt.clf()
            for c_name, rf, rt in zip(clf_name, roc_fpr, roc_tpr):
                plt.plot(rf, rt, lw = 2, alpha = 0.7, label = c_name)
            plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
            plt.xlim((-0.01, 1.02))
            plt.ylim((-0.01, 1.02))
            plt.xticks(np.arange(0, 1.1, 0.1), fontsize=20)
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.ylabel('True Positive Rate', fontsize=20)
            plt.grid(b=True, ls=':')
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=20)
            plt.title(u'ROC Curve', fontsize=30)
            plt.savefig('ROC_curve.png')

            break




