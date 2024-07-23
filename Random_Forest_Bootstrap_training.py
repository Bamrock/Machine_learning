import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

n_bt = 3
# df = pd.read_csv("Fibrosis.csv",encoding='gbk')
df = pd.read_csv("NAFLD.csv",encoding='gbk')
X = df.iloc[:,3:-1]
y = df.iloc[:,2]

prob_fold = []
real_fold = []
roc_fold = 0
sens_fold = 0
spec_fold = 0
acc_fold = 0
roc_fold_ci = []
sens_fold_ci = []
spec_fold_ci = []
acc_fold_ci = []

np.random.seed(42)
n_samples = int(len(df)*0.8)
bootstrap_indices = np.random.choice(df.index.values, size=(n_bt, n_samples))
bootstrapped_dfs = [df.loc[indices] for indices in bootstrap_indices]
print(df.shape)
for indices in bootstrap_indices:
    final_df = df.loc[indices]
    # print(final_df.shape)
    X = final_df.iloc[:, 3:-1]
    # X = df.iloc[:,:-2]
    y = final_df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1587, stratify=y)

    clf = RandomForestClassifier(max_depth=100,max_features=12)
    clf.fit(x_train, y_train)
    y_proba = clf.predict_proba(x_test)
    y_predict = clf.predict(x_test)

    prob_fold.append(y_proba)
    real_fold.append(y_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba[:, 1])

    confusion = confusion_matrix(y_test, y_predict)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (TP+TN)/float(TP+TN+FP+FN)
    sens = TP / float(TP+FN)
    spec = TN / float(TN+FP)
    roc_auc = auc(fpr, tpr)

    roc_fold_ci.append(roc_auc)
    sens_fold_ci.append(sens)
    spec_fold_ci.append(spec)
    acc_fold_ci.append(acc)
    roc_fold += roc_auc
    acc_fold += acc
    sens_fold += sens
    spec_fold += spec


print("********NAFLD********")
print("auc: %0.3f (+/- %0.3f)" % (np.mean(roc_fold_ci), np.std(roc_fold_ci) * 2))
print("acc: %0.3f (+/- %0.3f)" % (np.mean(acc_fold_ci), np.std(acc_fold_ci) * 2))
print("sens: %0.3f (+/- %0.3f)" % (np.mean(sens_fold_ci), np.std(sens_fold_ci) * 2))
print("spec: %0.3f (+/- %0.3f)" % (np.mean(spec_fold_ci), np.std(spec_fold_ci) * 2))

roc_fold_ci_NAFLD = np.mean(roc_fold_ci)
roc_fold_ci_NAFLD_low = np.mean(roc_fold_ci)-np.std(roc_fold_ci) * 2
roc_fold_ci_NAFLD_up = np.mean(roc_fold_ci)+np.std(roc_fold_ci) * 2
