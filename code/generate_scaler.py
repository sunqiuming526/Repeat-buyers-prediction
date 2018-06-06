import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sknn import mlp
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

feature_all = pickle.load(open("../data/feature_train/feat_final.pytmp"))
scaler = preprocessing.StandardScaler().fit(feature_all)
scaler_file = open("../data/model/scaler.pytmp", 'wb')
pickle.dump(scaler, scaler_file)
scaler_file.close()
