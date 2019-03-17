from MODEL import fit_model
from EDA import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from xgboost import XGBClassifier
import importlib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import time
import sys
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import RUSBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
path = os.getcwd()
print(path)


def undersampling_case():
#we would get a table which is formed by n_subsets' prediction.
#Maybe we could apply majority votes or take weighted mean value.
	train_data,test_data = load_data()
	X,y = get_Xy(train_data)

	ee = EasyEnsemble(n_subsets = 11)

	X_res,y_res = ee.fit_sample(X,y)
	j = 0
	pred_table = pd.DataFrame()
	for subset in X_res:
		new_sub = pd.DataFrame(subset,columns = X.columns)
		new_sub['ID_code'] = train_data['ID_code']
		new_sub['target'] = y_res[j]
		score,prediction,hols = fit_model(new_sub,test_data,5)
		pred_table[j] = prediction
		j += 1
		print('We got one prediction!!!!!!!!!!!')

	return pred_table

def feature_selection():

	train_set,test_set = load_data()

	X  = train_set.drop(['target'],axis=1)
	y= train_set['target']


	k_best = SelectKBest(f_classif,k=5)

	k_best.fit_transform(X,y)

	k_best.pvalues_

	p_values = pd.DataFrame({'column':X.columns,'p_value':k_best.pvalues_}).sort_values('p_value')

	return p_values

def voting():

	train_data,test_data = load_data()
	X,y = get_Xy(train_data)

	ee = EasyEnsemble(n_subsets = 11)

	X_res,y_res = ee.fit_sample(X,y)

	clf1 = RandomForestClassifier(n_estimators = 100, max_depth =5,max_leaf_nodes = 30)
	clf2 = LogisticRegression(random_state =1 ,solver = 'liblinear',multi_class = 'ovr', tol = 1e-6)
	clf3 = GaussianNB()

	ensemble_clf = VotingClassifier(estimators = [('RFC',clf1),('LR',clf2),('GNB',clf3)],voting = 'soft')

	for clf, label , X in zip([clf1,clf2,clf3,ensemble_clf],['RandomForestClassifier','LogisticRegression','GaussianNB','VotingClassifier'],X_res):

		scores = cross_val_score(clf,X,y_res[0],cv=5,scoring = 'accuracy')

		print('Accuracy is {} and correspond label is {}'.format(scores.mean(),label))

def compare_with_new_feature():

	train_data,test_data = load_data()

	score1,prediction1,impt1 = fit_model(train_data,test_data,2)

	train_data['f1'] = train_data['var_76'] - train_data['var_71']
	test_data['f1'] = test_data['var_76'] - test_data['var_71']

	score2,prediction2,impt2 = fit_model(train_data,test_data,2)

	return score1,score2