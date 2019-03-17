import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def load_data():
#load data
	train_data = pd.read_csv('train.csv')

	test_data = pd.read_csv('test.csv')

	return train_data,test_data

def fit_model(train_df,test_df,folds):
# Choose to use Lightgbm classifier
# After fitting our model, the feature importance would be obtained.
	param = {'bagging_freq': 5,          
	'max_bin':330 ,       
    'bagging_fraction': 0.330,   
    'boost_from_average':'false',   
    'boost': 'gbdt',             
    'feature_fraction': 0.05,     
    'learning_rate': 0.0080,
    'max_depth': -1,             
    'metric':'auc',                
    'min_data_in_leaf': 80,     
    'min_sum_hessian_in_leaf': 10,
    'num_leaves': 13,            
    'num_threads': 8,              
    'tree_learner': 'serial',   
    'objective': 'binary',      
    'verbosity': 1
	}


	features = [c for c in train_df.columns if c not in ['ID_code','target']]

	target = train_df['target']

	skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2019)

	oof = np.zeros(len(train_df))

	predictions = np.zeros(len(test_df))

	feature_importance_df = pd.DataFrame()


	for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):

		print("fold {}".format(fold_))

    	
		trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    	
		val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

		num_round=500000000
		
		clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 5000)


		oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
		fold_importance_df = pd.DataFrame()
		fold_importance_df["feature"] = features
		fold_importance_df["importance"] = clf.feature_importance()
		fold_importance_df["fold"] = fold_ + 1
		feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
		predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds

	print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

	pred = pd.DataFrame({'ID_code':test_df['ID_code'],'target':predictions})
	pred.to_csv('hehe5.csv',index=False)


	return roc_auc_score(target, oof),predictions,fold_importance_df.sort_values(by='importance')


def submit_file(predictions,test_data):
#input 'predictions' comes from the last function
	pred = pd.DataFrame({'ID_code':test_data['ID_code'],'target':predictions})

	pred.to_csv('submission.csv',index=False)



