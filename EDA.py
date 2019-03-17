import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import importlib
path = os.getcwd()
print(path)

# os.chdir('/home/yanzhang/Pydata/santer_prediction')

def load_data():
#load data
	train_data = pd.read_csv('train.csv')

	test_data = pd.read_csv('test.csv')

	return train_data,test_data

def get_Xy(train_data):
	# Input is the original train set we got from the last function
# This function is to get our training dataset and target values
	X = train_data.drop(['ID_code','target'],axis =1)

	y = train_data['target']

	return X,y

def get_split(X,y):
#we could split train data into validation set and training set 
# and it could be used to check performance when some changes take place in features  
	train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)

	return train_X,val_X,train_y,val_y

def check_missing_value_for_dataframe(dataset):
	#The input is any dataset we want to check
#Before this, we need to show the data type, like numerical values and categorical values. 
#It could be done by using 'df.info()'
#Fortunately, it's all numerical type.
#Then missing value is another concern. 
	for feature in dataset.columns:

		Total = dataset[feature].isnull().sum()

		Percent = dataset[feature].isnull().sum()/dataset[feature].isnull().count()*100

		print('The feature {} missing value number is {}, and the percent is {}'.format(feature,Total,Percent) )

def check_target_distribution(train_data):
# Most of time, we would meet imbalanced distribution of the target 

	m = train_data['target'].value_counts(0)

	ax = m.plot.bar()

def feature_density_plot(train_data,lower_limit,upper_limit):
#Every feature is grouped by target value and check its distribution in training set
#And it would take a little bit longer if we plot all features. 
#Hence we can set attributes into several groups.
#Here we go with twenty features each time
#Guarantee 'upper_limit - lower_limit = 20' and 'lower_limit >= 2'
	fig = plt.figure()

	train_data = train_data.drop(['ID_code'],axis=1)
	i=1

	for feas in train_data.columns[lower_limit:upper_limit]:

		data1 = train_data[train_data['target'] == 1][feas]
		data0 = train_data[train_data['target'] == 0][feas]

		fig.add_subplot(4,5,i)

		sns.kdeplot(data1,label = '1')
		sns.kdeplot(data0,label = '0')

		plt.xlabel(feas)

		i+=1

	plt.show()

def feature_correlation(train_data):
#We would like to check the correlations between features and target values.
#Or using 'Heatmap' to illustrate the correlations 
	train_data = train_data.drop(['ID_code'],axis=1)
	Cor = train_data.corr()
	print('{}'.format(Cor))
	for fea in train_data.columns: 
		he = Cor.sort_values(by=fea,axis=0) 
		print('The feature is {} and most related feature is {}'.format(fea,
			he.index[-2] if abs(he.values[0][0]) < he.values[0][-2] else he.index[0])) 

def check_duplicate_values(train_data,test_data):
#We need to check is there any duplicate values in both the train set and test set.

	train_data = train_data.drop(['ID_code','target'],axis=1)
	test_data = test_data.drop(['ID_code'],axis=1)
	features = train_data.columns
	table_for_train = pd.DataFrame(index = ['values','max_times'])
	table_for_test = pd.DataFrame(index = ['values','max_times'])

	for f in features:

		trains = train_data[f].value_counts()
		tests = test_data[f].value_counts()

		table_for_train[f] = [trains.idxmax(),trains.max()]
		table_for_test[f] = [tests.idxmax(),tests.max()]


	return table_for_train.T.sort_values(by='max_times',ascending =False),table_for_test.T.sort_values(by='max_times',ascending = False)

def plot_value_count():
#similar with the last function, go deeper into the features
	train_data,test_data = load_data()
	traintable , testtable = check_duplicate_values(train_data,test_data)
	i = 1
	fig = plt.figure(figsize = (15,10))

	for feas in traintable.index[:45]:
		fig.add_subplot(3,15,i)
		sns.distplot(train_data[feas].value_counts(),kde=True)
		i+=1

