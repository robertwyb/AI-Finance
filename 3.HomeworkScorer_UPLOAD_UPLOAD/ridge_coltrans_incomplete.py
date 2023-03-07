"""
Fill in the missing code. The lines with missing code have the string "#####" or '*'
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
Actually, we added np.random.seed() to fix the results, so you can check them.

You will be filling in code in two types of models:
1. a regression model and
2. a classification model.

Most of the time, because of similarities,
you can cut and paste from one model to the other.
But in a few instances, you cannot do this, so
you need to pay attention.
Also, in some cases,
you will find a "hint" for a solution 
in one of the two scripts (regression or classification)
that you can use as inspiration for the other.

This double task gives you the opportunity to look at the results
in both regression and classification approaches.

At the bottom, you will find some questions that we pose.
You do not need to write and turn in the answer to these questions,
but we strongly recommend you find out the answers to them.

"""

"""
INSTRUCTIONS
In the previous version of this homework, 
we did not want to apply scaling to categorical features because
it is not really necessary
So in the previous homework,
we turned off scaling by setting the parameters to false:
scaler = StandardScaler(with_mean=False, with_std=False)
which was ok given that returns are mostly scaled already.

In this version of the same homework,
we are going to use a ColumnTransformer
to construct a feature preprocessor that 
diffferentiates between categorical features and numeric features.
The feature preprocessor applies scaling to numeric features and
applies one-hot-encoding to categorical features.
When using a ColumnTransformer the input must be a dataframe, not an array.
We skip data exploration analysis for this homework

Pipelines are very powerful.
We will only scratch the surface of pipelines in this course but
pipelines enable you to chain estimators together as in https://archive.is/bjxPJ
so you need to learn about them.
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys

np.random.seed(1) #to fix the results

#uncomment if you want to redirect print output to outputfile.txt
#file_path = 'outputfile.txt'
#sys.stdout = open(file_path, "w")


#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#save the close and open for white reality check
#openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open
#close = df['<CLOSE>'].copy() #for the case we want to enter trades at the close


##build window momentum features
for n in list(range(1,21)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n).fillna(0) #for trading with open
    #df[name] = df["<CLOSE>"].pct_change(periods=n) #for trading with close
    

#build date time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values


#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open
#df['retFut1'] = df['<CLOSE>'].pct_change(1).shift(-1).fillna(0) #if you wait until the close to enter the trade
#df = np.log(df+1)

#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#select the features (by dropping)
cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df.drop(cols_to_drop, axis=1, inplace=True)

#distribute the df data into X inputs and y target
X = df.drop(['retFut1'], axis=1)
y = df[['retFut1']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]


##########################################################################################################################
#set up the grid search and fit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true,y_pred) #spearman's rank correlation
    print (rho)
    return rho

def sharpe(y_true, y_pred):
    positions = np.where(y_pred> 0,1,-1 )
    dailyRet = pd.Series(positions).shift(1).fillna(0).values * y_true
    dailyRet = np.nan_to_num(dailyRet)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    return ratio

myscorer = None #uses the default r2 score, not recommended
#myscorer = "neg_mean_absolute_error"
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)

"""
INSTRUCTIONS
Use the Pipeline object to define a numeric_sub_pipeline.
The Pipeline parameters should be:
steps=a list of steps 
The list of steps should be:
steps=[('imputer', put a SimpleImputer function here)('scaler', put a StandardScaler function here)]
for the SimpleImputer select strategy='constant', fill_value=0
For guidance see:https://archive.is/hpwzH
"""

numeric_sub_pipeline = Pipeline([("imputer", SimpleImputer(strategy='constant', fill_value=0)),("scaler", StandardScaler())])

"""
INSTRUCTIONS
Use the Pipeline object to define a categorical_sub_pipeline.
The Pipeline parameters should be:
steps=a list of steps 
The list of steps should be:
steps=[('imputer', put a SimpleImputer function here)('onehot', put a OneHotEncoder function here)]
in OneHotEncoder set handle_unknown='ignore'
For guidance see: https://archive.is/hpwzH
"""

categorical_sub_pipeline = Pipeline([("imputer", SimpleImputer()),("onehot", OneHotEncoder(handle_unknown='ignore'))])
    
"""
INSTRUCTIONS
define numeric_features_ix index by using x_train.select_dtypes(...).columns
inside the dtypes include continuous floats only 
define categorical_features_ix index by using x_train.select_dtypes(...).columns
inside the dtypes include integers only
print(x_train.dtypes) to help yourself
For guidance see: https://archive.is/hpwzH
"""   

print(x_train.dtypes)
numeric_features_ix = x_train.select_dtypes(include=['float64']).columns
categorical_features_ix = x_train.select_dtypes(include=['int64']).columns

"""
INSTRUCTIONS
define a preprocessor column transformer by using the object ColumnTransformer
the ColumnTransformer parameters should be:
transformers=a list of 3-element tuples
remainder='passthrough' (to pass through --unchanged-- any remainder predictors not included in the numeric_features or the categorical_features)
For the 3-element tuples
transformers=[('num', put the numeric_sub_pipeline here, put the numeric_features_ix here),('cat', put the categorical_sub_pipeline here, put the categorical_features_ix here)]
Note: transformer 3-element tuples can be: ('name', function or pipeline, column_number_list or column_index)
For guidance see: https://archive.is/hpwzH
"""   

preprocessor = ColumnTransformer(transformers=[('num', numeric_sub_pipeline, numeric_features_ix),
                                               ('cat', categorical_sub_pipeline, categorical_features_ix)],
                                 remainder='passthrough')


ridge = Ridge(max_iter=1000) 

pipe = Pipeline(steps=[('preprocessor', preprocessor),('ridge', ridge)])


a_rs = np.logspace(-7, 0, num=20, endpoint = True)

param_grid =  [{'ridge__alpha': a_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

"""
INSTRUCTIONS
note that when using ColumnTransformer with a column_index
instead of fitting using as input x_train.values (an array)
we need to use x_train (a dataframe)
that has access to the column names

"""   

#grid_search.fit(x_train.values, y_train.values.ravel())
grid_search.fit(x_train, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_ridgereg.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train)> 0,1,-1 ) #POSITIONS

#dailyRet = pd.Series(positions).shift(1).fillna(0).values * x_train.ret1 #for trading at the close
dailyRet = pd.Series(positions).fillna(0).values * y_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated RidgeRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TrainCumulative"))


cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)
#positions2 = np.where(best_model.predict(x_test)> 0,1,-1 ) 
positions2 = np.where(grid_search.predict(x_test)> 0,1,-1 ) #POSITIONS

#dailyRet2 = pd.Series(positions2).shift(1).fillna(0).values * x_test.ret1 #for trading at the close
dailyRet2 = pd.Series(positions2).fillna(0).values * y_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated RidgeRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative"))

rho, pval = spearmanr(y_test,grid_search.predict(x_test)) #spearman's rank correlation: very small but significant

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  Rho={:0.6} PVal={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, rho, pval))


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(r'Results\%s.png' %("Residuals"))


#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])

"""
INSTRUCTIONS
since the pipeline has changed
you have to change the indexing of best_model
to access the model coefficients.

"""   
column_names = []
column_names = numeric_features_ix.values.tolist()
num_dummies = len(best_model[1].coef_.ravel().tolist())-len(column_names)
for i in range(1,num_dummies+1):
    column_names.append('dummies_'+str(i))
    
#plot the coefficients
importance = pd.DataFrame(zip(best_model[1].coef_.ravel().tolist(), column_names))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"))


"""""
QUESTIONS:
#What happens in the following code (look at the variable explorer):
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features_ix)], remainder='passthrough' )
x_train = X.iloc[0:10000]
x_train_out = transformer.fit_transform(x_train)

transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features_ix)] )
x_train = X.iloc[0:10000]
x_train_out = transformer.fit_transform(x_train)

A transformer is an object that has a fit and a transform method (and a fit_transform method)
A ColumnTransformer is a wrapper around a transformer, that 
outputs another transformer that just affects the features that you list.

A pipeline is another wrapper that inherits the methods of what is inside it.
A pipeline with a transformer inside can fit and transform.
A pipeline with a model inside can fit and predict.
A grid or random search object is a pipeline with a model inside, so it can fit and predict.
You can use a pipeline to fit and predict or to fit and transform, depending on the contents.

In conclusion:
x_train_out = preprocessor.fit_transform(x_train)
mypipe = Pipeline(steps=[('preprocessor', preprocessor)])
x_train_out_bis = mypipe.fit_transform(x_train)
mypipe_bis = Pipeline(steps=[('preprocessor', preprocessor),('ridge', ridge)])
x_train_preds = mypipe_bis.fit(x_train,y_train).predict(x_train)


"""""






