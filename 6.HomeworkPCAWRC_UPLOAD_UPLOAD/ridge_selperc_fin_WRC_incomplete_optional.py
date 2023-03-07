"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.

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
In this script you will incorporate a custom scorer (sharpe or cagr) 
directly into the feature selection step of the scikit-learn workflow.
For this we will use a combination of ColumnTransformer and FunctionTransformer
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
openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open
close = df['<CLOSE>'].copy() #for the case we want to enter trades at the close


##build window momentum features
for n in list(range(1,21)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
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
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_regression #Mutual information for a continuous target. (non linear)
from sklearn.feature_selection import f_regression #F-value between label/feature for regression tasks. (linear)
import detrendPrice 
import WhiteRealityCheckFor1 
import math

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

"""
INSTRUCTIONS:
We will be using SelectPercentile.
SelectPercentile has a parameter called score_func which accepts a scoring function,
as long as it has specific format.
The format has to be like that of f_regression (the model scoring function):
https://archive.is/qxCbT
Following this model, we have already filled in the missing code in fin_select below.

You will notice that it just returns 2 arrays that are global arrays: fin_arr and pval_arr.
So fin_select is complete.

The function that you need to fill in is select_prepare:
It prepares the 2 global arrays. 

Formally (but not functionally), 
select_prepare is a transforming function.
Why:
select_prepare is an argument of a function called FunctionTransformer that 
creates a scaler function called select_prep
which refers to select_prepare.
What does select_prepare have to do with scaling?
Nothing.

But:
Since select_prepare is a scaler function,
we can order it correctly in the pipeline.
select_prep is located in step 3 of the pipeline, 
right before feature selection in step 4.
But because select_prep is a scaling function 
it needs to return the "scaled" features.
So it just passes them through and returns them without change.
Meanwhile select_prep is doing the real work: 
filling in the global arrays that 
fin_select will use in the following step.

This is undoubtedly a practical trick to fool scikit-learn, because
as long as you pass the features along, 
you can string together as many fake scaler functions as you like,
one after the other, in the pipeline.

So what does select_prepare do while it passes the features (without changing them):
It sets up a loop where it test every regressor.
In the loop: 
the incoming model is fit to the regressor,
the model predicts,
the predictions are used to trade,
the trades are measured for SHARPE and CAGR
Finally the result (CAGR or SHARPE, you decide) is stored twice:
store it first into fin_arr, as a statistic and
store it again into pval_arr, as a p_value (requires conversion)

To convert the statistic into a p_value
you use the same trick used in logistic regression 
to turn a number into a probabiity:
you use the logistic function!

But you want the p_value, so
you subtract the probability from 1.

This is challenging homework, so
it is an optional homework
but if you can do it, 
it allows scikit-learn to do an enormous amount of work for you in no time,
thanks to its just in time c compiler.


"""
#global variables
fin_arr = np.zeros(x_train.shape[1])
pval_arr = np.zeros(x_train.shape[1])
counter = 0

def fin_select(X, y):
    #Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
    #Model: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
    return fin_arr, pval_arr

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def select_prepare(X, target):
    global fin_arr, pval_arr, counter
    fin_arr = np.zeros(X.shape[1])
    pval_arr = np.zeros(X.shape[1])
    scaler = StandardScaler()
    ridge = Ridge(max_iter=1000)
    #select_prepare is called multiple times, needs to run only once
    X_df = X.fillna(0)
    target = target.fillna(0)
    train_size = X_df.shape[0]//2
    X_train_df = X_df.iloc[:train_size]
    X_test_df = X_df.iloc[train_size:]
    X_train_arr = scaler.fit_transform(X_train_df) #need to scale here, why? will this scaling percolate to the rest of the pipeline?
    X_test_arr = scaler.transform(X_test_df) 
    retFut1_train_ser = pd.concat([X_train_df, target], axis=1)##### #do a join of X_train_df with the target, put the retFut1 series in retFut1_train_ser
    retFut1_test_ser = pd.concat([X_test_df, target], axis=1)##### #do a join of X_test_df with the target, put the retFut1 series in retFut1_test_ser
    if counter < 1:
        for i in range(X.shape[1]):
            ridge.fit(X_train_arr.reshape(-1,1),target.reshape(-1,1)) ##### #fit the  model to the scaled train data  (you will need to use .reshape(-1,1) on the input)
            preds = ridge.predict(X_test_arr.reshape(-1,1)) ##### #predict with the  model using scaled test data (you will need to use .reshape(-1,1) on the input)
            positions = np.where(preds > 0,1,-1 ) ##### #calculate the positions
            dailyRet = pd.Series(positions).shift(1).fillna(0).values * target ##### #calculate the daily returns of the system
            dailyRet = dailyRet.fillna(0)
            cumret = (1 + dailyRet).cumprod() - 1##### #calculate the cumulative returns
            cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1 ##### #calculate the cagr
            sharpe_ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet) #####  #calculate the sharpe ratio
            fin_arr[i] = cagr #use cagr or sharpe_ratio here
            pval_arr[i] = 1-sigmoid(cagr) #use cagr or sharpe_ratio here
    counter = counter + 1
    return X

#myscorer = None #uses the default r2 score, not recommended
#myscorer = "neg_mean_absolute_error"
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)

ridge = Ridge(max_iter=1000)

select_prep = FunctionTransformer(select_prepare, kw_args={'target': y_train.fillna(0)}) 

percentile = 50
selector = SelectPercentile(score_func=fin_select, percentile=percentile)
#selector = SelectPercentile(score_func=f_regression, percentile=percentile) 

numeric_sub_pipeline = Pipeline(steps=[
    ('select_prep', select_prep),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler',StandardScaler()),
    ('selector',selector)])

categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


print(x_train.dtypes)
numeric_features_ix = x_train.select_dtypes(include=['float64']).columns
categorical_features_ix = x_train.select_dtypes(include=['int64']).columns

#this code just turns an index numeric_features_ix 
#into a list of column numbers [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
numeric_features_no = [] 
for i in numeric_features_ix: 
    numeric_features_no.append(df.columns.get_loc(i))    

categorical_features_no = [] 
for i in categorical_features_ix: 
    categorical_features_no.append(df.columns.get_loc(i))    

#transformer 3-element tuples can be: ('name', function or pipeline, column_number_list or column_index)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, numeric_features_no),
        ('cat', categorical_sub_pipeline, categorical_features_no)], remainder='passthrough')
    
#the second time a ColumnTransformer is used, 
#a list of column numbers must be passed as parameter because 
#the x_train dataframe has been transformed to an array inside the pipeline
    
"""
INSTRUCTIONS
Define your pipeline with the custom_selector  as the first step in the pipeline because
the X input of select_prepare  needs to be a dataframe not an array, and
if you order the  custom_selector as the second step in the pipeline, the X input will be an array
"""

pipe = Pipeline(steps=[('selector', selector), ('preprocessor', preprocessor),('ridge', ridge)])#####

a_rs = np.logspace(-7, 0, num=20, endpoint = True)

param_grid =  [{'ridge__alpha': a_rs}]



grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

#grid_search.fit(x_train.values, y_train.values.ravel())
grid_search.fit(x_train.fillna(0), y_train.fillna(0).values.ravel())

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
positions = np.where(grid_search.predict(x_train.fillna(0))> 0,1,-1 ) #POSITIONS

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
positions2 = np.where(grid_search.predict(x_test.fillna(0))> 0,1,-1 ) #POSITIONS

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
plt.close("all") 

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])

#Detrending Prices and Returns and white reality check
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()


column_names = []
num_numeric = int(len(numeric_features_ix)*percentile/100)
for i in range(1,num_numeric+1):
    column_names.append('numeric_features_'+str(i))
num_dummies = len(best_model[1].coef_.ravel().tolist())-num_numeric
for i in range(1,num_dummies+1):
    column_names.append('dummies_'+str(i))

##plot the coefficients
importance = pd.DataFrame(zip(best_model[1].coef_.ravel().tolist(), column_names))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"))

"""
QUESTIONS

Regarding select_prepare:
Why are we using join in the lines that say:
retFut1_train_ser = ...
retFut1_test_ser = ...
hint: why use a counter?

Regarding select_prepare:
Why are we scaling the data?
X_train_arr = scaler.fit_transform(X_train_df) 
Will this scaling affect the returned X?

"""


