"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
Actually, we added np.random.seed() to fix the results, so you can check them.
"""

"""
In this notebook you will learn an important piece of information:
how to take advantage of scikit-learn'a parameter optimization routines 
to optimize parameters in the Ta-lib functions.
For this we will use a combination of FunctionTransformer and ColumnTransformer.

The Ta-lib functions are very useful for feature engineering but
you have to know what values to set to the various Ta-lib function parameters.

Most parameters in the Ta-lib functions have default values but
the values are optimal only for daily trading, and 
only for an index like the S&P500 or a typical stock.
If your model is predicting in a different timeframe or
if your model is predicting an atypical stock, then
you will probably have to search to optimize the parameters of the Ta-lib functions.
So here you will learn how to use scikit-learn to help you optimize Ta-lib.

As an example, we will use a smoothing Ta-lib function: EMA which
stands for exponential moving average or "exponential smoothing".

In finance, smoothing techniques are sometimes used to remove noise from the input data.
In this notebook you will use the scikit-learn utility FunctionTransformer
to extend the scikit-learn's input pre-processing capabilities
to apply and optimize exponential smoothing.

Exponential smoothing is implemented in both Pandas and in Ta-lib.
We shall see both implementations.

In Pandas:
pandas.DataFrame.ewm, also known as
panda's exponential moving window,
that is documented here:
https://archive.is/p92xp
https://archive.is/HhPAu

In Pandas' ewm: 
there is averaging implemented by the mean function (ewm(span=span, adjust=True).mean()) 
averaging is over a moving window (ewm) of length equal to span
the averaging is exponentially weighted and
the weights decay exponentially so as to
decrease the importance of oldest input in the window (input at the start or left-edge of the window) 
increase the importance of newest input in the window  (input at the end or right-edge of the window)
For more information, you can (optionally) read the material in the EWM_PANDAS folder.
Exponential smoothing can also be used for prediction as explained in ExponentialSmoothing.pptx
but in this notebook we use it for noise reduction only.

In Ta-lib:
Exponential smoothing in Ta-lib is implemented with the function:
real = EMA(close, timeperiod=30)
that is documented here:
https://archive.is/CvJo0

Clearly, applying any kind of smoothing,
including Panda's and Ta-lib's ewm,
requires the preservation of the original order of the input samples.
Thefore, Panda's and Ta-lib's ewm must be used in combination with TimeSeriesSplit.
Here we show you how to do this also.
In any case, when it comes to financial series prediction,
it is a good idea to use TimeSeriesSplit, whether
you use smoothing or not.

So far you have seen the operation of two data fold splitters
during the cross-validation process.
In On_regression_start_part1.pptx slide 30,
you saw the operation of the regular Kfold splitter.
In On_regression_start_part2.pptx slide 16
you saw the operation of a special splitter called TimeSeriesSplitter.

These two splitters are set side by side here:
/
(the pdf of this file is included with this homework:
Visualizing cross-validation — scikit-learn 0.24.2 documentation.pdf)
where the training folds are shown in blue, the validation folds in red.
Kfold is the first splitter shown, 
TimeSeriesSplitter is the last one at the bottom of the page.

The two splitters have one similarity and one difference.

The similarity is that they both have one validation fold per iteration.

The difference is that 
if you think of the data being split as a time series,
the TimeSeriesSplitter splitter creates training folds from strips of data that are continuous in time.
The Kfold splitter creates training folds by splicing together two strips that
were not originally adjacent in time.
So Kfold does not respect the original time structure.
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
#close = df['<CLOSE>'].copy() #for the case we want to enter trades at the close


##build window momentum features
for n in list(range(1,30)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n)#for trading with open
    #df[name] = df["<CLOSE>"].pct_change(periods=n)#for trading with close
    

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
from sklearn.model_selection import TimeSeriesSplit
import detrendPrice 
import WhiteRealityCheckFor1 
from sklearn.preprocessing import FunctionTransformer
import talib as ta

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
INSTRUCTIONS
We have already programmed the Pandas and Ta-lib versions of exponential smoothing for you.
Note that inside the pandas_ewm_smoother function, 
you call the corresponding Pandas function,
as per ExponentialSmoothing.pptx with adjust=True.
Similarly inside the talib_ewm_smoother function,
you call the corresponding Ta-lib function.
We will use pandas_ewm_smoother and talib_ewm_smoother inside the FunctionTransformer object.

"""

#PANDAS exponential smoothing:
def pandas_ewm_smoother(x_train, span=None):
    x_train = pd.DataFrame(x_train)
    x_train_smooth = x_train.ewm(span=span, adjust=True).mean()
    return  x_train_smooth.values

#Ta-lib exponential smoothing:
def talib_ewm_smoother(x_train, span=None):  #assumes x_train is an array
    w = np.arange(x_train.shape[0])
    for i in range(0,x_train.shape[1]):
        a = ta.EMA(x_train[:,i], timeperiod=span) #x_train is an array
        w = np.c_[w,a]
    return w[:,1:]


#myscorer = None #uses the default r2 score, not recommended
#myscorer = "neg_mean_absolute_error"
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)

"""
INSTRUCTIONS
For proper functioning of exponetial smoothing,
make sure that TimeSeriesSplit is being activated during RandomizedSearchCV and GridSearchCV
to preserve the chronological order of the input series during the construction of the folds.
TimeSeriesSplit is documented here:
https://archive.is/8FZtM
Change the value of split from split=5 to the split created by TimeSeriesSplit with
n_splits set to 5 and max_train_size set to 2000.

"""

#when using smoother, use TimesSeriesSplit
# split = 5
split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window
# split = TimeSeriesSplit(n_splits=5)

"""
INSTRUCTIONS
To use the exponential smoothing functions you need to use the FunctionTransformer object.
The instructions for the FunctionTransformer object are here:
https://archive.is/hsurj

Instantiate a FunctionTransformer object.
Assign the ewm_smoother function or the pandas_smoother_function to the FunctionTransformer object's func parameter.
Save the resulting object in "smoother".
Do this twice, once for pandas_ewm_smoother and again for talib_ewm_smoother.
Each time, assign the resulting object to smoother.
What the FunctionTranformer object is doing is wrapping around the original function 
and provide it with a fit_transform method.

"""

smoother = FunctionTransformer(pandas_ewm_smoother)
# smoother = FunctionTransformer(talib_ewm_smoother)

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('smoother', smoother),
    ('imputer2', SimpleImputer(strategy='constant', fill_value=0))])
    
categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
print(x_train.dtypes)
numeric_features_ix = x_train.select_dtypes(include=['float64']).columns
categorical_features_ix = x_train.select_dtypes(include=['int64']).columns

#Note: transformer 3-element tuples can be: ['name', function or pipeline, column_number_list or column_index]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, numeric_features_ix),
        ('cat', categorical_sub_pipeline, categorical_features_ix)], remainder='passthrough')

ridge = Ridge(max_iter=1000) 

pipe = Pipeline(steps=[('preprocessor', preprocessor),('ridge', ridge)])


a_rs = np.logspace(-7, 0, num=20, endpoint = True)

"""
INSTRUCTIONS
Note that pandas_ewm_smoother and talib_ewm_smoother have an argument called span that needs to be optimized.
Put possible span values are in spans_rs, for guidance see: https://archive.is/RPs90
Note that you must map each value of span in a separate mapping.
"""

spans_rs = [{'span': 6}, {'span': 10}, {'span': 20}, {'span': 50}]

"""
The tricky part is to construct the parameter grid correctly.
Start by proposing a key like 'preprocessor__kw_args' (it is the wrong key but close enough)
You will get an error.
Following the instructions of the error message:
Print out the list of possible keys by printing out: grid_search.get_params().keys()
Find the correct key and substitute it instead of 'preprocessor__kw_args'
"""


#helpful for setting up param_grid: grid_search.get_params().keys()
param_grid =  [{'preprocessor__num__smoother__kw_args':  spans_rs, 'ridge__alpha': a_rs}]


# grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)



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

#Detrending Prices and Returns and white reality check
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()


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

"""
QUESTIONS:
FunctionTransformer wraps a custom function into an object that has the fit_transform method:
When you run the following code, what do you get?:
    
def pandas_ewm_smoother(x_train, span=10):
    x_train = pd.DataFrame(x_train)
    x_train_smooth = x_train.ewm(span=span, adjust=True).mean()
    return  x_train_smooth.values
    
smoother = FunctionTransformer(pandas_ewm_smoother)
x_train_in = x_train.drop(['hour','day'], axis=1)
x_train_out = smoother.fit_transform(x_train_in)

When predicting returns, is it better apply exponential smoothing to prices or to returns?
Try both by changing

##build window momentum features
for n in list(range(1,30)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n)
    
to

for n in list(range(1,30)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n)



"""