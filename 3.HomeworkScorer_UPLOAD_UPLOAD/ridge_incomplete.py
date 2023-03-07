"""
Fill in the missing code. The lines with missing code have the string "#####" or '*'
"INSTRUCTIONS" comments explain how to fill in the mising code.
the outputfile.txt has the printouts from the program.
"RESULTS" comments explain what results to expect from the program.
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
By now you have seen a simple algorithmic trading workflow 
which we covered in the PandasHomework.
You have slso seen a simple machine learning workflow 
which we covered in the CocaColaHomework and the first two lessons.

In this homework, 
you will put the two workflows together.
You will incorporate 3 performance criteria
to the Scikit-Learn workflow:
1. The Sharpe Ratio
2. The Information Coefficient (Spearman's rho)
3. The new Phik correlation Coefficient.

Since Spearman's rho and the Phik correlation criterion
require thousands of samples to lower the sampling error,
we will train our model with 10 thuousand samples,
in such a way that by setting cv=5 in 
RandomizedSearchCV or GridSearchCV
each validation fold will have 2000 samples.
The test data too will have 2000 samples.

To obtain recent enough data despite the long lookback
you are going to use USDCAD currency data sampled every 3 hours:
    
USDCAD_H3_200001030000_202107201800.csv

Note:
1. we enter positions immediately after open instead of right before the close
2. we calculate the system returns by multiplying positions at time t by target returns at time t (more intuitive)
before, we were entering positions immediately before the close and
we calculated system returns by multiplying positions at time t-1 by target returns at time t
Trading the open is attractive for many reasons: e.g. 
https://archive.is/PIuUW
https://archive.is/nt9lE

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

#uncomment to direct the print standard output to the outputfile.txt
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

"""
INSTRUCTIONS
build window feaures
Use df.series.pct_change(periods= * ) to calculate 20 series
based on the <OPEN> price:
ret1 will have 1 period percent returns
ret2 will have 2 period percent returns
ret3 will have 3 period percent returns
etc.
Save these 20 series inside the df as columns with names:
"ret1", "ret2", "ret3", etc.

These return features are called window momentum features.
Momentum features measure perfornmance (high return) over a period.
They are called momentum features because 
high performance has been shown to have inertia, 
to extend into the future,though for how long
is umpredictable.
Momentum features resemble lag features in that 
they carry forward information  about the past.
But they are not exactly the same.
 
"""

##build window momentum features
for n in list(range(1,21)):
    name = f'ret{n}'
    df[name] = df['<OPEN>'].pct_change(periods=n).fillna(0)


"""
INSTRUCTIONS
Building date time features
recalling from the CocaCola homework the use of df.index.quarter.values generate a column of quarters, and
as per:
https://archive.is/2kpci
extract the hour from the index and save it in the df as a column with name "hour".
extract the dayofweek from the index and save it in the df as a column with name "day".
Use the Pandas get_dummies function  to get the hour and day dummies.
Only keep the dummies from "day" and "hour", not the original "day" and "hour" features.

"""

#build date time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(['hour', 'day'], axis=1, inplace=True)

"""
INSTRUCTIONS
build target so as to trade right after the open (we know today's open)
Use df['<OPEN>'].pct_change(*).shift(*) (substituting * by -n or n as appropriate, with n=1 or 2 or 3 etc.)
to calculate the "one period forward returns" (the prediction target).
Save the forward returns as a column in the df with the name:
 "retFut1".
"""
 
#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1) #if you enter the trade right after the open
#df = np.log(df+1)

#Since we are trading right after the open, we know today's open but
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"] #optional
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
#Exploratory Data Analysis
from scipy.stats import pearsonr
from scipy.stats import spearmanr

"""
INSTRUCTIONS
It is always good practice to visualize the data.
We are going to look at the correlations of the predictors with retFut1
This could help us select the best predictors intuitively
Inside plot_corr add code to plot the pairwise correlations
Use one of the models here: https://archive.is/GG25N
Optionally, substitute the regular ax with a heatmap following the example here: https://archive.is/rVHhF

"""

def plot_corr(corr, size=5, title="Pearson correlation"):
    """Function plots a graphical correlation matrix dataframe 
    Input:
        df: pandas corr DataFrame from e.g. corr = df.corr(method='pierson')
        size: vertical and horizontal size of the plot
        title: title of the plot
    """
    fig, ax = plt.subplots(figsize=(size, size))
    #optionally, substitute the regular ax with an sns.heatmap (the regular ax has horrible colors)
    ax = sns.heatmap(corr)
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    #plt.tight_layout() #optionally can use this instead of rotation='vertical'


"""
INSTRUCTIONS
Use the pandas dataframe method select_dtypes to exclude the 'uint8' categorical data by
using the parameter exclude. Put the result in df_filtered
"""

#get the non-categorical data  by excluding the uint8 dtype
#we do this because neither the spearman nor the pearson correlations can deal with categorical variables    
df_filtered = df.select_dtypes(exclude=['u1'])

#Plot the matrix containing the pair-wise spearman coefficients
#ideally we want to pay special attention to the correlation between each period return and retFut1    
corr = df_filtered.corr(method='spearman').round(2)
plot_corr(corr, size=20)
#plt.show()
plt.savefig(r'Results\%s.png' %("Spearman Correlation Matrix"))

#Plot the  matrix containing the pair-wise pearson coefficients
##ideally we want to pay special attention to the correlation between each period return and retFut1    
corr = df_filtered.corr(method='pearson').round(2)
plot_corr(corr, size=20)
#plt.show()
plt.savefig(r'Results\%s.png' %("Pearson Correlation Matrix"))

#You want to plot the significance of the correlations
#This is optional in our case because we have tons of data.
def calculate_pvalues(df, method='pearson'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        if method=='pearson':
            for c in df.columns:
                pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
        else:
            for c in df.columns:
                pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)                
    return pvalues

#plot the significance values
print('pearson significance values')
print(calculate_pvalues(df_filtered, method='pearson'))
print('spearman significance values')
print(calculate_pvalues(df_filtered, method='spearman'))

"""
RESULTS:
ret1 seems to have the largest absolute spearman or pearson correlation with retFut1
"""
plt.close("all")
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

"""
INSTRUCTIONS
Now we are going to introduce financial performance metrics into the scikit-learn worflow.
We are going to do this at the parameter optimization stage by using the scikit-learn make_scorer utility
(scikit-learn does not easily allow model loss functions to  be customized).
We have already programmed the sharpe ratio for you.
You need to fill in the information_coefficient (above) which
uses the spearman rank correlation documented here:
https://archive.is/l8Ypt
For inspiration in writing the missing code in information_coefficient, 
we suggest you look at the phi_k function 
in logistic_regression_with_grid_incomplete.py
For completeness sake, include both the rho and the pval, though 
only the rho is returned.

After you write the missing code in information_coefficient,
you will use the make_scorer utility (below) to create two custom scorer objects:
myscorerIC which will use the information_coefficient function and
myscorerSharpe which will use the sharpe function.
The instructions for the make_scorer utility are here:
https://archive.is/v987w
"""

def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true, y_pred)##### #spearman's rank correlation
    print(rho)
    return rho

def sharpe(y_true, y_pred):
    positions = np.where(y_pred> 0,1,-1 )
    dailyRet = pd.Series(positions).shift(1).fillna(0).values * y_true
    dailyRet = np.nan_to_num(dailyRet)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    return ratio


#myscorerNone = None #uses the default r2 score, not recommended
#myscorer = "neg_mean_absolute_error"
myscorerIC = make_scorer(information_coefficient)
myscorerSharpe = make_scorer(sharpe)

imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
#we turn off scaling because ..
scaler = StandardScaler(with_mean=False, with_std=False)
ridge = Ridge(max_iter=1000) 

pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("ridge", ridge)])

"""
INSTRUCTIONS
Define a np.logspace for the parameter grid pair: 'ridge__alpha': rs
The search space for C and lamda (=1/C) is as large as:
[0.0001, 0.001, 0.01,0.1,1,10,100,1000,10000], which is huge.
instead make the a_rs smaller and similar to:

array([1.00000000e-07, 2.33572147e-07, 5.45559478e-07, 1.27427499e-06,
       2.97635144e-06, 6.95192796e-06, 1.62377674e-05, 3.79269019e-05,
       8.85866790e-05, 2.06913808e-04, 4.83293024e-04, 1.12883789e-03,
       2.63665090e-03, 6.15848211e-03, 1.43844989e-02, 3.35981829e-02,
       7.84759970e-02, 1.83298071e-01, 4.28133240e-01, 1.00000000e+00])
"""

a_rs = np.logspace(-7, 0, num=20, endpoint=True)

param_grid =  [{'ridge__alpha': a_rs}]

"""
INSTRUCTIONS
Having defined your custom scorer object
you need to insert it as the value of the scoring parameter of
RandomizedSearchCV or
GridSearchCV
Set up the RandomizedSearchCV and save the output in grid_search
Set up the GridSearchCV and save the output in grid_search (but comment it out)
You can comment and uncomment these lines to try both ways to see the results.

Try the 3 scorers (None, myscorerIC and myscorerSharpe) and notice any changes
by looking at results_ridgereg.csv (the test mean score)
Changes between the 3 scorers will be small because 
of the very large amount of training data.

"""

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorerSharpe, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=*, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

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
#positions = np.where(best_model.predict(x_train.values)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #POSITIONS

#dailyRet = fAux.backshift(1, positions) * x[:train_set,0] # x[:train_set,0] = ret1
dailyRet = pd.Series(positions).shift(1).fillna(0).values * x_train.ret1
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
#positions2 = np.where(best_model.predict(x_test.values)> 0,1,-1 )
positions2 = np.where(grid_search.predict(x_test.values)> 0,1,-1 ) #POSITIONS


dailyRet2 = pd.Series(positions2).shift(1).fillna(0).values * x_test.ret1
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated RidgeRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative"))

rho, pval = spearmanr(y_test,grid_search.predict(x_test.values)) #spearman's rank correlation: very small but significant

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  Rho={:0.6} PVal={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, rho, pval))

"""
RESULTS
see:  outputfile.txt 
Spearman's RHO between actual and predicted is low to moderate but significant, financial metrics are reasonable
Good for a simple model such as this one
"""

"""
INSTRUCTIONS:
We will now plot the residuals,
first we need to calculate them:
save the residuals in residuals
"""


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test.values)
residuals = true_y - pred_y

"""
RESULTS
Now we plot the residuals.
The left panel plots the residual distribution versus the normal distribution.
The residuals are slightly non-normal.
In practice, this implies that the model is making more
large errors than "normal."
The right panel of the figure plots the autocorrelation
coefficients for the first 10 residuals lags, 
pointing to a significant negative correlation at lag 1. 
These defects mean that even though the model's predictions are not biased,
they will spread over a larger than needed interval.

"""

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

"""
RESULTS
The test for autocorrelation is the Ljung-Box test. 
If the p-value of the test is greater than the required significance (>.05) (not the case for this model)
we can conclude that the residuals are independent and very much like white noise. 
Otherwise, the residuals are autocorrelated and are not strongly stationary. (they are autocorrelated for this model)

"""

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])


"""
RESULTS
Now we plot the coefficients to see which ones are important for this model.
This model gives little importance to the hour and day coefficients.
Since the model inputs (the returns and dummies) are more or less already scaled:
When plotting the importance of the coefficients,
we need not divide the coefficients 
by their corresponding input standard deviations to compare their importance.

Note that to obtain the coefficients,
we queried the best_model but
if you print best model you will see that 
best_model is a pipeline with a number of steps,
and its index starts at zero.
To obtain the coefficients, 
we obtain estimator and
query its properties.
To do this properly, you consult the documentation in:
https://archive.is/DzuwU

"""

#plot the coefficients
importance = pd.DataFrame(zip(best_model[2].coef_, x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"))


"""
QUESTIONS (no need to turn in any answers, but do the research and find out):
    
What happens in the following code (look at the variable explorer):
pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='constant', fill_value=0)), ('s', StandardScaler())])
x_train_out = pipeline.fit_transform(x_train)


What is the difference between regular linear regression and ridge regression?

Does regularizing this regression require us to scale the input or not?

Regarding the line:
scaler = StandardScaler(with_mean=False, with_std=False)
what do these parameters do:
with_mean=False, with_std=False
Why are we doing this?
Hint: part of the answer is that returns are already relatively scaled and centered, but
that is not the whole answer.

Why only keep the dummies from "day" and "hour", not the original "day" and "hour" features?

Make sure you understand the difference between a model's loss function and the scorer.

Make sure you understand fully which scorer is being used 
when the scoring= None
by looking at the documentation of scikit-learn's ridge regression model:
https://archive.is/DzuwU

Make sure you understand what loss function ridge regression is using when you change scorers.
Read:
https://archive.is/nQ6Ha

Note:
the annualization factor used in the script 
is the default 252 (for 252 trading days per year).
the USDCAD data is approximately 8 time units per day (it varies)
so the correct annualization factor should be modified to 252*8. 
this correction does not alter the optimization results.


"""
