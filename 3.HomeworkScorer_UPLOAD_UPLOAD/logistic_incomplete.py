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
NOTE: Phik is slow: this file takes 5 minutes to run

Since Spearman's rho and the Phik correlation criterion
require thousands of samples to lower the sampling error,
we will train our model with 10 thuousand samples,
in such a way that by setting cv=5 in 
RandomizedSearchCV or GridSearchCV
each validation fold will have 2000 samples.
The test data too will have 2000 samples.

To obtain recent enough data despite the long lookback
you are going to use USDCAD currency data sampled every 3 hours here:
    
USDCAD_H3_200001030000_202107201800.csv

Note:
1. we enter positions immediately after open instead of right before the close
2. we calculate the system returns by multiplying positions at time t by target returns at time t (more intuitive)
before, we were entering positions immediately before the close and
we calculated system returns by multiplying positions at time t-1 by target returns at time t
Trading the open is attractive for many reasons: 
e.g. 
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


"""
INSTRUCTIONS
Let us now build window features
Use df.pct_change(periods= * ) to calculate 20 series
based on the <OPEN> price:
ret1 will have 1 period percent returns
ret2 will have 2 period percent returns
ret5 will have 5 period percent returns
ret20 will have 20 period percent returns.
Save these 4 series inside the df as columns with names:
"ret1", "ret2", "ret5", ..."ret20".

These return features are called window momentum features.
Momentum features measure perfornmance (high return) over a period.
They are called momentum features because 
high performance has been shown to have inertia, 
to extend into the future,though for how long
is umpredictable.
Momentum features resemble lag features in that 
they carry forward information  about the past.
But they are not the same.


"""
#build window features, assuming we know today's open. Include fillna(0)
for n in list(range(1,21)):
    name = f'ret{n}'
    df[name] = df['<OPEN>'].pct_change(periods=n).fillna(0)


"""
INSTRUCTIONS
Building date time features
recalling from the CocaCola homework the use of df.index.quarter.values generate a column of quarters, and
as per:
https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.dayofweek.html
extract the hour from the index and save it in the df as a column with name "hour".
extract the dayofweek from the index and save it in the df as a column with name "day".

"""

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)


"""
INSTRUCTIONS
build target so as to trade right after the open (we know today's open)
Use df['<OPEN>'].pct_change(*).shift(*) (substituting * by -n or n as appropriate, with n=1 or 2 or 3 etc.)
to calculate the "one period forward returns" (the prediction target).
Save the forward returns as a column in the df with the name:
 "retFut1".
"""

#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open


"""
INSTRUCTIONS
transform target
For logistic regression you need to change the targets 
from continuus (returns) to categorical (0, 1)
Use the np.where construction shown
in moving_average_crossover_simple_incomplete.py (HomeworkPandas)
to effect this transformation
"""

#transform the target
df['retFut1_categ'] =  np.where((df['retFut1'] > 0), 1, 0)
        
#Since we are trading right after the open, 
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#select the features (by dropping)
cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df_filtered = df.drop(cols_to_drop, axis=1)

#distribute the df data into X inputs and y target
X = df_filtered.drop(['retFut1', 'retFut1_categ',"hour","day"], axis=1) #these drops are note optional, why?
y = df_filtered[['retFut1_categ']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]

##########################################################################################################################
#Exploratory Data Analysis
# as per https://archive.is/TksK7

"""
It is always good practice to visualize the data.
We are going to look at the phik correlations of the predictors with retFut1_categ
This could help us select the best predictors intuitively
Until the invention of phik, this was very hard to do 
with categorical targets or categorical predictors

"""

import phik
from phik.report import plot_correlation_matrix
from phik import report

#list of continuous numerical variables
interval_cols = df_train.select_dtypes(include=['float64']).columns.values.tolist()
#Plot the correlation matrix containing the pair-wise phik coefficients
phik_overview = df_train.phik_matrix(interval_cols=interval_cols)
plot_correlation_matrix(phik_overview.values, 
                        x_labels=phik_overview.columns, 
                        y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map="Greens", 
                        title=r"correlation $\phi_K$", 
                        fontsize_factor=1.5, 
                        figsize=(20, 20))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Correlation Matrix"))

#Plot the  significances of the above phik correlations
#this is optional because we have lots of data. 
#You want to see significances colored in dark green, which are the highest significances.
significance_overview = df_train.significance_matrix(interval_cols=interval_cols)
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(20, 20))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Correlation Matrix Significance"))


#The global correlation coefficient is a useful measure expressing the total correlation 
#of one variable to all other variables in the dataset. 
#This gives us an indication of how well one variable can be modeled 
#using the other variables.

global_correlation, global_labels = df_train.global_phik(interval_cols=interval_cols)

plot_correlation_matrix(global_correlation, 
                        x_labels=[''], y_labels=global_labels, 
                        vmin=0, vmax=1, figsize=(5,10),
                        color_map="Greens", title=r"$g_k$",
                        fontsize_factor=1.5)
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Global Correlations"))

#the outlier significance matrix helps us understand better the phik correlation 
#between categorical and continuous variables and 
#between categorical and categorical variables.

var_1 = "hour" 
var_2 = "retFut1_categ"

outlier_signifs, binning_dict = df_train[[var_1, var_2]].outlier_significance_matrix(interval_cols=[], 
                                                                        retbins=True)

zvalues = outlier_signifs.values
xlabels = outlier_signifs.columns
ylabels = outlier_signifs.index

plot_correlation_matrix(zvalues, x_labels=xlabels, y_labels=ylabels, 
                        x_label=var_2,y_label=var_1,
                        vmin=-5, vmax=5, title='outlier significance',
                        identity_layout=False, fontsize_factor=1.2, 
                        figsize=(7, 5))
#plt.show()
plt.savefig(r'Results\%s.png' %("Outlier Significance Matrix for Hour"))

var_1 = "day"
var_2 = "retFut1_categ"

outlier_signifs, binning_dict = df_train[[var_1, var_2]].outlier_significance_matrix(interval_cols=[], 
                                                                        retbins=True)

zvalues = outlier_signifs.values
xlabels = outlier_signifs.columns
ylabels = outlier_signifs.index

plot_correlation_matrix(zvalues, x_labels=xlabels, y_labels=ylabels, 
                        x_label=var_2,y_label=var_1,
                        vmin=-5, vmax=5, title='outlier significance',
                        identity_layout=False, fontsize_factor=1.2, 
                        figsize=(7, 5))

#plt.show()
plt.savefig(r'Results\%s.png' %("Outlier Significance Matrix for Day"))

#a convenience function that allows us to generate all of the above with a single line of code:
#rep = report.correlation_report(df_train, significance_threshold=3, correlation_threshold=0.5)
#print(rep)

plt.close("all")

"""
RESULTS
If you look at the Phik correlation between each of the 21 retNs and retFut1_categ 
it is possible to identify a subset of 6 retNs that are the best ones.
Run the model with this subset of 6 retNs and
the results of the logistic regression will be better.
"""

##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
import phik
from phik.report import plot_correlation_matrix
from scipy.special import ndtr
from sklearn.impute import SimpleImputer


def phi_k(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
        phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
        phi_k_p_val = 1 - ndtr(phi_k_sig) 
    except:
        phi_k_corr = 0
        phi_k_p_val = 0
    print(phi_k_corr)
    return phi_k_corr


"""
INSTRUCTIONS
Now we are going to introduce financial performance metrics into the scikit-learn worflow.
We are going to do this at the parameter optimization stage by using the scikit-learn make_scorer utility
(scikit-learn does not easily allow model loss functions to  be customized).
We have already programmed the phi_k function (above) that calculates the phik correlation for you.
For completeness sake, we included both the phi_k and the phi_k_p_val, though 
only the phi_k_corr is returned.
The use of the try except block is optional.
You can use this phi_k custom scorer function code
as inspiration for filling the missing code 
in the information_coefficient custom scorer located 
in ridge_incomplete.py

You will now use the make_scorer utility to create one custom scorer object with name:
myscorerPhik which will use this phi_k function.
The instructions for the make_scorer utility are here:
https://archive.is/v987w
"""

#myscorerNone = None #use default accuracy score
myscorerPhik = make_scorer(phi_k, greater_is_better=True)

imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
scaler = StandardScaler(with_mean=False, with_std=False)
logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

pipe = Pipeline([("imputer", imputer),("scaler", scaler), ("logistic", logistic)])

"""
INSTRUCTIONS
Define a np.logspace for the parameter grid pair: 'logistic__C': c_rs
The search space for C and lamda (=1/C) is as large as:
[0.0001, 0.001, 0.01,0.1,1,10,100,1000,10000], which is huge.
Instead we want c_rs to be something more reduced like:
array([1.00000000e+03, 4.28133240e+02, 1.83298071e+02, 7.84759970e+01,
       3.35981829e+01, 1.43844989e+01, 6.15848211e+00, 2.63665090e+00,
       1.12883789e+00, 4.83293024e-01, 2.06913808e-01, 8.85866790e-02,
       3.79269019e-02, 1.62377674e-02, 6.95192796e-03, 2.97635144e-03,
       1.27427499e-03, 5.45559478e-04, 2.33572147e-04, 1.00000000e-04])    

"""

c_rs = np.logspace(3, -4, num=20, endpoint=True)
#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)
p_rs= ["l1", "l2"]

param_grid =  [{'logistic__C': c_rs, 'logistic__penalty': p_rs}]

"""
INSTRUCTIONS
Having defined your custom scorer object
you need to insert it as the value of the scoring parameter of
RandomizedSearchCV or
GridSearchCV (which is commented out, we include it so you can play with it)
Set up the RandomizedSearchCV and save the output in grid_search
Set up the GridSearchCV and save the output in grid_search (but comment it out)
You can comment and uncomment these lines to try both ways to see the results.

Try the 2 scorers (None, myscorerPhik) and notice any changes
by looking at results_logisticreg.csv (mean_test_score)
Changes between the 2 scorers will be small because 
of the very large amount of training data.

Run the program a few times, with and without the "day" and  "hour" dummies.
"""

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorerPhik, return_train_score=True) #####
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorerPhik, return_train_score=True) #####

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticreg.csv")


#########################################################################################################################

# Train Set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train.values)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #POSITIONS

#dailyRet = pd.Series(positions).shift(1).fillna(0).values * x_train.ret1 #for trading at the close
dailyRet = pd.Series(positions).fillna(0).values * df_train.retFut1 #for trading right after the open
dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated LogisticRegression on currency: train set')
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

#dailyRet2 = pd.Series(positions2).shift(1).fillna(0).values * x_test.ret1 #for trading at the close
dailyRet2 = pd.Series(positions2).fillna(0).values * df_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated LogisticRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative2"))

#metrics
accuracy_score = accuracy_score(y_test.values.ravel(), grid_search.predict(x_test.values))

#If this figure does not plot correctly select the lines and press F9 again
arr1 = y_test.values.ravel()
arr2 = grid_search.predict(x_test.values)
dfc = pd.DataFrame({'y_true': arr1, 'y_pred': arr2})
phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
significance_overview = dfc.significance_matrix(interval_cols=[])
phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
phi_k_p_val = 1 - ndtr(phi_k_sig) 
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(7, 5))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Significance2"))

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, accuracy_score))

"""
RESULTS
with myscorer = make_scorer(phi_k, greater_is_better=True)
Out-of-sample: CAGR=0.0215579 Sharpe ratio=0.672315 maxDD=-0.0545567 maxDDD=342 Calmar ratio=0.395146  phi_k_corr=0.12189 phi_k_p_val=0.000252484  accuracy_score=0.541
The CAGR of 2.15% is positive, the Sharpe ratio of .67 is ok.
Phik correlation (phi_k_corr=0.12) is small but positive and statistically significant (phi_k_p_val=0.000252484).
accuracy_score=0.541 is good (anything above .50 is welcome)
The results are surprisingly good, given this extremely simple model.

"""
"""
INSTRUCTIONS
We will now plot the residuals,
first we need to calculate them:
save the residuals in residuals
"""

#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test.values)
residuals = true_y - pred_y


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
plt.savefig(r'Results\%s.png' %("Residuals2"))
plt.close("all")

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])



"""
RESULTS
Now we plot the coefficients to see which ones are important for this model.
This model gives little importance to the hour and day coefficients, but more than the ridge model.
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
https://archive.is/nzCKd

"""

#plot the coefficients
importance = pd.DataFrame(zip(best_model[2].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients2"))

#In the future, you may need a classifier's continuous predictions, 
#especially when using the Alphalens evaluator, later on in the course, where
#you can use a model's predictions as input
#You can construct the classifier's continuous predictions 
#from the prediction probabilities obtained from the model's predict_proba method:

logistic = LogisticRegression(max_iter=1000, solver='liblinear') 
logistic.fit(x_train, y_train)
pred_proba = logistic.predict_proba(x_test) #compare with pred_categ = logistic.predict(x_test)
df = pd.DataFrame(pred_proba, columns=["downs","ups"])
df["continuous_predictions"] = np.where(df["ups"]>df["downs"], df["ups"], -1*df["downs"])

#Not all scikit-learn classifiers have a predict_proba method.
#If your classifier lacks a predict-proba method, 
#you can use the following scikit-learn classifier wrapper:

from sklearn.calibration import CalibratedClassifierCV
model = CalibratedClassifierCV(logistic) 
model.fit(x_train, y_train)
pred_class = model.predict(x_test)
model.predict_proba(x_test)



"""
QUESTIONS (no need to turn in any answers, but do the research and find out):
    

Is logistic regression a linear model or a non-linear one?

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

Why can't you use the Spearman's rho or the Sharpe ratio as a scorer 
of RandomizedSearchCV or GridSearchCV 
in the case of logistic regression?

Make sure you understand fully which scorer is being used 
when the scoring parameter has the value None
by looking at the documentation of scikit-learn's logistic regression model:
https://archive.is/nzCKd
Make sure you understand what loss function logistic regression is using when you change scorers.
Read:
https://archive.is/nQ6Ha

Why use phik?
For this model, we could be using accuracy for parameter search optimization  like so:
myscorerNone = None #uses the default metric, accuracy
we could also have used the F1 metric like so:
myscorerNone = "f1"
but
we wanted to use a metric (phik) for which we can provide the statistical significance.

Also accuracy needs the target labels to be balanced, so
to use accuracy you need to set the logistic regressor's class_weight parameter to 'balanced' like so:
logistic = LogisticRegression(max_iter=1000, solver='liblinear', class_weight ='balanced') 

How would you find out if the target's classes are balanced?

Using class_weights increases the model's dependence on the training data:
it uses the training data to learn the coefficients AND
it uses the training data to adjust the class weights to balance the targets.
This means that a model that uses class weights needs to be re-trained often because
it will have a greater chance to overfit.

The phik correlation does not require the target labels to be balanced but
it does require a very large sample 
(>1000 samples per cross validation iteration) to bring down the sampling error.
Fortunately we have enough data to do this because
we are using intraday data.
By the way, the F1 score does not need target labels to be balanced either.
The problem with the F1 score is that it is very unintuitive to the layman, unlike
a correlation value or accuracy.
To understand the scoring for multi-class classifiers you need to read ModelEvaluation.pdf
ModelEvaluation.pdf explains the various scorings used to evaluate classifiers.
Read the entire pdf. 
But concentrate on accuracy and the F1 score.

Note:
the annualization factor used in the script 
is the default 252 (for 252 trading days per year).
the USDCAD data is approximately 8 time units per day (it varies)
so the correct annualization factor should be modified to 252*8. 
this correction does not alter the optimization results.
"""