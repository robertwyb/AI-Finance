"""
IMPORTANT OBSERVATION ABOUT PHIK AND SPEARMAN CORRELATION:
These two metrics require a lot of data to be significant
It makes sense to use these two metrics 
if you are using high frequency (intraday) data (as is the case in this homework) or
you are predicting many stocks at once as with Alphalens (to be covered)
in either case you have a lot of data.
However, if you plan to predict the price of a single stock using daily weekly or monthly data
it makes better sense to use a different metric like
"mean absolute error" (for regression) or 
"F1 score or accuracy" (for classification) that
allows for a shorter lookback.
A shorter lookback is advantageous because
financial price information may be no longer relevant 
after some time has passed.
But if the lookback is short, then
to get a better sense of the performance of the model,
you need to apply the model repeatedly within a walk forward harness that
uses either a rolling fixed window or an expanding window (either is acceptable).
In general, a walk forward harnesss is a loop where 
the model predicts n items and moves forward n items (n can be 1 or any integer but zero).
In this script we show 
the use of a fixed window (rolling) walk forward harness 
In this setup, the model predicts step=1 and moves forward step=1
using a fixed size rolling window.

"""
"""
INSTRUCTIONS
GOTO WALK_FORWARD HARNESS and
optimize the length of the lookback_window, and step 
while keeping an eye on accuracy_scor.
After you find an optimal lookback_window and step
you can change line 158 to RandomizedSearchCV to speed up the program.
This exercise is just to convince you that
the size of the lookback has an important impact.
Turn in this program with the optimal settings "on".


"""
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys
 
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



#build window features, assuming we know today's open
for n in list(range(1,21)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n).fillna(0)#for trading with open
    #df[name] = df["<CLOSE>"].pct_change(periods=n).fillna(0) #for trading with close

#buld the best window features after the exploratory data analysis:
#for n in [1,2,3,4,11,14]:
#    name = 'ret' + str(n)
#    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
#    #df[name] = df["<CLOSE>"].pct_change(periods=n) #for trading with close

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)


#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open
#df['retFut1'] = df['<CLOSE>'].pct_change(1).shift(-1).fillna(0) #if you wait until the close to enter the trade
#df = np.log(df+1)

#transform the target
df['retFut1_categ'] = np.where((df['retFut1'] > 0), 1, 0)

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
X = df_filtered.drop(['retFut1', 'retFut1_categ',"hour","day"], axis=1) #these drops are note optional
y = df_filtered[['retFut1_categ']]

#select the samples
x_train = X.iloc[0:20000]
#x_test = X.iloc[20000:22000]

y_train = y.iloc[0:20000]
#y_test = y.iloc[20000:22000]

df_train = df_filtered.iloc[0:20000]
#df_test = df_filtered.iloc[20000:22000]

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit


def model_do_all(x_train_local, y_train_local, x_test_local):
    myscorer = None #use default accuracy score
    class_weight = "balanced" #if using accuracy
    #3myscorer="f1"
    #class_weight = None #if using f1 
    

    #split = TimeSeriesSplit(n_splits=5, max_train_size=1000) #fixed sized window
    split = TimeSeriesSplit(n_splits=5) #increasing size window
    
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    #we turn off scaling because we are using dummies (returns are already mostly scaled)
    scaler = StandardScaler(with_mean=False, with_std=False)
    logistic = LogisticRegression(max_iter=1000, solver='liblinear', class_weight=class_weight) 
    
    pipe = Pipeline([("imputer", imputer),("scaler", scaler), ("logistic", logistic)])
    
    c_rs = np.logspace(3, -4, num=20, endpoint = True)
    #penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)
    p_rs= ["l1", "l2"]
    
    param_grid =  [{'logistic__C': c_rs, 'logistic__penalty': p_rs}]
    
    #grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
    grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
    
    grid_search.fit(x_train_local.values, y_train_local.values.ravel())
    # Test set
    # Make "predictions" on test set (out-of-sample)

    predictions = grid_search.predict(x_test_local.values)
    predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=x_test_local.index).fillna(0)
    return predictions_df

#WALK FORWARD HARNESS
n = 10000/1 #(try various denominators from 2 to 5...)
lookback_window = round(n) #the length of the lookback window 
back_test_window =   round(n+70) #the length of the "backtest" (how many times the model moves forward)
step = 8 #try various steps 1, 4, 8  measures how often the model re-trains, e.g. every day, every 4 days etc.
# lists for storing
predictions_list = []
retFut1_df = pd.DataFrame(df_train.retFut1, index=df_train.index)

print(list(range(lookback_window, min(back_test_window, len(x_train)), step)))

#use a rolling forward fixed size window to fit and optimize the model and predict
for ix in range(lookback_window, min(back_test_window, len(x_train)), step):
    predictions_list.append(model_do_all(x_train_local = x_train.iloc[ix - lookback_window:ix, :], y_train_local = y_train.iloc[ix - lookback_window:ix, :], x_test_local = x_train.iloc[ix:ix+step, :]))

for x in list(range(1,len(predictions_list)+1)):
    if x < 2:
        predictions_df = pd.concat([predictions_list[1],predictions_list[2]], axis=0)
    else:
        if(x+1<len(predictions_list)):
            predictions_df = pd.concat([predictions_df,predictions_list[x+1]], axis=0)
            
            
df2 = predictions_df.join(y_train)
df2 = df2.join(retFut1_df)
df2["positions"] = np.where(df2.predictions> 0,1,-1 ) #financial POSITIONS
df2["dailyRet"] = df2.positions * df2.retFut1
df2["cumret"] = np.cumprod(df2.dailyRet+1) - 1


plt.figure(1)
plt.plot(df2.index, df2.cumret)
title = 'Cross-validated LogisticRegression on currency: test set'
plt.title(title)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.xticks(rotation=70)
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative"))

#metrics
accuracy_scor = accuracy_score(df2.retFut1_categ.values, df2.predictions.values)

#8 units per day approx so 252 trading days in a year * 8 units per day
cagr = (1 + df2.cumret.values[-1]) ** (8*252 / len(df2.cumret.values)) - 1 
maxDD, maxDDD = fAux.calculateMaxDD(df2.cumret.values)
ratio = (8*252.0 ** (1.0/2.0)) * np.mean(df2.dailyRet.values) / np.std(df2.dailyRet.values)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, accuracy_scor))

# /2-54, /3-53, /4-60, /5-45 (step=1)
