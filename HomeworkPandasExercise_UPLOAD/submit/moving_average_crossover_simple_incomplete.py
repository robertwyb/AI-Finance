"""
Note about the trading programs in this course
The trading programs we show are very idealized.
For example:
We do not consider limits in leverage or initial investment and 
this is idealistic since in reality there are limits to both as our pockets are not infinitely deep.
We do not consider trading costs etc. Again this is idealistic.
We assume it is possible to invest in an infinitesimally small fraction of a stock. Again this is idealistic. 
Later on, we will address how to limit some of these assumptions.
Because of all these assumptions impact profits and profits are not linear, 
our trading programs cannot be used to calculate profits realistically, rather
they are used to give an idea of profit to risk ratio, accuracy etc
"""
import WhiteRealityCheckFor1 #you can ignore this for now
import detrendPrice #you can ignore this fornow
import pandas as pd
import numpy as np
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

start_date = '2000-01-01' 
end_date = '2018-12-31' 
#end_date = datetime.now() 

symbol = '^GSPC' 
msg = "" 
address = symbol + '.csv'

try:
    dfP = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    dfP.to_csv(address, header = True, index=True, encoding='utf-8') 
except Exception:
    msg = "yahoo problem"
    dfP = pd.DataFrame()

dfP = pd.read_csv(address, parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)


dfP['42d'] = np.round(dfP['Close'].rolling(window=42).mean(),2)
dfP['252d'] = np.round(dfP['Close'].rolling(window=252).mean(),2)
#print(dfP.tail)

#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))
dfP['42-252'] = dfP['42d'] - dfP['252d']


dfP['pct_rets'] = dfP['Close'].pct_change()#INSERT CODE HERE USING CLOSE PRICES#

X = 0


dfP['Stance'] = np.where((dfP['42-252'] > X), 1, 0)
dfP['Stance'] = np.where((dfP['42-252'] < X), -1, dfP['Stance'])#INSERT CODE HERE#


dfP['syst_rets'] = dfP['pct_rets'] * dfP['Stance'].shift(1) #INSERT CODE HERE FOR SYSTEM RETURNS#
dfP['syst_cum_rets'] = np.cumprod(1+dfP['syst_rets']) #INSERT CODE HERE#
dfP['mkt_cum_rets'] =  np.cumprod(1+dfP['pct_rets'])#INSERT CODE HERE#

dfP[['mkt_cum_rets','syst_cum_rets']].plot(grid=True,figsize=(8,5)) #plotting returns percent cumul

start = 1 # or should start at dfP['sys_cum_rets.iloc].iloc[2]

start_val = start
end_val = dfP['syst_cum_rets'].iat[-1]

start_date = dfP.iloc[0].name
end_date = dfP.iloc[-1].name
days = (end_date - start_date).days 

periods = 360 #360 accounting days

TotalAnnReturn = (end_val-start_val)/start_val/(days/periods)

years = days/periods
CAGR = ((((end_val/start_val)**(1/years)))-1)


try:
    sharpe = (dfP['syst_rets'].mean()/dfP['syst_rets'].std()) * np.sqrt(periods)  #INSERT CODE HERE FOR THE SHARPE PERFORMANCE METRIC#
except ZeroDivisionError:
    sharpe = 0.0

print ("TotalAnnReturn in percent = %f" %(TotalAnnReturn*100))
print ("CAGR in percent = %f" %(CAGR*100))
print ("Sharpe Ratio = %f" %(round(sharpe,2)))

"""
#white reality check
#Detrend prices before calculating detrended returns
dfP['DetClose'] = detrendPrice.detrendPrice(dfP.Close).values #you can ignore this for now
#these are the detrended returns to be fed to White's Reality Check
dfP['Det_pct_rets']= (dfP['DetClose']- dfP['DetClose'].shift(1)) / dfP['DetClose'].shift(1) #you can ignore this for now
dfP['Det_syst_rets']= dfP['Det_pct_rets']*dfP['Stance'].shift(1) #you can ignore this for now
WhiteRealityCheckFor1.bootstrap(dfP.Det_syst_rets #you can ignore this for now

"""
dfP.to_csv(r'Results\dfP_simple_MACO.csv')
#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))



