
# coding: utf-8

# In[51]:

from __future__ import print_function
import datetime, time
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import random
import os
import warnings
from time import gmtime, strftime
from matplotlib.dates import YearLocator, MonthLocator

try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # Matplotlib prior to 1.5.
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )

from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore') # Get rid of some annoying divide by zero in log warnings
max_int_value = 4294967295
yr =  364
total2active = 7/5.0 # Ratio of days the market is open to all days


# In[52]:

def predictions_mls(filename, company, refcompany, dt1, dt2,num_of_states,test_num):
# Generate samples starting in the most likely actual current state
    
    days_future = 365
    
    model = joblib.load(filename) 
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) 
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[5] for q in quotes])[1:]

    # Take diff of close value. Note that this makes
    # len(diff) = len(close_t) - 1 therefore, other quantities also need to be shifted by 1
    
    diff = np.diff(close_v)
    dates = dates[1:]
    close_v = close_v[1:]
    
    # Unpack quotes Company2
    quotes2 = quotes_historical_yahoo_ochl(refcompany, dt1, dt2)
    close_v2 = np.array([q[2] for q in quotes2])
    diff2 = np.diff(close_v2)
    close_v2 = close_v2[1:]
    
    #print (diff2.shape)

    delta = diff2.shape[0]-diff.shape[0]
    delta = abs(delta)
    
    diff0=np.pad(diff, (delta,0), mode='constant', constant_values=0)
    close_v=np.pad(close_v, (delta,0), mode='constant', constant_values=0)
    
    #print (diff.shape)
    #print (diff0.shape)
    
    X = np.column_stack([diff0,diff2])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 

    days = int(days_future//total2active) # 251 open market days in a year
    print (days, strftime("%Y-%m-%d %H:%M:%S", gmtime())) #debugging purposes

    if (num_of_states>1):
        startprob = np.zeros(num_of_states)
        startprob[lstate_prob.argmax()] = 1.0
    else:
        startprob = [ 1.]

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,max_int_value)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0
    
     
    allpredictions = np.zeros((test_num, yr))
    for test in range(test_num):  
        final_price = close_v[-1]
        j = 0
        for i in range(days):
            if ((final_price+X[i][0]) > 0 ):
                final_price += X[i][0]
            if (j>1 and i%5==0):
                allpredictions[test][j] = final_price
                allpredictions[test][j+1] = final_price
                allpredictions[test][j+2] = final_price
                j = j+3
            else:
                allpredictions[test][j] = final_price
                j = j + 1
                
        while (j<allpredictions.shape[1]):
            allpredictions[test][j] = final_price
            j = j+1
            
        rseed = random.randrange(0,max_int_value)
        X, Z = model_2_sample.sample(days, random_state=rseed)
        
    predictions_year = allpredictions.mean(axis=0)
    print ("Avg. Prediction: ",predictions_year[-1])
    
    fname = "Year_of_predictions_"+str(company)+"_States_"+str(num_of_states)+"_adv.csv"
    fname = os.path.join('./sims3', fname)
    np.savetxt(fname, predictions_year, delimiter=",")

    return allpredictions[:,days_future-2],allpredictions[:,(days_future-2)/4],allpredictions[:,(days_future-2)/36]



# In[53]:

def getrealprice(company, dt2, days_future):
    
    dt3 = dt2 + datetime.timedelta(days=days_future)
    quotes = quotes_historical_yahoo_ochl(company, dt2,dt3) 
    close_v = np.array([q[2] for q in quotes])
    
    return close_v[-1]


# In[54]:

'''
import os
import gmodel3

d1 = datetime.date(2015, 1, 1)
d2 = datetime.date(2015, 7, 1)

company = 'PAAS'
refcompany = 'SPY'

num_tests = 2000
num_states = 5

date1 = datetime.date(2007, 1, 1)
date2 = datetime.date(2015, 1, 1)
gmodel3.cmodel(company,refcompany,date1,date2,num_states)

filename = company+"_"+str(num_states)+"_states_model_adv.pkl"
filename = os.path.join('./sims3', filename)
rslt = predictions_mls(filename, company, refcompany, d1, d2, num_states, num_tests)
'''


# In[55]:

'''
print ("Starting price: ",getrealprice(company, d2, 0))
print ("Prediction 10 days out ",np.average(rslt[2]))
print ("1 qtr out: ",np.average(rslt[1]))
print ("1 year out:",np.average(rslt[0]))
print ("Real ending price: ",getrealprice(company, d2, 365))
'''


# In[ ]:



