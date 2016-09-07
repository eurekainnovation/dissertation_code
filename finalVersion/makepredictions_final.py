
# coding: utf-8

# In[1]:

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
from math import ceil

try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )

from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore') # Get rid of some annoying divide by zero in log warnings
max_int_value = 4294967295
total2active = 7/5.0 # Ratio of days the market is open to all days


# In[2]:

def predictions_mls(filename, company, dt1, dt2,num_of_states,test_num, days_future, tr_prob):
# Generate samples starting in the most likely actual current state
       
    model = joblib.load(filename) 
    
    rp = getrealprice_series(company, dt2,days_future)
    days = rp.size
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) 
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])


    # Take diff of close value and shift by 1    
    diff = np.diff(close_v)

    dates = dates[1:]
    close_v = close_v[1:]    
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 
    

    
    # If more than one state, make sure we start at the most likely current state
    if (num_of_states>1):
        startprob = np.zeros(num_of_states)
        startprob[lstate_prob.argmax()] = 1.0
    else:
        startprob = [ 1.]

    # Prepare the model for sampling
    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    #Make sure to randomize the samples
    random.seed()
    rseed = random.randrange(0,max_int_value)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    
    # Make predictions
    avg_prediction = 0 
    allpredictions = np.zeros((test_num, days)) #added two in case there was a weekend at the end
    
    for test in range(test_num): 
        
        final_price = rp[0] #start at day 0 of the real prices
        allpredictions[test][0] = final_price   #day 0 prediction same as current real price
        
        for i in range(1, days):
            final_price += X[i][0]

            allpredictions[test][i] = final_price
            
        rseed = random.randrange(0,max_int_value)
        X, Z = model_2_sample.sample(days, random_state=rseed)



    predictions = allpredictions.mean(axis=0)
    predictions_var = allpredictions.var(axis=0)
    predictions_median =  np.median(allpredictions, axis=0)    

    
    errors = predictions - rp 
    tr_prob_vector = np.full((predictions.size),tr_prob)
    
    data = [predictions,rp, errors, tr_prob_vector, 
            predictions_var,predictions_median]

    err_final = errors[-1]
    
    print ("Start Price: ",rp[0],"Avg. Prediction: ",str(num_of_states),"states:" ,
           predictions[-1]," Real Price:", rp[-1]) 
    print (" Error end of predictions:", err_final,"Delta Start-End:", rp[0]-rp[-1],"\n")
    #print ("Real prices:", rp)
    #print ("Predicted prices", predictions)
    
    fname = "Predictions_"+str(company)+"_States_"+str(num_of_states)+"_stats.csv"
    fname = os.path.join('./sims_final', fname)
    np.savetxt(fname, data, delimiter=",")

    
    return



# In[3]:

def getrealprice_series(company, dt2, days_future):
    
    dt3 = dt2 + datetime.timedelta(days=days_future)
    quotes = quotes_historical_yahoo_ochl(company, dt2,dt3) 
    close_v = np.array([q[2] for q in quotes])
    
    return close_v


# In[4]:

def getrealprice(company, dt2, days_future):
    
    dt3 = dt2 + datetime.timedelta(days=days_future)
    quotes = quotes_historical_yahoo_ochl(company, dt2,dt3) 
    close_v = np.array([q[2] for q in quotes])
    
    return close_v[-1]


# In[ ]:

def predict_one(filename, company, dt1, dt2,num_of_states, days_future, tr_prob):
# Generate samples starting in the most likely actual current state
       
    model = joblib.load(filename) 
    
    rp = getrealprice_series(company, dt2,days_future)
    days = rp.size
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) 
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])

    # Take diff of close value and shift by 1    
    diff = np.diff(close_v)

    dates = dates[1:]
    close_v = close_v[1:]    
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 
    

    
    # If more than one state, make sure we start at the most likely current state
    if (num_of_states>1):
        startprob = np.zeros(num_of_states)
        startprob[lstate_prob.argmax()] = 1.0
    else:
        startprob = [ 1.]

    # Prepare the model for sampling
    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    #Make sure to randomize the samples
    random.seed()
    rseed = random.randrange(0,max_int_value)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    
    # Make predictions
    predictions = np.zeros(days) #added two in case there was a weekend at the end
    
        
    final_price = rp[0] #start at day 0 of the real prices
    predictions[0] = final_price   #day 0 prediction same as current real price

    for i in range(1, days):
        final_price += X[i][0]
        predictions[i] = final_price
            
    return predictions

