
# coding: utf-8

# In[35]:

from __future__ import print_function
import datetime, time
from sklearn.externals import joblib
import numpy as np
import random
import sys
import warnings
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


# In[36]:

def predictions_mls(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in the most likely actual current state
    
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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    startprob = np.zeros(num_of_states)
    startprob[lstate_prob.argmax()] = 1.0

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions



# In[ ]:

def predictions_mlns(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in the most likely actual current state
    
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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_states = model.predict(X)
    lstate = hidden_states[-1] 

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    startprob = np.zeros(num_of_states)
    startprob[model.transmat_[lstate].argmax()] = 1.0

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions



# In[ ]:

def predictions_lls(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in the most likely actual current state
    
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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    startprob = np.zeros(num_of_states)
    startprob[lstate_prob.argmin()] = 1.0

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions


# In[37]:

def predictions_rand(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in a random state 

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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    startprob = np.zeros(num_of_states)
    for start_st_prob in range(num_of_states):
        startprob[start_st_prob] = 1.0/num_of_states

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions


# In[38]:

def predictions_vtr(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in a random state 

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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_states = model.predict(X)

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    startprob = np.zeros(num_of_states)
    startprob[hidden_states[-1]] = 1.0

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions


# In[39]:

def getrealprice(company, dt2, days_future):
    
    dt3 = dt2 + datetime.timedelta(days=days_future)
    quotes = quotes_historical_yahoo_ochl(company, dt2,dt3) 
    close_v = np.array([q[2] for q in quotes])
    
    return close_v[-1]


    


# In[ ]:

def predictions_psd(filename, company, dt1, dt2,num_of_states,test_num, days_future):
# Generate samples starting in the most likely actual current state
    
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
    
    X = np.column_stack([diff])

    # Predict the most likely current internal hidden state
    hidden_probs = model.predict_proba(X)
    lstate_prob = hidden_probs[-1] 

    total2active = 364/251 # Ratio of days the market is open to all days
    days = days_future//total2active # 251 open market days in a year
    predictions = [] # Might be useful to store the predictions for future use
    print (days)

    #startprob = np.zeros(num_of_states)
    startprob = lstate_prob

    model_2_sample = GaussianHMM(n_components=num_of_states, covariance_type="full")
    model_2_sample.startprob_ = startprob
    model_2_sample.transmat_ = model.transmat_
    model_2_sample.means_ = model.means_
    model_2_sample.covars_ = model.covars_

    random.seed()
    rseed = random.randrange(0,sys.maxint)
    X, Z = model_2_sample.sample(days, random_state=rseed)
    avg_prediction = 0

    for test in range(test_num):  
        final_price = close_v[-1]
        for i in range(days):
            if ((final_price+X[i]) > 0 ):
                final_price += X[i]

        predictions.append(final_price[0])
        rseed = random.randrange(0,sys.maxint)
        X, Z = model_2_sample.sample(days, random_state=rseed)

    return predictions


# In[40]:

'''
investment = np.zeros(close_v_p.size)
investment[0] = close_v_p[0]
print ("Starting price: ", investment[0])

Xseq = X[:]

Xseq = np.append(Xseq,Xp[:], axis=0)

print ("States to avoid:", state_to_avoid)


for p in range(close_v_p.size-1):
    ptime = X.size + p
    #hidden_states_p  = model.predict(Xseq[:ptime])
    hidden_probs_p = model.predict_proba(Xseq[:ptime])
    lstate_prob = hidden_probs_p[-1]
    if ((lstate_prob[state_to_avoid] < tmpresult[0])or(lstate_prob[state_to_avoid] > tmpresult[1])):
        investment[p+1] = (investment[p] + diff_p[p])
    else:
        investment[p+1] = (investment[p])


print ("Last price:",close_v_p[-1])
print ("Investment:",investment[-1])
print ("Last state probabilities:",lstate_prob)
'''


# In[ ]:




# In[ ]:



