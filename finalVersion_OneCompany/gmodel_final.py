
# coding: utf-8

# In[2]:

from __future__ import print_function
import datetime, time
from sklearn.externals import joblib
import numpy as np
import os
import warnings
import quandl

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


# In[3]:

def cmodel(company, dt1, dt2, num_of_states):
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) #Here we set the time range

    # Unpack the quotes !
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])

    # Take diff of close value and shift by 1
    
    diff = np.diff(close_v)  

    
    dates = dates[1:]
    close_v = close_v[1:]
    
    X = np.column_stack([diff])    
    
    # Create HMM instance and fit 
    model = GaussianHMM(n_components=num_of_states, covariance_type="full", n_iter=1000).fit(X)
    #print ("Model Covars: ", model.covars_)

    expected_days = 1
    tr_mls = 1
    
    if (num_of_states > 1):
        #Identify the most likely last hidden state
        
        try:
            hidden_probs = model.predict_proba(X)
        except:
            model = GaussianHMM(n_components=num_of_states, covariance_type="diag", n_iter=1000).fit(X)
            hidden_probs = model.predict_proba(X)
            
        lstate_prob = hidden_probs[-1] 
        mls = lstate_prob.argmax()

        # self transition probability for the most likely last hidden state
        tr_mls = model.transmat_[mls][mls]

        # we make use of the geometric series formula to calculate the number
        # of days expected to stay at the current state
        expected_days = (1.0 / (1 - tr_mls))
    
    # we save the model for future use
    fname = str(company)+"_"+str(num_of_states)+"_states_model_final.pkl"
    joblib.dump(model, os.path.join('./sims_final', fname)) 
    
    #return expected days
    return expected_days, tr_mls


# In[ ]:



