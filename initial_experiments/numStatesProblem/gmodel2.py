
# coding: utf-8

# In[17]:

from __future__ import print_function
import datetime, time
from sklearn.externals import joblib
import numpy as np
import os
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


# In[18]:

def cmodel(company, dt1, dt2, num_of_states=5):
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) #Here we set the time range

    # Unpack quotes
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[5] for q in quotes])[1:]

    # Take diff of close value. Note that this makes
    # len(diff) = len(close_t) - 1 therefore, other quantities also need to be shifted by 1
    
    diff = np.diff(close_v)
    dates = dates[1:]
    close_v = close_v[1:]
    
    #print (diff)
    # Pack diff and volume for training.
    X = np.column_stack([diff])

    # Create HMM instance and fit
    model = GaussianHMM(n_components=num_of_states, covariance_type="full", n_iter=1000).fit(X)
    fname = str(company)+"_"+str(num_of_states)+"_states_model.pkl"
    joblib.dump(model, os.path.join('./sims1', fname))

    #joblib.load("filename.pkl")  


# In[19]:

'''
# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
hidden_probs = model.predict_proba(X)

print ("\nMost likely state of last observation:",hidden_states[-1])
#print ("# of observations",hidden_states.size)
print ("\nProb. distr. of states for last observation: \n")
#print (hidden_probs.size,hidden_states.size)
print (np.array_str(hidden_probs[-1], precision=3, suppress_small=True))
#print ("# of hidden states",model.n_components)
'''


# In[ ]:



