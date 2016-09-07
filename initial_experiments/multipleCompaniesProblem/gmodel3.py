
# coding: utf-8

# In[1]:

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


# In[1]:

def cmodel(company, refcompany, dt1, dt2, num_of_states):
    
    quotes = quotes_historical_yahoo_ochl(company, dt1, dt2) #Here we set the time range
    quotes2 = quotes_historical_yahoo_ochl(refcompany, dt1, dt2) #Here we set the time range

    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[5] for q in quotes])[1:]

    # Take diff of close value. Note that this makes
    # len(diff) = len(close_t) - 1 therefore, other quantities also need to be shifted by 1
    
    diff = np.diff(close_v)
    dates = dates[1:]
    close_v = close_v[1:]
    
    # Unpack quotes Company2
    close_v2 = np.array([q[2] for q in quotes2])
    diff2 = np.diff(close_v2)
    close_v2 = close_v2[1:]
    
    delta = diff2.shape[0]-diff.shape[0]
    delta = abs(delta)
    
    diff0=np.pad(diff, (delta,0), mode='constant', constant_values=0)
    close_v=np.pad(close_v, (delta,0), mode='constant', constant_values=0)
       
    X = np.column_stack([diff0,diff2])

    # Create HMM instance and fit
    model = GaussianHMM(n_components=num_of_states, covariance_type="full", n_iter=1000).fit(X)
    fname = str(company)+"_"+str(num_of_states)+"_states_model_adv.pkl"
    joblib.dump(model, os.path.join('./sims3', fname))


# In[ ]:



