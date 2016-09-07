
# coding: utf-8

# In[1]:

from __future__ import print_function
import generatekdes3
import datetime, time
import numpy as np
import pandas as pd

import os
import multiprocessing as mp


# In[2]:

sp = np.genfromtxt(os.path.join('./sims3', 'companies_final.csv'), delimiter=',', dtype=None)


# In[3]:

tickers = sp[:,0]
industries = sp[:,1]


# In[4]:

reference = 'IEF'
num_to_process = tickers.size # change if only processing sub-group for the moment
ticker_sub = tickers[:num_to_process]
industry_sub = industries[:num_to_process]

companies_sub = np.vstack((ticker_sub,industry_sub))


# In[5]:

#err = np.genfromtxt(os.path.join('./sims3', 'error_companies.csv'), delimiter=',', dtype=None)


# In[10]:

#industry_sub


# In[7]:

companies_sub[0,:]


# In[ ]:




# In[8]:

#pool = mp.Pool()

for t,i in zip(companies_sub[0,:],companies_sub[1,:]):
    
    if (not (i == 'Financials')):
        continue
    print (t)    
    generatekdes3.create(t, reference, i)
    #pool.apply_async(generatekdes2.create, args=(t, i))

#pool.close()
#pool.join()

