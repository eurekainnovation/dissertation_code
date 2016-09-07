
# coding: utf-8

# In[15]:

from __future__ import print_function

import generate_final

import random
import datetime, time
import numpy as np
import pandas as pd
import os
import multiprocessing as mp


# In[9]:

# Extract the list of companies for which predictions will be made
sp = np.genfromtxt(os.path.join('./sims_final', 'companies_final.csv'), delimiter=',', dtype=None)


# In[10]:

tickers = sp[:,0]
industries = sp[:,1]


# In[11]:


num_to_process = tickers.size 
ticker_sub = tickers[:num_to_process]
industry_sub = industries[:num_to_process]

companies_sub = np.vstack((ticker_sub,industry_sub))


# In[17]:

#industry_sub


# In[13]:

#companies_sub[0,:]


# In[14]:

'''
Note: Uncomment pool lines and comment generate_final.create
when using the multiprocessing module to speed-up.
'''
#pool = mp.Pool()

for t,i in zip(companies_sub[0,:],companies_sub[1,:]):
    print (t) 
    delta_st = random.randint(14,365) # random start delta jump, from 30 days to a year
    #try:
    generate_final.create(t, delta_st)
    #pool.apply_async(generate_final.create, args=(t, delta_st))
    #except:
    #    print ("ERROR!")
#pool.close()
#pool.join()


# In[ ]:



