
# coding: utf-8

# In[68]:

import gmodel_final
import makepredictions_final

import datetime, time
import numpy as np
import pandas as pd
import os

from math import ceil



# In[69]:

def create(company, delta_start):
    
    # delta_start is a random number of days to shift when we finish training the model and start predicting.

    ########################################
    num_tests = 2000 
    print (" Company: ", company, "Delta start days: ", delta_start)
    ########################################

    #Training time period
    date1 = datetime.date(2000, 1, 1)
    date2 = datetime.date(2010, 1, 1) + datetime.timedelta(days=delta_start)

    # Note, for the one state model the expected remain time is inf, therefore it will be stored in tmp and ignored
    tmp1, tmp2 = gmodel_final.cmodel(company,date1,date2,1) # Generate 1-state model
    d, tr_prob = gmodel_final.cmodel(company,date1,date2,5) # Generate 5-states model

    # Note: days_into_future is used to define the number of days into the future we are going to predict. 
    days_into_future = 364
    
    # Prepare the model file name for 1 and 5 states, to load it in the function that makes predictions
    filename1 = company+"_1_states_model_final.pkl"
    filename1 = os.path.join('./sims_final', filename1)
    filename5 = company+"_5_states_model_final.pkl"
    filename5 = os.path.join('./sims_final', filename5)    

    # We are going to make market predictions for 1 & 5 states models, using MLS starting strategy
    # The results will be stored by the makepredictions function in a CSV file   
    
    
    np.array(makepredictions_final.predictions_mls(filename1, company, date1, 
                                                       date2, 1, num_tests, days_into_future, tr_prob))
    
    np.array(makepredictions_final.predictions_mls(filename5, company, date1, 
                                                       date2, 5, num_tests, days_into_future, tr_prob))

