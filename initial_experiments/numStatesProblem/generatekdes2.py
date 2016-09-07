
# coding: utf-8

# In[68]:

import gmodel
import makepredictions
import datetime, time
import numpy as np
import pandas as pd
import os
from scipy.stats.kde import gaussian_kde
import scipy.stats as stats


# In[69]:

def create(company, num_states, days_into_future):

    ########################################
    num_tests = 20000 #20000 seems to be ideal compromise between quality and time to compute
    print (num_tests)
    ########################################

    #now = datetime.datetime.now()
    #datetime.date(now.year, now.month, now.day)
    date1 = datetime.date(2005, 1, 1)
    date2 = datetime.date(2015, 1, 1)

    gmodel.cmodel(company,date1,date2,num_states)

    d1 = datetime.date(2015, 1, 1)
    d2 = datetime.date(2015, 7, 1)

    #s = pd.Series([1,3,5,np.nan,6,8])
    real = makepredictions.getrealprice(company, d2, days_into_future)
    
    filename = company+"_"+str(num_states)+"_states_model.pkl"
    filename = os.path.join('./sims1', filename)

    results1 = np.array(makepredictions.predictions_mls(filename, company, d1, 
                                                       d2, num_states, num_tests,days_into_future))


    results2 = np.array(makepredictions.predictions_rand(filename, company, d1, 
                                                        d2, num_states, num_tests,days_into_future))


    results3 = np.array(makepredictions.predictions_vtr(filename, company, d1, 
                                                       d2, num_states, num_tests,days_into_future))


    results4 = np.array(makepredictions.predictions_psd(filename, company, d1, 
                                                       d2, num_states, num_tests,days_into_future))
    
    results5 = np.array(makepredictions.predictions_lls(filename, company, d1, 
                                                       d2, num_states, num_tests,days_into_future))
    
    results6 = np.array(makepredictions.predictions_mlns(filename, company, d1, 
                                                       d2, num_states, num_tests,days_into_future))


    pd.options.display.float_format = '{:,.6f}'.format

    df1 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results1.size*["MLS"]),
                        'C' : days_into_future,
                        'D' : results1,
                        'E' : real,
                        'F' : results1 - real
                         })

    df2 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results2.size*["RAND"]),
                        'C' : days_into_future,
                        'D' : results2,
                        'E' : real,
                        'F' : results2 - real
                         })

    df3 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results3.size*["VTR"]),
                        'C' : days_into_future,
                        'D' : results3,
                        'E' : real,
                        'F' : results3 - real
                         })

    df4 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results4.size*["PSD"]),
                        'C' : days_into_future,
                        'D' : results4,
                        'E' : real,
                        'F' : results4 - real
                         })
    
    df5 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results5.size*["LLS"]),
                        'C' : days_into_future,
                        'D' : results5,
                        'E' : real,
                        'F' : results5 - real
                         })

    df6 = pd.DataFrame({ 
                        'A' : num_states,
                        'B' : pd.Categorical(results6.size*["MLNS"]),
                        'C' : days_into_future,
                        'D' : results6,
                        'E' : real,
                        'F' : results6 - real
                         })

    samp1 = df1['D'].values
    samp2 = df2['D'].values
    samp3 = df3['D'].values
    samp4 = df4['D'].values
    samp5 = df5['D'].values
    samp6 = df6['D'].values

    df1_pdf = gaussian_kde(samp1)
    df2_pdf = gaussian_kde(samp2)
    df3_pdf = gaussian_kde(samp3)
    df4_pdf = gaussian_kde(samp4)
    df5_pdf = gaussian_kde(samp5)
    df6_pdf = gaussian_kde(samp6)

    x = np.linspace(np.amin(df1['D'].values),np.amax(df1['D'].values)*.77,200)

    kdes = np.array([df1_pdf(real)[0], df2_pdf(real)[0], df3_pdf(real)[0], 
                     df4_pdf(real)[0], df5_pdf(real)[0], df6_pdf(real)[0]])
    
    rank = stats.rankdata(-1*kdes)
    


    dfkde = pd.DataFrame({ 
                        'A' : company,
                        'B' : num_states,
                        'C' : pd.Categorical(["MLS", "Rand","Vtr","Psd","Lls","Mlns"]),
                        'D' : days_into_future,
                        'E' : kdes,
                        'F' : rank,
                         })

    fname = company +"_"+ str(num_states) + "_" + str(days_into_future) + "_kdes.csv"
    dfkde.to_csv(os.path.join('./sims1', fname))
    #pd.read_csv('foo.csv')
    #dfkde.to_excel(fname + '.xlsx', sheet_name='Sheet1')

