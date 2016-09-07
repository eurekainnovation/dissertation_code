
# coding: utf-8

# In[22]:

import gmodel3
import makepredictions3
import datetime, time
import numpy as np
import pandas as pd
import os
from scipy.stats.kde import gaussian_kde
import scipy.stats as stats


# In[23]:

def create(company, reference, industry):

    ########################################
    num_tests = 2000 
    print (num_tests)
    
    date1 = datetime.date(2005, 1, 1)
    date2 = datetime.date(2015, 1, 1)

    for num_states in range(1,10):
        gmodel3.cmodel(company,reference, date1,date2,num_states)

    d1 = datetime.date(2015, 1, 1)
    d2 = datetime.date(2015, 7, 1)
    
    filename = []
    
    for s in range(10):
        fname = company+"_"+str(s+1)+"_states_model_adv.pkl"
        filename.append(os.path.join('./sims3', fname))

    results1 = np.array(makepredictions3.predictions_mls(filename[0], company, reference, d1, 
                                                       d2, 1, num_tests))

    results2 = np.array(makepredictions3.predictions_mls(filename[1], company, reference, d1, 
                                                        d2, 2, num_tests))

    results3 = np.array(makepredictions3.predictions_mls(filename[2], company, reference, d1, 
                                                       d2, 3, num_tests))

    results4 = np.array(makepredictions3.predictions_mls(filename[3], company, reference, d1, 
                                                       d2, 4, num_tests))
    
    results5 = np.array(makepredictions3.predictions_mls(filename[4], company, reference, d1, 
                                                       d2, 5, num_tests))
    
    results6 = np.array(makepredictions3.predictions_mls(filename[5], company, reference, d1, 
                                                       d2, 6, num_tests))
    
    results7 = np.array(makepredictions3.predictions_mls(filename[6], company, reference, d1, 
                                                       d2, 7, num_tests))

    results8 = np.array(makepredictions3.predictions_mls(filename[7], company, reference, d1, 
                                                       d2, 8, num_tests))
    
    results9 = np.array(makepredictions3.predictions_mls(filename[8], company, reference, d1, 
                                                       d2, 9, num_tests))
    
    pd.options.display.float_format = '{:,.6f}'.format
    
    days = [365,90,10]
    names = ['year','90days','10days']
    
    
    for i in range(0,3):
        
        real = makepredictions3.getrealprice(company, d2, days[i])
        
        avg_predict = np.array((np.average(results1[i]),np.average(results2[i]),np.average(results3[i]),
                               np.average(results4[i]),np.average(results5[i]),np.average(results6[i]),
                               np.average(results7[i]),np.average(results8[i]),np.average(results9[i])))

        df1_pdf = gaussian_kde(results1[i])
        df2_pdf = gaussian_kde(results2[i])
        df3_pdf = gaussian_kde(results3[i])
        df4_pdf = gaussian_kde(results4[i])
        df5_pdf = gaussian_kde(results5[i])
        df6_pdf = gaussian_kde(results6[i])
        df7_pdf = gaussian_kde(results7[i])
        df8_pdf = gaussian_kde(results8[i])
        df9_pdf = gaussian_kde(results9[i])

        kdes = np.array([df1_pdf(real)[0], df2_pdf(real)[0], df3_pdf(real)[0], 
                         df4_pdf(real)[0], df5_pdf(real)[0], df6_pdf(real)[0],
                         df7_pdf(real)[0], df8_pdf(real)[0], df9_pdf(real)[0]])

        rank = stats.rankdata(-1*kdes)

        dfkde = pd.DataFrame({ 
                            'A' : company,
                            'B' : pd.Categorical(["1 state", "2 states","3 states","4 states","5 states",
                                                  "6 states","7 states","8 states","9 states"]),
                            'C' : days[i],
                            'D' : kdes,
                            'E' : rank,
                            'F' : real,
                            'G' : avg_predict,
                            'H' : avg_predict - real,
                             })

        fname = str(company) +"_"+ str(industry) + "_" + names[i] + "_kdes_" + str(reference) + "_adv.csv"
        dfkde.to_csv(os.path.join('./sims3', fname))

        '''
        import matplotlib.pyplot as plt
        x = np.linspace(real/2.0,real*1.5,500)
        print ("Real price:",i," ",real)
        plt.plot(x,df1_pdf(x),'r', label='1-state')
        plt.plot(x,df3_pdf(x),'c', label='3-state') # distribution function
        plt.plot(x,df5_pdf(x),'y', label='5-state') # distribution function
        plt.plot(x,df7_pdf(x),'b', label='7-state') # distribution function
        plt.plot(x,df9_pdf(x),'k', label='9-state') # distribution function
        plt.plot((real, real), (0, df6_pdf(real)+.05), 'g--')
        #hist(samp,normed=1,alpha=.3) # histogram
        plt.legend(loc='best', shadow=True)
        plt.show()
        '''


# In[25]:

#create('WFC','XLF','technology')


# In[ ]:




# In[ ]:




# In[ ]:



