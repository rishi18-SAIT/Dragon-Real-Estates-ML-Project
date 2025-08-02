#!/usr/bin/env python
# coding: utf-8

# In[6]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')


# In[11]:


features = np.array([[-5.43942006,4.12628155, -1.6165014, -0.67288841,
         -11.44443979304, -0.86091034, -0.44352175,  3.12628155, 
                      -1.35893781,-0.43525657, -0.4898311 , -1.23083158, 0.41070422]])
model.predict(features)


# In[ ]:




