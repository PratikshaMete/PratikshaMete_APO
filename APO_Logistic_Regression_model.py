#!/usr/bin/env python
# coding: utf-8

# In[18]:


# import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle


# # Load the data

# In[2]:


#LOAD DATA
data_file = r"C:\Users\Pratiksha\Desktop\CreditCard_FraudDetector\creditcard.csv"
data=pd.read_csv(data_file)


# In[3]:


data[0:6]


# In[4]:


#find total observations in dataset
len(data.index)


# In[5]:


len(data.columns)


# # Define Predictor and Target 

# In[6]:


#FIT LOGISTIC REGRESSION MODEL
#Predictor
X = data.drop(["Class"], axis=1)

# Traget
y = data['Class']


# In[7]:


X.head()


# In[9]:


y.head()


# # Train Test split

# In[10]:


#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# # Instantiate the model 

# In[14]:


#instantiate the model
log_regression = LogisticRegression()

#fit the model using the training data
log_regression.fit(X_train,y_train)


# # Predict Fraud or Not

# In[15]:


#use model to make predictions on test data
y_pred = log_regression.predict(X_test)


# # Confusion matrix

# In[16]:


#MODEL DIAGNOSTICS
Confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
Confusion_matrix


# # Accuracy

# In[20]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Plot ROC Curve

# In[17]:


#plot ROC curve
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()


# # Save the model as a Pickel file

# In[21]:


#Save the model
pickle.dump(log_regression,open('regmodel.pkl','wb'))


# # Use the pickle file to Load the model to compare results
# 

# In[33]:


# Loading the model to compare results
model=pickle.load(open('regmodel.pkl','rb'))
#print(X_test)
print(X_test[0:1])
X_test1 = X_test[0:1]
#print(model.predict(X_test[0:99]))


# In[34]:


print(model.predict(X_test1))


# In[ ]:




