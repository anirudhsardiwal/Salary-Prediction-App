#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# 
# WHO is a specialized agency of the UN which is concerned with the world population health. Based upon the various parameters, WHO allocates budget for various areas to conduct various campaigns/initiatives to improve healthcare. Annual salary is an important variable which is considered to decide budget to be allocated for an area.  
#   
# We have a data which contains information about 32561 samples and 15 continuous and categorical variables. Extraction of data was done from 1994 Census dataset.  
#   
# The goal here is to build a binary model to predict whether the salary is >50K or <50K.

# ## Data Dictionary

# 1. <b>age:</b> age  
# 2. <b>workclass:</b> workclass  
# 3. <b>fnlwgt:</b> samplting weight  
# 4. <b>education:</b> highest education  
# 5. <b>education-no. of years:</b> number of years of education in total  
# 6. <b>marrital status:</b> marrital status  
# 7. <b>occupation:</b> occupation  
# 8. <b>relationship:</b> relationship  
# 9. <b>race:</b> race  
# 10. <b>sex:</b> sex  
# 11. <b>capital gain:</b> income from investment sources other than salary/wages  
# 12. <b>capital loss:</b> income from investment sources other than salary/wages  
# 13. <b>working hours:</b> nummber of working hours per week  
# 14. <b>native-country:</b> native country  
# 15. <b>salary:</b> salary  
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix


# In[2]:


adult_data=pd.read_csv("adult.data.csv")


# ### EDA

# In[3]:


adult_data.head()


# In[4]:


adult_data.info()


# There are no missing values. 6 variables are numeric and remaining categorical. Categorical variables are not in encoded format

# ### Check for duplicate data

# In[5]:


dups = adult_data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print(adult_data.shape)


# There are 24 duplicates that needs to be removed

# In[6]:


adult_data.drop_duplicates(inplace=True) 


# In[7]:


dups = adult_data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print(adult_data.shape)


# ### Geting unique counts of all Objects

# In[8]:


adult_data.salary.value_counts(normalize=True)


# In[9]:


for feature in adult_data.columns: 
    if adult_data[feature].dtype == 'object': 
        print(feature)
        print(adult_data[feature].value_counts())
        print('\n')
        


# workclass, occupation,native-country has ?  
# Since, high number of cases have ?, we will convert them into a new level

# In[10]:


# Replace ? to new Unk category
adult_data.workclass=adult_data.workclass.str.replace('?', 'Unk')
adult_data.occupation = adult_data.occupation.str.replace('?', 'Unk')
adult_data['native-country'] = adult_data['native-country'].str.replace('?', 'Unk')


# In[11]:


adult_data.describe(include='all').transpose()


# ### Checking for Outliers

# In[12]:


# construct box plot for continuous variables
plt.figure(figsize=(10,10))
adult_data.boxplot()


# ### Treating Outliers

# In[13]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[14]:


for column in adult_data.columns:
    if adult_data[column].dtype != 'object': 
        lr,ur=remove_outlier(adult_data[column])
        adult_data[column]=np.where(adult_data[column]>ur,ur,adult_data[column])
        adult_data[column]=np.where(adult_data[column]<lr,lr,adult_data[column])


# In[15]:


# construct box plot for continuous variables
plt.figure(figsize=(10,10))
adult_data.boxplot()


# ### Checking for Correlations

# In[16]:


adult_data.head()


# In[17]:


adult_data['education'].unique()


# In[18]:


adult_data.describe()


# capital gain and capital loss are both 0 after removing outliers. These 2 variables can be dropped

# In[19]:


adult_data.drop(['capital gain','capital loss'], axis = 1,inplace=True)


# There is hardly any correlation between the numeric variables

# ### Converting all objects to categorical codes

# In[20]:


for feature in adult_data.columns: 
    if adult_data[feature].dtype == 'object': 
        print('\n')
        print('feature:',feature)
        print(pd.Categorical(adult_data[feature].unique()))
        print(pd.Categorical(adult_data[feature].unique()).codes)
        adult_data[feature] = pd.Categorical(adult_data[feature]).codes


# In[21]:


adult_data.head()


# ### Train Test Split

# In[22]:


# Copy all the predictor variables into X dataframe
X = adult_data.drop('salary', axis=1)

# Copy target into the y dataframe. 
y = adult_data[['salary']]


# In[23]:


# Split X and y into training and test set in 80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 , random_state=1)


# ### Logistic Regression Model

# In[24]:


# Fit the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# ### Predicting on Training and Test dataset

# In[25]:


ytrain_predict = model.predict(X_train)
ytest_predict = model.predict(X_test)


# ### Getting the Predicted Classes and Probs

# In[26]:


ytest_predict_prob=model.predict_proba(X_test)
pd.DataFrame(ytest_predict_prob).head()


# ## Model Evaluation

# In[27]:


# Accuracy - Training Data
model.score(X_train, y_train)


# ### AUC and ROC for the training data

# In[28]:


# predict probabilities
probs = model.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr)


# In[29]:


# Accuracy - Test Data
model.score(X_test, y_test)


# ### AUC and ROC for the test data

# In[30]:


# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr)


# ### Confusion Matrix for the training data

# In[31]:


confusion_matrix(y_train, ytrain_predict)


# In[32]:


print(classification_report(y_train, ytrain_predict))


# ### Confusion Matrix for test data

# In[33]:


cnf_matrix=confusion_matrix(y_test, ytest_predict)
cnf_matrix


# In[34]:


#Test Data Accuracy
test_acc=model.score(X_test,y_test)
test_acc


# In[35]:


print(classification_report(y_test, ytest_predict))


# # Conclusion

# Accuracy on Test data is 76% and on Train data is 74%.  
# AUC is 64% for both.   
# Recall and Precision is low and same on both data.  
# While the model results between training and test sets are similar, indicating no under or overfitting issues, overall prediction of the model is weaker in terms of predicting salary > 50k. Considering the class imabalance ratio is moderate and not high, with more training data, the model is expected to perform even better.

# Note: Alternatively, one hot encoding can also be done instead of label encoding on categorical variables before building the logistic regression model. Do play around with these techniques using one hot encoding as well.

# #Running IN Google Colab
# Importing jupyter notebook
# 1. Login to Google
# 2. Go to drive.google.com
# 3. Upload jupyter notebook file into the drive
# 4. double click it, or right click -&gt; open with -&gt; google colaboratory
# Alternatively,
# 1. Login to Google
# 2. Go to https://colab.research.google.com/notebooks/intro.ipynb#recent=true
# 3. Upload the jupyter notebook
# 
# Loading dataset into colab
# #Use the below code to load the dataset
# from google.colab import files
# uploaded = files.upload() # upload file here from local
# import io
# df2 = pd.read_csv(io.BytesIO(uploaded[&#39;Filename.csv&#39;])) #give the filename in quotes
# 
# Go to Runtime > change Runtime type > check if it points to Python

# In[36]:


import pickle


# In[38]:


pickle_out = open("adult_flask.pkl", "wb")
pickle.dump(model, pickle_out)
loaded_model = pickle.load(open("adult_flask.pkl","rb"))
result = loaded_model.score(X_test, y_test)
print(result)


# In[71]:


adult_data.columns


# In[ ]:




