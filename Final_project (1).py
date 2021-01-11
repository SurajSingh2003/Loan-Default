#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


os.chdir('C:\\Users\\suraj\\Desktop\\DSP25')


# In[3]:


##impporting the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import boxcox


# In[5]:


df=pd.read_csv("python_project2.csv")


# In[6]:


##finding columns with missing values more than 25%

NA_col=df.isnull().sum()
Na_col=NA_col[NA_col.values>(0.25*len(df))]
plt.figure(figsize=(20,4))
Na_col.plot(kind='bar')
plt.title('List of column with missing value more than 25%')
plt.show()


# In[7]:


df.info()


# In[8]:


## variables with missing values 

df.isnull().sum()


# In[9]:


df.columns


# In[10]:


## converting variables to datetime format


# In[11]:


df.next_pymnt_d = pd.to_datetime(df.next_pymnt_d,errors='ignore')


# In[12]:


df.last_pymnt_d = pd.to_datetime(df.last_pymnt_d,errors='ignore')


# In[13]:


df.last_credit_pull_d= pd.to_datetime(df.last_credit_pull_d,errors='ignore')


# In[14]:


var=['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'issue_d', 'pymnt_plan', 'desc', 'purpose', 'title', 'zip_code',
       'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
       'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
       'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
       'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt',
       'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'default_ind']


# In[15]:


## states with total number of applicants 

df['addr_state'].value_counts()


# In[16]:


## loop to remove the columns with missing value more than 25%

for v in var:
    if (df[v].isnull().sum())>=213992:
        df=df.drop([v],axis=1)
        print(v)


# In[17]:


df.shape


# In[18]:


## checking the number of unique values in variable 

df['policy_code'].unique()


# In[19]:


## removing the variables not necessary for the analysis 


var_1= ['id','member_id', 'emp_title', 'title', 'zip_code', 'addr_state','policy_code']


# In[20]:


df['default_ind'].value_counts()


# In[21]:


for b in var_1:
    df=df.drop([b],axis=1)


# In[22]:


df.purpose.unique()


# In[23]:


m=df['purpose'].value_counts()
m.columns=['purpose','count']


# In[24]:


plt.subplots(figsize=(20,8))
plt.plot(m)


# In[25]:


n=df['verification_status'].value_counts()
print(n)
plt.subplots(figsize=(20,8))
plt.plot(n)


# In[26]:


cactegory_varubales=['term','grade', 'sub_grade','purpose',
       'emp_length', 'home_ownership', 'verification_status','earliest_cr_line', 'pymnt_plan','application_type','initial_list_status']


# In[27]:


for variable in cactegory_varubales:
    df[variable]= pd.get_dummies(df[variable])


# In[28]:


df.shape


# In[29]:


df.head()


# In[30]:


df1=df.filter(['loan_amnt', 'term', 'int_rate', 'installment', 'grade',
       'home_ownership', 'annual_inc', 'verification_status', 'pymnt_plan',
       'purpose', 'dti','issue_d', 'delinq_2yrs', 'inq_last_6mths','application_type',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc','out_prncp',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'default_ind'])
df1.shape


# In[31]:


correlations = df1.corr()
ax = plt.subplots(figsize=(30,30))
sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',cmap = 'twilight_shifted',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
sns.heatmap(correlations, annot = True)

plt.show()


# In[32]:


df1.issue_d


# In[33]:


df1=df1.set_index("issue_d")


# In[34]:


df1.shape


# In[35]:


test= df1.loc["Dec-15":"Jun-15"]


# In[36]:


train= df1.loc["May-15":"Jun-07"]


# In[37]:


train


# In[38]:


test


# In[39]:


train.isnull().sum()


# In[40]:


test.isnull().sum()


# In[41]:


train.shape


# In[42]:


test.shape


# In[43]:


x_train= train.iloc[:,0:20].values


# In[44]:


x_test= test.iloc[:,0:20].values


# In[45]:


y_train=train['default_ind'].values


# In[46]:


y_test=test['default_ind'].values


# In[47]:


x_train


# In[48]:


x_test


# In[49]:


## filling the missing values

from sklearn.preprocessing import Imputer
new=Imputer(missing_values='NaN', strategy ='mean', axis=0)


# In[50]:


x_train[:,18:19]=new.fit_transform(x_train[:,18:19])
x_test[:,18:19]=new.fit_transform(x_test[:,18:19])


# In[51]:


pd.DataFrame(x_train).head()


# In[52]:


pd.DataFrame(x_test).head()


# In[56]:


##transformation of varuables using standardrization 

from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_train= scl.fit_transform(x_train)
x_test= scl.fit_transform(x_test)


# In[57]:


plt.figure(figsize=(10,10))
sns.countplot(x='loan_amnt', data=df)
plt.title('Loan Amount')
plt.xticks(rotation = 90)
plt.show();


# In[58]:


plt.figure(figsize=(10,10))
sns.countplot(x='int_rate', data=df)
plt.title('Interest Rate')
plt.xticks(rotation = 90)
plt.show();


# # Model fitting to data  

# In[59]:


## DECISION TREE MODEL



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)


# In[60]:


y_predict= classifier.predict(x_test)
y_predict


# In[61]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_predict)
accuracy


# In[62]:


print(confusion_matrix(y_test,y_predict))


# In[63]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))


# In[64]:


## RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier 
classifier1 = RandomForestClassifier(n_estimators =500, criterion = 'entropy')
classifier1.fit(x_train, y_train)


# In[65]:


y_pred= classifier1.predict(x_test)
y_pred


# In[66]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[67]:


print(confusion_matrix(y_test,y_pred))


# In[68]:


print(classification_report(y_test,y_pred))


# In[69]:


## BAGGING CLASSIFIER

from sklearn.ensemble import BaggingClassifier
claass= BaggingClassifier(n_estimators =500, max_samples = 0.1)
claass.fit(x_train,y_train)


# In[70]:


y_prd= claass.predict(x_test)
y_prd


# In[71]:


accuracy = accuracy_score(y_test,y_prd)
accuracy


# In[72]:


print(confusion_matrix(y_test,y_prd))


# In[73]:


print(classification_report(y_test,y_prd))


# In[74]:


## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)


# In[76]:


y_predc = logmodel.predict(x_test)


# In[77]:


accuracy = accuracy_score(y_test,y_predc)
accuracy


# In[78]:



cm=confusion_matrix(y_test,y_predc)
print(cm)
sns.heatmap(cm, annot=True)


# In[79]:



print(classification_report(y_test,y_predc))


# In[80]:


# MODEL VALIDATION AND SELECTION


# In[81]:


from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
kf


# In[82]:


def get_score(model, x_train, x_test, y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)


# In[83]:


get_score(LogisticRegression(),x_train, x_test, y_train,y_test)


# In[84]:


get_score(BaggingClassifier(),x_train, x_test, y_train,y_test)


# In[85]:


get_score(RandomForestClassifier(),x_train, x_test, y_train,y_test)


# In[86]:


get_score(DecisionTreeClassifier(),x_train, x_test, y_train,y_test)


# In[ ]:


## with cross validationb score its clear that baggig classifier is the best fitted model for our analysis .

