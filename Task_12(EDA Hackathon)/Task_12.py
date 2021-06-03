#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
import math


# In[3]:


data=pd.read_csv('data.csv')


# In[4]:


data.head()


# In[5]:


data= data.drop(['Unnamed: 0'], axis=1)


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.dropna(inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


sns.distplot(data['assists'])


# In[11]:


sns.distplot(data['roadKills'])


# In[14]:


df=data


# In[15]:


count, bins_count = np.histogram(data.kills, bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of Kills')
plt.legend()


# In[16]:


count, bins_count = np.histogram(df.DBNOs, bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of Number of enemy players knocked')
plt.legend()


# In[17]:


def visualization (col, num_bin=10):
    title = col[0].upper() + col[1:]
    f,axes=plt.subplots()
    plt.xlabel(title)
    plt.ylabel('Log Count')
    axes.set_yscale('log')
    data.hist(column=col,ax=axes,bins=num_bin)
    plt.title('Histogram of ' + title)
    plt.show()
    
    tmp = data[col].value_counts().sort_values(ascending=False)

    print('Min value of ' + title + ' is: ',min(tmp.index))
    print('Max value of ' + title + ' is: ',max(tmp.index))


# In[18]:


visualization('assists')


# In[19]:



visualization('roadKills')


# In[20]:


data.drop(data[data['roadKills']>=10].index,inplace=True)


# In[21]:


visualization('roadKills')


# In[22]:


visualization('kills')


# In[23]:


data.drop(data[data['kills']>=35].index,inplace=True)


# In[24]:


visualization('kills')
visualization('killStreaks')


# In[25]:



visualization('teamKills')


# In[26]:


visualization('headshotKills', num_bin=10)


# In[27]:


visualization('vehicleDestroys',num_bin=5)


# In[28]:



visualization('revives',num_bin=20)


# In[29]:


visualization('damageDealt', num_bin=1000)


# In[30]:


visualization('weaponsAcquired',num_bin=30)


# In[31]:


data.drop(data[data.weaponsAcquired>=50].index,inplace=True)


# In[32]:


visualization('boosts',num_bin=20)


# In[33]:



visualization('heals', num_bin=100)


# In[34]:


data.drop(data[data.heals>=40].index,inplace=True)


# In[35]:


visualization('walkDistance',num_bin=250)


# In[36]:


visualization('boosts',num_bin=20)


# In[37]:


data.drop(data[data['walkDistance']>=10000].index,inplace=True)


# In[38]:


visualization('walkDistance',num_bin=250)


# In[39]:


visualization('rideDistance',num_bin=500)


# In[40]:


data.drop(data[data.rideDistance >=15000].index, inplace=True)


# In[41]:


visualization('longestKill', num_bin=100)


# In[42]:


data.drop(data[data['longestKill']>=800].index,inplace=True)


# In[43]:


data.shape


# In[44]:


cols_to_drop = ['Id','matchId','groupId','matchType']
cols_to_fit = [col for col in data.columns if col not in cols_to_drop]
corr = data[cols_to_fit].corr()


# In[45]:


plt.figure(figsize=(9,7))
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,linecolor='white',linewidths=0.1,cmap='RdBu')
plt.show()


# In[46]:


df=data


# In[47]:


print('The average person kills {:.4f} players'.format(df['kills'].mean()))
print('50% of people have ',df['kills'].quantile(0.50),' kills or less')
print('75% of people have ',df['kills'].quantile(0.75),' kills or less')
print('99% of people have ',df['kills'].quantile(0.99),' kills or less')
print('while the most kills recorded in the data is', df['kills'].max())


# In[48]:


data = df.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(20,15))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title('Kill Count',fontsize=15)
plt.xlabel('Kills', fontsize=15)
plt.ylabel('Count',fontsize=13)
plt.show()


# #### Maximum number of enemy players killed in a short time
# This is the number of enemy players killed in a short time by each player.

# In[49]:


df['matchType'].value_counts()


# In[50]:


plt.figure(figsize=(20,15))
sns.countplot(df['matchType'], )
plt.title('Match Type',fontsize=15)
plt.xlabel('Match Type', fontsize=15)
plt.ylabel('Count',fontsize=13)
plt.show()


# Observations :
# #### From the above graph, it is clear that the most played matchtype is squad-fpp
# #### The least played matchtype is normal-duo

# ### Damage to enemy players:

# In[51]:


data = df.copy()

data = data[data['kills']==0]
plt.figure(figsize=(15,10))
plt.title('Damage Dealt by 0 killers',fontsize=15)
sns.distplot(data['damageDealt'])
plt.xlabel('Damage Dealt', fontsize=15)
plt.ylabel('Density',fontsize=13)
plt.show()


# Here, we see a distribution of how much damage, players that dont kill anyone, can inflict on there enemies. We can see that most players dont deal out too much, this is most likely all the new players trying to figure out the controls and getting to know the game while they continually get beaten up by the more expereince players.

# In[52]:


data = df[df['winPlacePerc'] == 1]

plt.figure(figsize=(15,10))
plt.title('Match duration for winners',fontsize=15)
sns.distplot(data['matchDuration'])
plt.xlabel('Match Duration', fontsize=15)
plt.ylabel('Density',fontsize=13)
plt.show()


# In[53]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot of all numerical data')
df.boxplot() 


# In[54]:


plt.scatter(df.winPlacePerc, df['walkDistance'])
plt.title('Relation between Targer of Prediction and walk Distance')


# In[55]:


plt.scatter(df.winPlacePerc, df['winPoints'])
plt.title('Relation between Targer of Prediction and winPoints')


# In[56]:


plt.figure(figsize=(15,10))
sns.jointplot(x='winPlacePerc', y='killStreaks', data=df, color='b')
plt.xlabel('Win Place Prec', fontsize=15)
plt.ylabel('Kill streaks',fontsize=13)
plt.show()


# In[57]:


print('The average person kills {:.4f} players on their own team'.format(df['teamKills'].mean()))
print('50% of people have killed ',df['teamKills'].quantile(0.50),' team players')
print('75% of people have killed ',df['teamKills'].quantile(0.75),' team players')
print('99% of people have killed ',df['teamKills'].quantile(0.99),' team players')
print('while the most kills recorded in the data is', df['teamKills'].max())


# In[58]:


sns.jointplot(x='winPlacePerc', y='teamKills', data=df, ratio=3, color='r')


# #### Hypothesis Testing

# In[59]:


#Help from Python
from scipy.stats import shapiro

DataToTest = data['winPlacePerc']

stat, p = shapiro(DataToTest)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print('Normal distribution')
else:
    print('Not a normal distribution')


# In[60]:


#Lets genrate normally distributed data from Python
from numpy.random import randn
DataToTest = randn(100)


# In[61]:


DataToTest


# In[62]:


stat, p = shapiro(DataToTest)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print('Normal distribution')
else:
    print('Not a normal distribution')


# In[63]:


df['damageDealt'].mean()


# $$z= \frac{\bar{x}-\mu}{\frac{\sigma}{\sqrt n}}$$
# $$H_{o}: \space \space  \mu = 130$$$$H_{a}: \space \space  \mu \ne 130$$
# step 1: create the hypothesis (Null and Alternate Hypothesis)
# 
# Step 2: Appropriate statistical test
# 
# step 3: let set $\alpha$ as .05 i.e Type l error
# 
# step 4: Get data
# 
# Step 5: Analyze

# # This is a 2 sided test

# value of $z$ at .05 making it .025 for 2 sided we know from $z table=$ $\underline{+}$1.96

# In[64]:


sampData=df['damageDealt'][np.argsort(np.random.random(1000))[:70]]


# In[65]:


meanSampData=sampData.mean()
hypMean=130
N=70
standPop=np.std(df['damageDealt'])


# In[66]:


(meanSampData-hypMean)/(standPop/math.sqrt(N))


# as calculated z score -1.20 is more than -1.96 (tabular z score), Hence we accept the null hypothesis
# 
# Observed value = -1.20
# Critical value = -1.96

# # Ttest

# $$t= \frac{\bar{x}-\mu}{\frac{\sigma}{\sqrt n}}$$
# expected mean hence $\mu$ degree of freedom =N-1

# In[68]:


import scipy.stats as st


# In[69]:


st.ttest_1samp(sampData,130)


# In[70]:


0.09>0.05


#  Hence we accept the null hypothesis
