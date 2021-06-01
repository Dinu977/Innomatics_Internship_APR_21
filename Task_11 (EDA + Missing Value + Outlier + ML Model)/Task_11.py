#!/usr/bin/env python
# coding: utf-8

# ##### The Census Income dataset has 48,842 entries. Each entry contains the following information about an individual:
# 

# <b>age</b>: <br>
# The age of an individual <br>
# Integer greater than 0
# 
# <br>
# 
# <b>workclass</b>: a general term to represent the employment status of an individual
# ○ Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
# Without-pay, Never-worked.
# 
# <br>
# <b>fnlwgt</b>: final weight. In other words, this is the number of people the census believes
# the entry represents..
# <br>Integer greater than 0
# 
# <br>
# <br>
# <b>education</b>: the highest level of education achieved by an individual.
# ○ Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc,
# 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 
# <b>education­num:</b> the highest level of education achieved in numerical form.
# <br>Integer greater than 0
# 
# <br>
#  <b>marital­status:</b> marital status of an individual. Married-civ-spouse corresponds to a
# civilian spouse while Married-AF-spouse is a spouse in the Armed Forces.<br>
#  marital-status: marital status of an individual. Married-civ-spouse corresponds to a
# civilian spouse while Married-AF-spouse is a spouse in the Armed Forces.
# <br>
# ○ Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
# Married-spouse-absent, Married-AF-spouse.
# <br>
# <br>
# <b>occupation:</b> the general type of occupation of an individual
# <br>○ Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,
# Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical,
# Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
# Armed-Forces.
# <br>
# <br>
# 
# <b>relationship:</b> represents what this individual is relative to others. For example an
# individual could be a Husband. Each entry only has one relationship attribute and is
# somewhat redundant with marital status. We might not make use of this attribute at all
# <br>○ Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# <br>
# <br>
# <b>race:</b> Descriptions of an individual’s race
# <br>○ White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# <br>
# <br>
# <b>sex:</b> the biological sex of the individual
# <br>○ Male, Female
# <br>
# <br>
# 
# <b>capital-gain:</b> capital gains for an individual
# ○ Integer greater than or equal to 0
# <br>
# <br>
# <b>capital-loss:</b> capital loss for an individual
# <br>○ Integer greater than or equal to 0
# <br>
# <br>
# <b>hours-per-week:</b> the hours an individual has reported to work per week
# <br>○ continuous.
# <br>
# <br>
# <b>native-country:</b> country of origin for an individual
# <br>○ United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
# Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran,
# Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal,
# Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia,
# Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El­Salvador,
# Trinadad&Tobago, Peru, Hong, Holand­Netherlands
# <br>
# <br>
# <b>the label:</b> whether or not an individual makes more than $50,000 annually.
# <br>○ <=50k, >50k
# <br>
# <br>

# In[139]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sms


# In[140]:


data=pd.read_csv('adult.csv')
data.head(10)


# In[141]:


data.shape


# In[142]:


data.dtypes


# In[210]:


df['workclass'].value_counts()


# In[212]:


df.isnull().sum()


# In[213]:


df.describe()


# In[ ]:





# In[144]:


data.nunique()


# In[145]:


data.describe().T


# In[146]:


df=data


# In[ ]:





# ## Univariate Analysis

# PDF

# In[159]:


sms.distplot(df['age'])


# There are more number people around 35-45 years

# In[161]:


sms.distplot(df['fnlwgt'])


# In[163]:


sms.distplot(df['educational-num'])


# Maximum number of educational-num is around 9

# In[ ]:





# In[165]:


sms.distplot(df['capital-gain'])


# In[166]:


sms.distplot(df['capital-loss'])


# In[167]:


sms.distplot(df['hours-per-week'])


# Maximum number of hours-per-week is around 40

# In[ ]:





# In[ ]:





# In[147]:


plt.hist(data.age, rwidth = 0.9, color = 'orange')
plt.title('Age Histogram')


# In[148]:


count, bins_count = np.histogram(df.age, bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of Age')
plt.legend()


# In[149]:


plt.hist(df.fnlwgt, rwidth = 0.9, color = 'red')
plt.title('Fnlwgt Histogram')


# In[150]:


count, bins_count = np.histogram(df.fnlwgt, bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of fnlwgt')
plt.legend()


# In[151]:


plt.hist(df['educational-num'], rwidth = 0.9, color = 'yellow')
plt.title('educational-num')


# In[152]:


count, bins_count = np.histogram(df['educational-num'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of educational-num')
plt.legend()


# In[153]:


plt.hist(df['capital-gain'], rwidth = 0.9, color = 'cyan')
plt.title('capital-gain')


# In[154]:


count, bins_count = np.histogram(df['capital-gain'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of capital-gain')
plt.legend()


# In[155]:


plt.hist(df['capital-loss'], rwidth = 0.9, color = 'green')
plt.title('capital-loss')


# In[156]:


count, bins_count = np.histogram(df['capital-loss'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of capital-loss')
plt.legend()


# In[157]:


plt.hist(df['hours-per-week'], rwidth = 0.9, color = 'blue')
plt.title('hours-per-week')


# In[158]:


count, bins_count = np.histogram(df['hours-per-week'], bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.title('PDF and CDF of hours-per-week')
plt.legend()


# In[ ]:





# Countplot

# In[168]:


sms.countplot(df['age'])


# In[169]:


sms.countplot(df['educational-num'])


# In[171]:


sms.countplot(df['hours-per-week'])


# In[173]:


sms.countplot(df['capital-gain'])


# In[175]:


sms.countplot(df['capital-loss'])


# In[177]:


sms.countplot(df['hours-per-week'])


# In[178]:


df.plot.hist()


# Workclass

# In[180]:


plt.figure(figsize=(12,5))

total = float(len(df['income']))

a = sms.countplot(x='workclass',data=df)

for f in a.patches:
    height = f.get_height()
    a.text(f.get_x() + f.get_width()/2., height+3, '{:1.2f}'.format((height/total)*100),ha="center")
plt.show()


# most of them belong to private workclass that is around 75%. without-play and never-play workclass has min count

# In[182]:


#Educational distribution

plt.figure(figsize=(20,5))

a= float(len(['income']))

a= ss.countplot(x='education',data=df)
for s in a.patches:
    height = s.get_height()
    a.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
plt.show()


# marital-status DistribuHs-grad has 32.32% of all the education attribute. pre-school has min.

# In[184]:


#Marital Status Distribution

plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sms.countplot(x="marital-status", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Married-civ-spouse has maximum number of samples. Married-AF-spouse has minimum number of obs.

# In[186]:


#Occupational Distribution

plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sms.countplot(x="occupation", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Prof-specialty has the maximum count. Armed-Forces has minimum samples in the occupation attribute.

# In[189]:


#Relationship Distribution

plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sms.countplot(x="relationship", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Husband has maximum percentage among all.

# In[191]:


#Race distribution

plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sms.countplot(x="race", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# white is maximun among all about 85.50%. black is second maximun.

# In[193]:


#Gender Distribution

total = float(len(df) )

ax = sms.countplot(x="gender", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# there are 2 unique categories in gender. frequency of male is higher than female.

# In[195]:


# Income distribution

plt.figure(figsize=(5,5))
total = float(len(df) )

ax = sms.countplot(x="income", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In income there is 2 group,group1(who earns more than 50k) 23.93% belong to income and group2(who earns less than 50k) 76% belong to income

# In[ ]:





# In[196]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot of all numerical data')
df.boxplot() 


# In[ ]:





# In[70]:


sms.countplot(x ='gender', data = df)


# In[71]:


sms.countplot(y ='education', data = df)


# In[72]:


sms.countplot(y ='marital-status', data = df)


# In[74]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sms.countplot(y = 'native-country', data = df)


# In[76]:


sms.countplot(y = 'workclass', data = df)


# In[78]:


sms.countplot(y = 'occupation', data = df)


# In[79]:


sms.countplot(y = 'relationship', data = df)


# In[81]:


sms.countplot(y = 'race', data = df)


# # Bivariate Analysis
# 

# In[82]:


plt.scatter(df.age, df['fnlwgt'])
plt.title('Relation between age and fntwgt')


# In[83]:


plt.scatter(df.age, df['educational-num'])
plt.title('Relation between age and educational num')


# In[ ]:





# In[84]:


# hexbin plot
plt.hexbin(df.age, df.fnlwgt, gridsize = 50, cmap ='Greens')


# In[85]:


plt.hexbin(df.age, df['educational-num'], gridsize = 50, cmap ='Greens')


# In[199]:


# Relationship between income and age

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="age", data=df)
plt.show()


# Income group(<=50k) has lower median "age"(34 year) than the Income group(>50k) which has median "age"(42 year).

# In[201]:


# Relationship between income and finalwgt

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="fnlwgt", data=df)
plt.show()


# In[203]:


# Relationship between income and educational-num

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="educational-num", data=df)
plt.show()


# In[205]:


# Relationship between income and capital-gain

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="capital-gain", data=df)
plt.show()


# Most of the capital gains value is accumulated at 0 for both the income group .

# In[207]:


#Relationship between income and capital-loss

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="capital-loss", data=df)
plt.show()


# This boxplot is similar to the capital gain boxplot where most of the values are concentrated on 0.
# 

# In[209]:


# Relationship between income and hours-per-week

plt.figure(figsize=(5,5))
sms.boxplot(x="income", y="hours-per-week", data=df)
plt.show()


# In[ ]:





# In[87]:


# pairplot
sms.pairplot(df)


# pairplot shows relation between every two variables

# In[ ]:





# In[88]:


data['workclass'].value_counts()


# mode is "Private" class

# In[89]:


data.columns


# In[90]:


data['occupation'].value_counts()


# In[91]:


data['native-country'].value_counts()


# In[92]:


data['marital-status'].value_counts()


# In[93]:


data['gender'].value_counts()


# In[94]:


data['race'].value_counts()


# In[95]:


data['income'].value_counts()


# In[96]:


data['education'].value_counts()


# In[97]:


sms.countplot(data['income'],palette='coolwarm',hue='gender',data=data)


# In[98]:


sms.countplot(data['income'],palette='coolwarm',hue='race',data=data)


# In[99]:


sms.countplot(data['income'],palette='coolwarm',hue='relationship',data=data)


# In[100]:


data['workclass']=data['workclass'].replace('?','Private')


# In[101]:


data['occupation']=data['occupation'].replace('?','Prof-specialty')


# In[102]:


data['native-country']=data['native-country'].replace('?','United-States')


# ## Feature Transformation

# In[103]:


data.education=data.education.replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],'school')
data.education=data.education.replace('HS-grad','high school')
data.education=data.education.replace(['Assoc-voc','Assoc-acdm','Prof-school','Some-college'],'higher')
data.education=data.education.replace('Bachelors','undergrad')
data.education=data.education.replace('Masters','grad')
data.education=data.education.replace('Doctorate','doc')


# In[104]:


data['education'].value_counts()


# In[105]:


data['marital-status'].value_counts()


# In[106]:


#marital status
data['marital-status']=data['marital-status'].replace(['Married-civ-spouse','Married-AF-spouse'],'Married')
data['marital-status']=data['marital-status'].replace('Never-married','not married')
data['marital-status']=data['marital-status'].replace(['Divorced','Separated','Widowed','Married-spouse-absent'],'other')


# In[107]:


data['marital-status'].value_counts()


# In[108]:


data.income=data.income.replace('<=50K',0)
data.income=data.income.replace('>50K',1)


# In[109]:


data['income'].value_counts()


# In[110]:


data.corr()


# In[111]:


sms.heatmap(data.corr(),annot=True)


# In[116]:


df.hist(figsize=(12,12), layout=(3,3), sharex=False);


# In[117]:


df.plot(kind='box', figsize=(12,12), layout=(3,3), sharex=False, subplots=True);


# In[118]:



px.pie(df, values='educational-num', names='education', title='% of edu', 
      color_discrete_sequence = px.colors.qualitative.T10)


# In[119]:



sms.countplot(df['education'], hue='gender', data=df, palette='seismic');


# ## Model building

# In[120]:


X= df.drop(['income'], axis=1)
y = df['income']


# In[121]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[122]:



df1= df.copy()
df1= df1.apply(LabelEncoder().fit_transform)
df1.head()


# In[123]:


ss= StandardScaler().fit(df1.drop('income', axis=1))


# In[124]:


X= ss.transform(df1.drop('income', axis=1))
y= df['income']


# In[125]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# ## logistic regression

# In[127]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

model = lr.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Acc on training data: {:,.3f}".format(lr.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(lr.score(X_test, y_test)))


# ## Random Forest classifier

# In[128]:



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model1 = rfc.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

print("Acc on training data: {:,.3f}".format(rfc.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(rfc.score(X_test, y_test)))


# In[129]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[130]:


print(confusion_matrix(y_test, prediction1))


# In[131]:


print(classification_report(y_test, prediction1))


# In[132]:


#Precision: tp/tp+fp

print('Precision =' , 10332/(10332+1286))


# In[133]:



# recall= tp/tp+fn

print('Recall =', 10332/(10332+806))


# ## for class : 1(>50k)

# In[134]:


print('Precision = ', 2229/(2229+806))


# In[135]:


print('Recall= ', 2229/(2229+1286))


# In[ ]:




