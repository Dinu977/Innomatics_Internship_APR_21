#!/usr/bin/env python
# coding: utf-8

# # Dataset Description:

# The dataset was released by Aspiring Minds from the Aspiring Mind Employment Outcome 2015 (AMEO). The study is primarily limited  only to students with engineering disciplines. The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills. The dataset also contains demographic features. The dataset  contains  around  40 independent variables and 4000 data points. The independent variables are both continuous and categorical in nature. The dataset contains a unique identifier for each candidate

# # Objective:
# 

# We have to analyse on the given dataset on aspiring_minds_employability_outcomes_2015 dataset.
# We have to do univariant and bivariant analysis on variables.
# Also, we have to do hypothetical testing on “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a freshgraduate.” Test this claim with the data given to us.
# And also check whether there is a relationship between gender and designation or not?
# 

# import pandas as pd

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[3]:


data=pd.read_excel('aspiring_minds_employability_outcomes_2015.xlsx')


# In[4]:


data.head(10)


# In[5]:


data.shape


# In[6]:


data.describe


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# # Univariate Analysis
# 

# # ID variable

# In[10]:


sn.distplot(data['ID'])


# In[11]:


plt.hist(data.ID, rwidth = 0.9, color = 'orange')
plt.title('ID Histogram')


# In[12]:


sn.countplot(data['ID'])


# In[13]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot ID numerical data')
data.boxplot('ID')


# ID values are present in the range of (0.25-0.4)*1e^6

# # Salary variable

# In[14]:


sn.distplot(data['Salary'])


# In[15]:


plt.hist(data.Salary, rwidth = 0.9, color = 'orange')
plt.title('Salary Histogram')


# In[16]:


sn.countplot(data['Salary'])


# In[17]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Salary numerical data')
data.boxplot('Salary')


# Many employees got a salary in the range of (0.25 to 0.5)*1e^6

# # 10percentage

# In[18]:


sn.distplot(data['10percentage'])


# In[19]:


plt.hist(data['10percentage'], rwidth = 0.9, color = 'orange')
plt.title('10percentage Histogram')


# In[20]:


sn.countplot(data['10percentage'])


# In[21]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot 10percentage numerical data')
data.boxplot('10percentage')


# Most of the students got 75-90 % marks in 10th standard.

# # 12 Percentage

# In[22]:


sn.distplot(data['12percentage'])


# In[23]:


plt.hist(data['12percentage'], rwidth = 0.9, color = 'orange')
plt.title('12percentage Histogram')


# In[24]:


sn.countplot(['12percentage'])


# In[25]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot 12percentage numerical data')
data.boxplot('12percentage')


# Most of the students got 65-80 % marks in 12th standard.

# # 12 graduation

# In[26]:


sn.distplot(data['12graduation'])


# In[27]:


plt.hist(data['12graduation'], rwidth = 0.8, color = 'orange')
plt.title('12graduation Histogram')


# In[28]:


sn.countplot(['12graduation'])


# In[29]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot 12graduation numerical data')
data.boxplot('12graduation')


# Most of the students passed 12th standard between 2007 t0 2010.

# # College ID

# In[30]:


sn.distplot(data['CollegeID'])


# In[31]:


plt.hist(data['CollegeID'], rwidth = 0.8, color = 'orange')
plt.title('College Id Histogram')


# In[32]:


sn.countplot(data['CollegeID'])


# In[33]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot College ID numerical data')
data.boxplot('CollegeID')


# Most of the college Id present between 0 to 2500.

# # College Tier

# In[34]:


sn.distplot(data['CollegeTier'])


# In[35]:


plt.hist(data['CollegeTier'], rwidth = 0.8, color = 'orange')
plt.title('College Tier Histogram')


# In[36]:


sn.countplot(data['CollegeTier'])


# In[37]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot College Tier numerical data')
data.boxplot('CollegeTier')


# Few students got placed in tier 1 company.

# # collegeGPA

# In[38]:


sn.distplot(data['collegeGPA'])


# In[39]:


plt.hist(data['collegeGPA'], rwidth = 0.8, color = 'orange')
plt.title('College GPA Histogram')


# In[40]:


sn.countplot(data['collegeGPA'])


# In[41]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot College GPA numerical data')
data.boxplot('collegeGPA')


# Most of students scored between 65-75 GPA. 

# # College City ID

# In[42]:


sn.distplot(data['CollegeCityID'])


# In[43]:


plt.hist(data['CollegeCityID'], rwidth = 0.8, color = 'orange')
plt.title('College city id Histogram')


# In[44]:


sn.countplot(data['CollegeCityID'])


# In[45]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot College city id numerical data')
data.boxplot('CollegeCityID')


# Most of the college city Id present between 0 to 2500.

# # college city tier

# In[46]:


sn.distplot(data['CollegeCityTier'])


# In[47]:


plt.hist(data['CollegeCityTier'], rwidth = 0.8, color = 'orange')
plt.title('College city Tier Histogram')


# In[48]:


sn.countplot(data['CollegeCityTier'])


# In[49]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot College city tier numerical data')
data.boxplot('CollegeCityTier')


# Most students present in the 0 college city tier.

# # Graduation Year

# In[50]:


sn.distplot(data['GraduationYear'])


# In[51]:


plt.hist(data['GraduationYear'], rwidth = 0.8, color = 'orange')
plt.title('Graduation Year Histogram')


# In[52]:


sn.countplot(data['GraduationYear'])


# In[53]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Graduation year numerical data')
data.boxplot('GraduationYear')


# Many students graduated between 2012 to 2014.

# # English

# In[54]:


sn.distplot(data['English'])


# In[55]:


sn.countplot(data['English'])


# In[56]:


plt.hist(data['English'], rwidth = 0.8, color = 'orange')
plt.title('English score Histogram')


# In[57]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot English numerical data')
data.boxplot('English')


# Most of students score 400 to 600 score in english.

# # Logical

# In[58]:


sn.distplot(data['Logical'])


# In[59]:


sn.countplot(data['Logical'])


# In[60]:


plt.hist(data['Logical'], rwidth = 0.8, color = 'orange')
plt.title('Logical score Histogram')


# In[61]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Logical numerical data')
data.boxplot('Logical')


# Most of students score 450 to 600 score in logical.

# # Quant

# In[62]:


sn.distplot(data['Quant'])


# In[63]:


sn.countplot(data['Quant'])


# In[64]:


plt.hist(data['Quant'], rwidth = 0.8, color = 'orange')
plt.title('Quant score Histogram')


# In[65]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Quant numerical data')
data.boxplot('Quant')


# Most of students score 500 to 650 score in Quant.

# # Computer Programming

# In[66]:


sn.distplot(data['ComputerProgramming'])


# In[67]:


sn.countplot(data['ComputerProgramming'])


# In[68]:


plt.hist(data['ComputerProgramming'], rwidth = 0.8, color = 'orange')
plt.title('ComputerProgramming score Histogram')


# In[69]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Computer Programming numerical data')
data.boxplot('ComputerProgramming')


# Most students scored 450 to 550 score and same number of students also scored -1.

# # Electronics And Semicon

# In[70]:


sn.distplot(data['ElectronicsAndSemicon'])


# In[71]:


sn.countplot(data['ElectronicsAndSemicon'])


# In[72]:


plt.hist(data['ElectronicsAndSemicon'], rwidth = 0.8, color = 'orange')
plt.title('Electronics And Semicon score Histogram')


# In[73]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Electronics And Semicon numerical data')
data.boxplot('ElectronicsAndSemicon')


#  Electronics And Semicon department students average score is between 300 to 400. 

# # Computer Science

# In[74]:


sn.distplot(data['ComputerScience'])


# In[75]:


sn.countplot(data['ComputerScience'])


# In[76]:


plt.hist(data['ComputerScience'], rwidth = 0.8, color = 'orange')
plt.title('Computer Sciencescore Histogram')


# In[77]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot Computer Science numerical data')
data.boxplot('ComputerScience')


#  Computer Science department students average score is between 400 to 500. 

# In[78]:


sn.distplot(data['conscientiousness'])


# In[79]:


sn.countplot(data['conscientiousness'])


# In[80]:


plt.hist(data['conscientiousness'], rwidth = 0.8, color = 'orange')
plt.title('conscientiousness Histogram')


# In[81]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot conscientiousness numerical data')
data.boxplot('conscientiousness')


# Many students score marks in conscientiousness between -0.5 to 0.75.

# # extraversion

# In[82]:


sn.distplot(data['extraversion'])


# In[83]:


sn.countplot(data['extraversion'])


# In[84]:


plt.hist(data['extraversion'], rwidth = 0.8, color = 'orange')
plt.title('Extraversion Histogram')


# In[85]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot extraversion numerical data')
data.boxplot('extraversion')


# Many students score -0.75 to 0.75 marks in extraversion.

# # nueroticism

# In[86]:


sn.distplot(data['nueroticism'])


# In[87]:


sn.countplot(data['nueroticism'])


# In[88]:


plt.hist(data['nueroticism'], rwidth = 0.8, color = 'orange')
plt.title('nueroticism Histogram')


# In[89]:


plt.figure(figsize = (14, 7))
plt.title('Boxplot nueroticism numerical data')
data.boxplot('nueroticism')


# Many students score between -1 to 0.5 marks in nueroticism.

# # Gender 

# In[90]:


plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sn.countplot(x="Gender", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# There are 76.06 % male and 23.94% female are present in dataset. 

# # 10Board

# In[91]:


plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sn.countplot(x="10board", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[92]:


data['10board'].value_counts()


# Most students studied from cbse board and state board.

# # 12 board

# In[93]:


plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sn.countplot(x="12board", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[94]:


data['12board'].value_counts()


# Most students studied from cbse board and state board.

# # Designation

# In[95]:


plt.figure(figsize=(15,8))
total = float(len(data) )

ax = sn.countplot(x="Designation", data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[96]:


data['Designation'].value_counts()


# There are maximum number of software engineer employees in the company.

# # Bivariant  Analysis

# # Relation between Salary   & college GPA

# In[97]:


#Scatter Plot
plt.scatter(data['Salary'], data['collegeGPA'])
plt.title('Relation between salary and collegeGPA')


# In[98]:


# hexbin plot
plt.hexbin(data.Salary, data.collegeGPA, gridsize = 50, cmap ='Greens')


# Many employees get nearly 0.5*1e^6 salary whose college GPA was between 60 to 80 %.

# # Relation between Salary   & computer programming

# In[99]:


#Scatter Plot
plt.scatter(data['Salary'], data['ComputerProgramming'])
plt.title('Relation between salary and ComputerProgramming')


# In[100]:


# hexbin plot
plt.hexbin(data.Salary, data.ComputerProgramming, gridsize = 50, cmap ='Greens')


# Many employees get nearly 0.5*1e^6 to 1e^6 salary who score between 200 to 600 in computer programming.

# # Relation between Salary   & Domain

# In[101]:


#Scatter Plot
plt.scatter(data['Salary'], data['Domain'])
plt.title('Relation between salary and domain')


# In[102]:


# hexbin plot
plt.hexbin(data.Salary, data.Domain, gridsize = 50, cmap ='Greens')


# Employees get nearly 0.5*1e^6 salary whose domain lies between 0.5 to 1.

# # Relation between Salary   & agreeableness

# In[103]:


#Scatter Plot
plt.scatter(data['Salary'], data['agreeableness'])
plt.title('Relation between salary and agreeableness')


# In[104]:


# hexbin plot
plt.hexbin(data.Salary, data.agreeableness, gridsize = 50, cmap ='Greens')


# Employees get nearly 0.5*1e^6 salary whose agreeableness lies between 0 to 1.

# In[105]:


# pairplot
sn.pairplot(data)


# pairplot shows relation between every two variables

# In[107]:


#Relationship between salary and gender

#boxplot
plt.figure(figsize=(5,5))
sn.boxplot(x="Gender", y="Salary", data=data)
plt.show()


# In[108]:


#Relationship between salary and designation

#boxplot
plt.figure(figsize=(5,5))
sn.boxplot(x="Designation", y="Salary", data=data)
plt.show()


# For all dependent variables , the average salary is about 0.5*1e^6.

# #  Hypothetical Testing

# Testing for “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.”

# Alternate Hypotesis=> H1 : $\mu$<300000

# Null hypothesis=> H0: $\mu$>=300000

# In[109]:


# function for calculating z-score
def calculate_t_score(sample_mean, pop_mean, sample_std, sample_size):
    numerator = sample_mean - pop_mean
    denominator = sample_std/(sample_size**0.5)
    return numerator/denominator


# In[110]:


# sample and population mean is given

sample = data['Salary'] 
pop_mean = 300000


# In[111]:


# calculating sample mean

sample_mean = sample.mean()
sample_mean


# In[112]:


import statistics as st
sample_stdev=st.stdev(sample)
sample_stdev


# In[113]:


t_score = calculate_t_score(sample_mean, pop_mean, sample_stdev, len(sample))
t_score


# In[114]:


from scipy.stats import t
CI=0.95
alpha=1-CI
dof=len(sample)-1

t_critical= t.ppf(alpha,dof)
t_critical


# In[115]:


# Conclusion using left tail test

if t_score<-t_critical:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# # Conclude : 

# We are not able to reject Null hypothesis i.e “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer we can earn more than 2.5-3 lakhs as a freshgraduate.”

# # Is there a relationship between specialisation and gender?

# In[116]:


data['Gender'].value_counts()


# In[117]:


data['Designation'].value_counts()


# Lets make a bold Claim that Gender and Dedignation are dependent.
# 
# Alternate Hypothesis => H1:They are Dependent                                                                                                      
#  

# Null Hypothesis => H0:They are Independent

# In[118]:


#  Looking at the freqency distribution

pd.crosstab(data['Designation'], data['Gender'], margins=True)


# In[119]:


# These are the observed frequencies

observed = pd.crosstab(data['Designation'], data['Gender'])

observed


# In[120]:


# chi2_contigency returns chi2 test statistic, p-value, degree of freedoms, expected frequencies
from scipy.stats import chi2
from scipy.stats import chi2_contingency
chi2_contingency(observed)


# In[121]:


# Computing chi2 test statistic, p-value, degree of freedoms

chi2_test_stat = chi2_contingency(observed)[0]
pval = chi2_contingency(observed)[1]
df = chi2_contingency(observed)[2]
print(pval,chi2_test_stat)


# In[122]:


confidence_level = 0.95

alpha = 1 - confidence_level

chi2_critical = chi2.ppf(1 - alpha, df)

chi2_critical


# In[123]:


if(chi2_test_stat > chi2_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# In[124]:


if(pval < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# # conclusion:

# We have to reject rhe NULL hypothesis i.e. both the Gender and Designation variables are dependent on each other.

# In[ ]:




