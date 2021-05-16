#!/usr/bin/env python
# coding: utf-8

# # Import stats_data.csv file using pandas module

# In[2]:



import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 
import statistics as stats
import seaborn as sn

df = pd.read_csv('stats_data.csv')
df.head(5)


# # Mean  ($ \mu $)
# * Mean stands for average of given data.
# * Value of mean is equal to sum of all data divided by number of data.
# ### $\mu $ = $\frac{x_1  +  x_2 + .... + x_n}{n} $
# * It is a type of Central tendency.
# * It is also called a univariant.
# 

# ### Q.  Calculate mean value of Mthly_HH_Income feature from stats_data.csv

# In[3]:


data = df.Mthly_HH_Income
data.ravel()


# In[4]:


# Method I : With Library Function
from statistics import mean
mean = mean(data)
print(mean)


# In[5]:



sum_of_all_data = 0
for i in range(0, len(data)):
    sum_of_all_data+=data[i]
print(sum_of_all_data/len(data))


# In[6]:




y = []
for i in range(0, len(df.Mthly_HH_Income)):
    y.append(1)

plt.scatter(df.Mthly_HH_Income, y)
plt.scatter(mean, 1, label = 'mean', marker = '*', color = 'red', s = 300)

plt.title('Mean of Mthly_HH_Income')
plt.legend()
plt.show()


# In[ ]:





# # Median
# * Median is the middle number in a sorted list of numbers.
# * n = Total number of data
# * ###### Median = $\frac{n}{2}$th term ; when n $\in$ odd number
# * ###### Median = (($\frac{n}{2}$)th term + ($\frac{n}{2}$+1)th term)/2 ; When n $\in$ Even number

# ###  Q. Calculate median value of Mthly_HH_Income feature from stats_data.csv

# In[7]:



from statistics import median
print(median(data))


# In[8]:



n = len(data)
x = data.tolist()
x.sort()
if n%2 == 0:
    median = (x[n//2] + x[n//2-1])/2
    print(median)
else:
    median = x[n//2]
    print(median)


# In[9]:




y = []
for i in range(0, len(df.Mthly_HH_Income)):
    y.append(1)

plt.scatter(df.Mthly_HH_Income, y)
plt.scatter(median, 1, label = 'median', marker = '*', color = 'green', s = 300)

plt.title('Median of Mthly_HH_Income')
plt.legend()
plt.show()


# In[ ]:





# # Mode
# * Most frequent data in the list called Mode.
# * A set of data may have one mode, more than one mode, or no mode at all.
# * It is robust to outlier.

# ###  Q. Calculate mode value of No_of_Fly_Members feature from stats_data.csv

# In[10]:


df.No_of_Fly_Members.ravel()


# In[11]:


mode_data = df.No_of_Fly_Members


# In[12]:



from statistics import mode
print(mode(mode_data))


# In[13]:


# Method II
element = mode_data.tolist()
frequency = []
maxi = 0
s = list(set(element))
for i in range(0, len(s)):
    freq = element.count(s[i])
    frequency.append(freq)
    if freq>maxi:
        maxi = freq
        mode = s[i]
print(mode)


# In[14]:




plt.bar(s, frequency, label = 'Longest bar is mode', color = 'yellow')
plt.title('Mode')
plt.legend()
plt.show()


# In[ ]:





# # Variance
# * The variance is a measure of variability.
# * Variance tells the degree of spread present in data set.
# * Larger the spread, Larger the variance.
# 
# ***There are two types of variance :***
# * i) Population variance : It is denoted by $\sigma^{2}$
# * ii) Sample variance : It is denoted by $S^{2}$
# 
# * ##### Population Variance($\sigma^{2}$) = $\frac{\Sigma(x_i-\mu)^{2}}{n}$
# * ##### Sample variance( $S^{2}$)= $\frac{\Sigma(x_i-\bar{x})^{2}}{n-1}$

# ### Q. Calculate Vaiance of No_of_Fly_Members.

# In[15]:


var_data = df.No_of_Fly_Members


# In[16]:




print('Population variance : ', stats.pvariance(var_data))
print('Sample variance : ', stats.variance(var_data))


# In[17]:




total = 0
mean = stats.mean(var_data)
for i in range(0, len(var_data)):
    total = total + (var_data[i]-mean)**2
pop_var = total/len(var_data)
print('Population variance : ', pop_var)
sample_var = total/(len(var_data)-1)
print('Sample variance : ', sample_var)


# In[18]:



y = []
for i in range(0, len(var_data)):
    y.append(1)

plt.scatter(var_data, y)
plt.scatter(stats.pvariance(var_data), 1, label = 'Population Variance', marker = '*', color = 'red', s = 300)
plt.scatter(stats.variance(var_data), 1, label = 'Sample Variance', marker = '.', color = 'black', s = 100)

plt.title('Variance of Mthly_HH_Income')
plt.legend()
plt.show()


# In[ ]:





# # Standard Deviation 
# * It is used to calculate the dispersion spread. 
# * It is nothing but the square root of Variance.
# * ##### Population Standard Deviation ( $\sigma$) =$\sqrt[2]{PopulationVariance}$ 
# * ##### Sample Standard Deviation (S)= $\sqrt[2]{SampleVariance}$

# ### Q. Calculate Standard deviation of No_of_Fly_Members.

# In[19]:




print('Population std deviation', stats.pstdev(var_data))
print('Sample std deviation', stats.stdev(var_data))


# In[20]:




pop_std_dev = pop_var**0.5
sample_std_dev = sample_var**0.5
print('Population std deviation : ', pop_std_dev)
print('Sample std deviation : ', sample_std_dev)


# In[21]:




y = []
for i in range(0, len(var_data)):
    y.append(1)

plt.scatter(var_data, y)
plt.scatter(stats.pstdev(var_data), 1, label = 'Population std dev', marker = '*', color = 'orange', s = 200)
plt.scatter(stats.stdev(var_data), 1, label = 'Sample std dev', marker = '.', color = 'black', s = 100)

plt.title('Standard deviation of Mthly_HH_Income')
plt.legend()
plt.show()


# In[ ]:





# # Correlation
# * Correlation measure the strength of a linear relationship between two quantitative variables.
# * It is denoted by $\rho$.
# * positive correlation means, If one variable increases then second variable also increases.
# * Negative correaltion means, If one variable increses then value of second variale will decrese.
# * ##### Correlation Coefficient($\rho$) = $\frac{\Sigma(x_i-\mu_x)*(y_i-\mu_y)}{n\sigma_x\sigma_y}$

# ### Q. Calculate Correlation between Mthly_HH_Income and Mthly_HH_Expense.

# In[22]:


# Data
Income = df.Mthly_HH_Income
Expense = df.Mthly_HH_Expense


# In[23]:




print("Correlation Coefficient : \n", np.corrcoef(Income, Expense))


# In[24]:




Income_mean = stats.mean(Income)
Expense_mean = stats.mean(Expense)
Income_std = stats.pstdev(Income)
Expense_std = stats.pstdev(Expense)

total = 0
for i in range(len(Income)):
    total+= (Income[i]-Income_mean) * (Expense[i]-Expense_mean)

corr_coeff = total/(len(Income) * Income_std * Expense_std)
print('Correlation Coefficient : ', corr_coeff)


# In[25]:




plt.scatter(Income, Expense)
plt.title('Income Vs Expense Graph')
plt.xlabel('Mthly_HH_Income')
plt.ylabel('Mthly_HH_Expense')
plt.show()


# In[ ]:





# # Normal Distribution
# * Normal distribution, also known as the Gaussian distribution.
# ### Features of normal distribution
# * It is symmetric about mean.
# * In normal distribution, Mean = median = mode.
# * Bell shpaed curve
# * Exactly half of the values are to the left of center and exactly half the values are to the right.
# * It follows the 68%, 95%, 99.7% probability distribution rules.

# In[26]:



normal=np.random.normal(df.Mthly_HH_Income)
sn.distplot(normal)


# In[27]:


normal_12000 = np.random.normal(loc = 20, scale = 5, size=12000)
sn.distplot(normal_12000)


# In[28]:


mean_normal= normal_12000.mean()

sigma = normal_12000.std()

one_std_right= mean_normal+sigma
one_std_left= mean_normal-sigma

two_std_right= mean_normal+(2*sigma)
two_std_left= mean_normal-(2*sigma)

three_std_right= mean_normal+(3*sigma)
three_std_left= mean_normal-(3*sigma)


# In[29]:


plt.figure(figsize=(20,12))
sn.set_style("darkgrid")
sn.distplot(normal_12000)

plt.axvline(mean_normal, color='black', label='Mean')

plt.axvline(one_std_right, color='yellow', label='Mean + 1SD')
plt.axvline(one_std_left, color='yellow', label='Mean - 1SD')
plt.axvline(two_std_right, color='green', label='Mean + 2SD')
plt.axvline(two_std_left, color='green', label='Mean - 2SD')
plt.axvline(three_std_right, color='blue', label='Mean + 3SD')
plt.axvline(three_std_left, color='blue', label='Mean - 3SD')
plt.title('Normal Distribution(68 - 95 - 99.7)')
plt.legend()
plt.show()


# In[ ]:





# # Positively Skewed and Negatively Skewed
# ### 1. Positive Skewed
# * It is also called right-skewed distribution. 
# * In positive skewed the mean, median and mode of the distribution are positive rather than negative or zero
# 
# ### 2. Negative Skewed
# * It is also called Left-skewed distribution.
# * more data are clustered around the right tail side.

# In[30]:


from PIL import Image
img =Image.open("pearson.png")
img_array = np.asarray(img)
plt.imshow(img_array)
plt.show()


# ### Effect on mean , median and mode due to skewness:
# * If skewness = 0, then all mean, median and mode are equal.
# * If skewness > 0, then mode < median < mean.
# * If skewness < 0 , then mean < median < mode.

# In[ ]:





# # Q-Q Plot 
# * A Q-Q plot is type of scatter plot and it is created by plotting two datasets of quantiles against one another.
# * It tells that the  distributions of two variables are same or not with respect to the locations.

# In[31]:


from scipy import stats
stats.probplot(df.Mthly_HH_Income, dist="norm", plot=plt)
plt.grid()


# In[ ]:





# # Box Cox 
# * Box cox transformation is used to convert non-normal dependent variable into normal curve.

# In[32]:


pareto_rv = np.random.pareto(df.Mthly_HH_Income)
sn.distplot(pareto_rv)


# In[33]:


stats.probplot(pareto_rv, dist="norm", plot=plt)
plt.grid()


# In[34]:




x_t, x = stats.boxcox(pareto_rv)
print(x)


# In[35]:


stats.probplot(x_t, dist="norm", plot=plt)
plt.grid()


# In[ ]:





# In[ ]:




