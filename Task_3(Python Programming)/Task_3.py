#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1

import cmath
c=complex(input())
print(abs(c))
print(cmath.phase(c))


# In[2]:


# Question 2

import math

ab = float(input())
bc = float(input())

angle_mbc = round(math.degrees(math.atan2(ab, bc)))
print(str(angle_mbc)+chr(176))


# In[3]:


# Question 3


for i in range(1,int(input())+1):
    print((((10**i)-1)//9)**2)


# In[4]:


# Question 4

import math

a=int(input())
b=int(input())
print(a//b)
print(a%b)
print(divmod(a,b))


# In[6]:


# Question 5

import math
a=int(input())
b=int(input())
m=int(input())
print(pow(a,b))
print(pow(a,b,m))


# In[ ]:


# Question 6
import math
a=int(input())
b=int(input())
c=int(input())
d=int(input())
print(pow(a,b)+pow(c,d))


# In[ ]:




