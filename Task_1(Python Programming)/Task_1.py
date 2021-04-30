#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1
print("Hello, World!")


# In[2]:


# Question 2

n=int(input())
if(n%2!=0):
    print("Weird")
else:
    if n==2 or n==4 or n>20:
        print("Not Weird")
    else:
        print("Weird")


# In[3]:


# Question 3
a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)


# In[4]:


# Question 4
a = int(input())
b = int(input())
print(a//b)
print(a/b)


# In[5]:


# Question 5
n = int(input())
for i in range(0,n):
    print(i*i)


# In[6]:


# Question 6
def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%4==0:
        if year%100==0:
            if year%400==0:
                return True
            else:
                return False
        else:
            return True;
    return leap

year = int(input())
print(is_leap(year))


# In[7]:


# Question 7
n = int(input())
for i in range(1,n+1):
    print(i,end="")

