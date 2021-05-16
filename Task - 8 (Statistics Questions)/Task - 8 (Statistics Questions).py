#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1

def fact(n):
    fac=1
    for i in range(n):
        fac*=(i+1)
    return fac
    
lst=list(map(float,input().split(' ')))

p= lst[0]/(lst[0]+lst[1])
fact_6=fact(6)
su=0

for i in range(3,7):
    t=fact_6/(fact(i)*fact(6-i))
    
    k=p**i * (1-p)**(6-i)
    su+= t*k
    
print(round(su,3))


# In[2]:


# Question 2

def fact(n):
    fac=1
    for i in range(1,n+1):
        fac*=i
    return fac

lst=list(map(int,input().split()))

p=lst[0]/100
n=lst[1]
fac_10=fact(n)
su=0
ans=0

for i in range(3):
    t=fac_10/(fact(i)*fact(n-i))
    k= p**i *(1-p)**(n-i)
    tk=t*k
    if(i==2):
        ans=tk
    su+=tk
    
print(round(su,3))
print(round((1-su+tk),3))


# In[3]:


# Question 3

import math

def norm(mean,std,x):
    sq=(x-mean)
    t=sq/(std*(2**0.5))
    t=math.erf(t)
    t+=1
    t=t/2
    return t

lst=list(map(float,input().split(' ')))
mean=lst[0]
std=lst[1]

x=float(input())
ans=norm(mean,std,x)
print(round(ans,3))

lst=list(map(float,input().split(' ')))

ans=norm(mean,std,lst[1])-norm(mean,std,lst[0])
print(round(ans,3))


# In[4]:


# Question 4

import math

def norm(mean,std,x):
    sq=(x-mean)
    t=sq/(std*(2**0.5))
    t=math.erf(t)
    t+=1
    t=t/2
    return t

lst=list(map(float,input().split(' ')))
mean=lst[0]
std=lst[1]

x=float(input())
ans=norm(mean,std,x)
print(round(((1-ans)*100),2))

x=float(input())

ans=norm(mean,std,x)

print(round(((1-ans)*100),2))
print(round(ans*100,2))


# In[5]:


# Question 5

import math

x=int(input())
n=int(input())
sam_mean=int(input())*n
std=int(input())*(n**0.5)

z=(x-sam_mean)/std

t=0.5*(1+math.erf(z/(2**0.5)))

print(round(t,4))


# In[6]:


# Question 6

import math
x=int(input())
n=int(input())
mu=float(input())
std=float(input())

mu_sum=mu*n
std_sum=std*(n**0.5)

z=(x-mu_sum)/std_sum

ans=0.5*(1+math.erf(z/(2**0.5)))
print(round(ans,4))


# In[7]:


# Question 7


n=int(input())
mu=int(input())
std=int(input())
p=float(input())
z=float(input())

z=z*(std/(n**0.5))

A=mu-z
B=z+mu
print(round(A,2))
print(round(B,2))


# In[8]:


# Question 8

import statistics as st

n=int(input())

X=list(map(float,input().strip().split(' ')))
Y=list(map(float,input().strip().split(' ')))

mu_x=st.mean(X)
mu_y=st.mean(Y)

std_x=st.pstdev(X)
std_y=st.pstdev(Y)

su=0

for i in range(n):
    t=(X[i]-mu_x)*(Y[i]-mu_y)
    su+=t
    
cov=su/n

pearson_coeff= cov/(std_x*std_y)

print(round(pearson_coeff,3))


# In[9]:


# Question 9

import statistics as st

x=[]
y=[]
for i in range(5):
    lst=list(map(int,input().split(' ')))
    x.append(lst[0])
    y.append(lst[1])

mu_x=st.mean(x)
mu_y=st.mean(y)
std_x=st.pstdev(x)
std_y=st.pstdev(y)

su=0
for i in range(5):
    su+= (x[i]-mu_x)*(y[i]-mu_y)
    
cov=su/5

p_coeff= cov/(std_x*std_y)

b=(p_coeff*std_y)/std_x

a=mu_y-(b*mu_x)

ans=a+b*80

print(round(ans,3))


# In[ ]:


# Question 10

import numpy as np

m,n = [int(i) for i in input().strip().split(' ')]

X = []
for i in range(n):
    t=[1.0]
    X.append(t)
    
Y = []
for i in range(n):
    data = input().strip().split(' ')
    t=data[:m]
    for j in range(m):
        X[i].append(t[j])
        
    Y.append(data[m:])
    
    
q = int(input())

X_new = []
for x in range(q):
    X_new.append(input().strip().split(' '))
    X_new[x].insert(0,1.0)
    
X = np.array(X,float)
Y = np.array(Y,float)
X_new = np.array(X_new,float)


#calculate beta
beta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y) )


#print
for i in range(q):
    t=np.dot(X_new[i],beta)
    
    print(round(t[0],2))


# In[ ]:




