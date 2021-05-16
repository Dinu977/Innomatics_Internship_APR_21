#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Question 1

import numpy

def arrays(arr):
    my_arr = numpy.array(arr, float)
    return my_arr[::-1]

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# In[5]:


# Question 2: Shape and Reshape.

import numpy as np

list1 = list(map(int, input().split()))
arr = np.array(list1)
print(arr.reshape(3, 3))


# In[14]:


# Question 3

import numpy as np

n, m = map(int, input().split())
arr = []
for i in range(n):
    x = []
    x = list(map(int, input().split()))
    arr.append(x)
    
my_arr = np.array(arr)
print(my_arr.T)
print(my_arr.flatten())


# In[33]:


# Question 4 

import numpy as np

n, m, p = map(int, input().split())

list_N, list_M = [], []

for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    list_N.append(x)
array_N = np.array(list_N)

for j in range(0, m):
    x = []
    x = list(map(int, input().split()))
    list_M.append(x)
array_M = np.array(list_M)

print(np.concatenate((array_N, array_M), axis = 0))


# In[51]:


# Question 5 

import numpy as np

x = tuple(map(int, input().split()))
print(np.zeros(x, dtype = np.int))
print(np.ones(x, dtype = np.int))


# In[4]:


# Question 6 

import numpy as np

n, m = map(int, input().split())
np.set_printoptions(sign=' ')
print(np.eye(n, m, 0))


# In[9]:


# Question 7

import numpy as np

n, m = map(int, input().split())

arr1, arr2 = [], []
for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    arr1.append(x)
ar1 = np.array(arr1)

for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    arr2.append(x)
ar2 = np.array(arr2)

print(ar1+ar2)
print(ar1-ar2)
print(ar1*ar2)
print(ar1//ar2)
print(ar1%ar2)
print(ar1**ar2)


# In[13]:


# Question 8 

import numpy as np

np.set_printoptions(legacy = '1.13')
arr1 = np.array(list(map(float, input().split())))

print(np.floor(arr1))
print(np.ceil(arr1))
print(np.rint(arr1))


# In[17]:


# Question 9

import numpy as np

n, m = map(int, input().split())
list1 = []

for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    list1.append(x)

arr1 = np.array(list1)
sum_along_0_axis = np.sum(arr1, axis = 0)
print(np.prod(sum_along_0_axis))


# In[20]:


# Question 10 : Min and Max

import numpy as np

n, m = map(int, input().split())
list1 = []
for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    list1.append(x)
arr = np.array(list1)
min_arr = arr.min(axis = 1)
print(min_arr.max())


# In[30]:


# Question 11 

import numpy as np

n, m = map(int, input().split())

list1 = []

for i in range(0, n):
    x = []
    x = list(map(int, input().split()))
    list1.append(x)
 
arr = np.array(list1)
print(np.mean(arr, axis = 1))
print(np.var(arr, axis = 0))
print("{:.11f}".format(np.std(arr, axis = None)))


# In[32]:


# Question 12

import numpy as np

n = int(input())
a = []
for i in range(0, n):
    x = list(map(int, input().split()))
    a.append(x)
A = np.array(a)

b = []
for i in range(0, n):
    x = list(map(int, input().split()))
    b.append(x)
B = np.array(b)

print(np.dot(A, B))


# In[33]:


# Question 13

import numpy as np

A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))
print(np.inner(A, B))
print(np.outer(A, B))


# In[35]:


# Question 14

import numpy as np
P = np.array(list(map(float, input().split())))
x = int(input())
print(np.polyval(P, x))


# In[43]:


# Question 15 

import numpy as np

n = int(input())
a = []
for i in range(0, n):
    x = list(map(float, input().split()))
    a.append(x)
    
arr = np.array(a)
x = "{:.2f}".format(np.linalg.det(arr))

if float(x) == round(float(x)):
    print(float(x))
else:
    print(x)


# In[ ]:




