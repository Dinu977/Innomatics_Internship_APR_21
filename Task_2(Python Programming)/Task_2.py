#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Question 1
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    k = int(input())
    n = int(input())
    
    ans_lst=[]
    for x1 in range(0,x+1):
        for y1 in range(0,y+1):
            for k1 in range(0,k+1):
                s=x1+y1+k1
                if(s!=n):
                    ans_lst.append([x1,y1,k1])
print(ans_lst)


# In[ ]:


# Question 2

def average(array):
    # your code goes here
    s=set(array)
    return sum(s)/len(s)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# In[ ]:


# Question 3

n,m=input().split()
l=list(map(int,input().split()))
a=set(list(map(int,input().split())))
b=set(list(map(int,input().split())))
h=0
for i in l:
    if i in a:
        h+=1
    if i in b:
        h-=1
print(h)


# In[ ]:


# Question 4

m=int(input())
s1=set(map(int,input().split()))
n=int(input())
s2=set(map(int,input().split()))
l=list(s1^s2)
l.sort()
for i in l:
    print(i)


# In[ ]:


# Question 5

n=int(input())
s=set()
for i in range(0,n):
    s.add(input())
print(len(s))


# In[ ]:


# Question 6 

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    x=max(arr)
    while x in arr:
        arr.remove(x)
    print(max(arr))


# In[ ]:


# Question 7

if __name__ == '__main__':
    l=[]
    s=100
    for _ in range(int(input())):
        name = input()
        score = float(input())
        s=min(s,score)
        l.append([name,score])
    
    sm=100
    for stu in l:
        if stu[1]!=s:
            sm=min(sm,stu[1])
    li=[]
    for nam,scr in l:
        if scr==sm:
            li.append(nam)
    li.sort()
    for n in li:
        print(n)


# In[ ]:


# Question 8
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    x=student_marks[query_name]
    y=sum(x)/len(x)
    print("{:.2f}".format(y))


# In[ ]:


# Question 9

if __name__ == '__main__':
    l=[]
    N = int(input())
    for i in range(N):
        c=input().split()
        if c[0]=='insert':
            p=int(c[1])
            v=int(c[2])
            l.insert(p,v)
        elif c[0]=='print':
            print(l)
        elif c[0]=='remove':
            v=int(c[1])
            if v in l:
                l.remove(v)
        elif c[0]=='append':
            v=int(c[1])
            l.append(v)
        elif c[0]=='sort':
            l.sort()
        elif c[0]=='pop':
            l.pop()
        elif c[0]=='reverse':
            l.reverse()


# In[ ]:


# Question 10

if __name__ == '__main__':
    n = int(input())
    integer_list =list(map(int, input().split()))
    print(hash(tuple(integer_list)))


# In[ ]:


# Question 11

n = int(input())
s = set(map(int, input().split()))
m=int(input())
for i in range(m):
    c=input().split()
    if c[0]=='pop':
            s.pop()
    elif c[0]=='remove':
        v=int(c[1])
        if v in s:
            s.remove(v)
    elif c[0]=='discard':
        v=int(c[1])
        s.discard(v)
print(sum(s))


# In[ ]:


# Question 12

n=int(input())
s1=set(map(int,input().split()))
m=int(input())
s2=set(map(int,input().split()))
print(len(s1|s2))


# In[ ]:


# Question 13

n=int(input())
s1=set(map(int,input().split()))
m=int(input())
s2=set(map(int,input().split()))
print(len(s1&s2))


# In[ ]:


# Question 14

n=int(input())
s1=set(map(int,input().split()))
m=int(input())
s2=set(map(int,input().split()))
print(len(s1-s2))


# In[ ]:


# Question 15

n=int(input())
s1=set(map(int,input().split()))
m=int(input())
s2=set(map(int,input().split()))
print(len(s1^s2))


# In[ ]:


# Question 16

n=int(input())
s=set(map(int,input().split()))
m=int(input())
for i in range(m):
    op=input().split()
    t=set(map(int,input().split(' ')))
    if op[0]=='update':
        s.update(t)
    elif op[0]=='intersection_update':
        s.intersection_update(t)
    elif op[0]=='difference_update':
        s.difference_update(t)
    elif op[0]=='symmetric_difference_update':
        s.symmetric_difference_update(t)
print(sum(s))
        


# In[ ]:


# Question 17

k=int(input())
lst=list(map(int,input().split()))
s={}
for i in lst:
    if i in s:
        s[i]=s[i]+1
    else:
        s[i]=1
for key in s:
    if s[key]==1:
        print(key)
        break


# In[ ]:


# Question 18

t=int(input())
for i in range(t):
    n=int(input())
    s1=set(map(int,input().split()))
    m=int(input())
    s2=set(map(int,input().split()))
    print(s1.issubset(s2))


# In[ ]:


# Question 19

s=set(map(int,input().split()))
n=int(input())
flag=0
for i in range(n):
    t=set(map(int,input().split()))
    if(t.issubset(s)==False or len(t)==len(s)):
        flag=1
        break
if flag==0:
    print("True")
else:
    print("False")

