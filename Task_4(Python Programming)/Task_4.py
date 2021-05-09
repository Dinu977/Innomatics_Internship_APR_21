#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Question 1

def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# In[ ]:


# Question 2

def split_and_join(line):
    # write your code here
    a=line.split()
    return '-'.join(a)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# In[ ]:


# Question 3

def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# In[ ]:


# Question 4

def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# In[ ]:


# Question 5

def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)-len(sub_string)+1):
        if string[i:i+len(sub_string)] == sub_string:
            count+=1
    return count


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


# In[ ]:


# Question 6

if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))


# In[ ]:


# Question 7

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# In[ ]:


# Question 8

import textwrap

def wrap(string, max_width):
    return  "\n".join([string[i:i+max_width] for i in range(0, len(string), max_width)])

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# In[ ]:


# Question 9

lst=list(map(int,input().split()))

st=".|."
t=".|."
for i in range(lst[0]//2):
print(st.center(lst[1],'-'))
st+=t+t
    
print("WELCOME".center(lst[1],'-'))

st=st[6:]
for i in range(lst[0]//2):
print(st.center(lst[1],'-'))
st=st[6:]


# In[ ]:


# Question 10

def print_formatted(number):
    # your code goes here
            n=bin(number).replace("0b","")
            t=len(n)
    
            for i in range(1,number+1):
                print(str(i).rjust(t),end=' ')
                print(oct(i).replace("0o","").rjust(t),end=' ')
                st=hex(i).replace("0x","")
                st=st.replace("a","A")
                st=st.replace("b","B")
                st=st.replace("c","C")
                st=st.replace("d","D")
                st=st.replace("e","E")
                st=st.replace("f","F")
                print(st.rjust(t),end=' ')
                print(bin(i).replace("0b","").rjust(t),end=' ')
                print()
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# In[ ]:


# Question 11

import string
def print_rangoli(n):
    # your code goes here
    alpha = string.ascii_lowercase
    L = []
    for i in range(n):
        s = "-".join(alpha[i:n])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
    print('\n'.join(L[:0:-1]+L))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


# In[ ]:


# Question 12

import string
def solve(s):
    return string.capwords(s, ' ')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


# In[ ]:


# Question 13

def minion_game(string):
    # your code goes here
    vowel = 'AEIOU'
    stuart, kevin = 0, 0
    for i in range(0, len(string)):
        if string[i] not in vowel:
            stuart+=len(string)-i
        else:
            kevin+=len(string)-i
    
    if stuart>kevin:
        print('Stuart', stuart)
    elif kevin>stuart:
        print('Kevin', kevin)
    else:
        print('Draw')    

if __name__ == '__main__':
    s = input()
    minion_game(s)


# In[ ]:


# Question 14

def merge_the_tools(string, k):
    # your code goes here
            n=len(string)
    
            for i in range(0,n,k):
                st=string[i:i+k]
                s=set()
                ans=""
                for j in range(k):
                        if st[j] in s:
                            continue
                        else:
                            ans+=st[j]
                            s.add(st[j])
        
                print(ans)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

