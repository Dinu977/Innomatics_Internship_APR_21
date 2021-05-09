#!/usr/bin/env python
# coding: utf-8

# In[40]:


# Question 1 

T = int(input())
while T>0:
    s = input()
    number = s.split('.')
    flag = 0

    for i in range(1, len(s)):
        if s[i] == '+' or s[i] == '-':
            print('False')
            flag = 1
            break
    if flag == 0:
        if len(number) == 2:
            if len(number[1]) == 0:
                print('False')
            elif any(i.isalpha() for i in s):
                print('False')
            else:
                print('True')
        else:
            print('False')
    T-=1


# In[47]:


# Question 2 

import re
s = input()
y = re.split("[.,]", s)
for i in y:
    print(i)


# In[3]:


# Question 3 

import re
case = re.search(r'([a-zA-Z0-9])\1+', input())
print(case.group(1) if case else -1)


# In[18]:


# Question 4 

s = input()
v = 'aeiouAEIOU'
c = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
flag = 0
for i in range(0, len(s)-1):
    if s[i] in c:
        j = i
        count = 0
        for j in range(i+1, len(s)):
            if s[j] in v:
                count+=1
            else:
                break
        if count>=2 and s[j] in c:
            print(s[i+1:j])
            flag = 1
    else:
        pass
if flag == 0:
    print(-1)


# In[36]:


# Question 5 

s = input()
sub = input()
flag = 0
for i in range(0, len(s)-len(sub)+1):
    if s[i:i+len(sub)] == sub:
        print(f'({i}, {i+len(sub)-1})')
        flag = 1
    else:
        pass
if flag == 0:
    print('(-1, -1)')


# In[18]:


# Question 6 

import re
print('\n'.join(re.sub(R'(?<= )(&&|\|\|)(?= )', lambda x: 'and' if x.group()=='&&' else 'or', input()) for _ in range(int(input().strip()))))


# In[5]:


# Question 7 

import re

thousand = 'M{0,3}'
hundred = '(C[MD]|D?C{0,3})'
ten = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
print (bool(re.match(thousand + hundred+ten+digit +'$', input())))


# In[31]:


# Question 8 

def start_with(number):
    if 7<=int(number[0])<=9:
        return True
    return False


T = int(input())
while T>0:
    number = input()
    if number.isdigit() and start_with(number) and len(number) == 10:
        print('YES')
    else:
        print('NO')
    
    T-=1


# In[32]:


# Question 9 

import email.utils

def is_valid(address):
    if ('@' in address) and ('.' in address):
        username = address[:address.index('@')]
        first = username[0].isalpha()
        username = username.replace("_","")
        username = username.replace("-","")
        username = username.replace(".","")
        address = address[address.index('@')+1:]
        domain = address[:address.index('.')]
        extension = address[address.index('.')+1:]
        return first and username.isalnum() and domain.isalpha() and extension.isalpha() and len(extension) < 4
        
    else:
        return False

n = int(input())
for i in range(n):
    E_address = input()
    if is_valid(email.utils.parseaddr(E_address)[1]):
        print(E_address)


# In[8]:


# Question 10

import re

if __name__ == "__main__":
    reg = re.compile(r"(:|,| +)(#[abcdefABCDEF1234567890]{3}|#[abcdefABCDEF1234567890]{6})\b")

    n = int(input())
    
    for i in range(n):
        line  = input()
        items = reg.findall(line)

        if items:
            for item in items:    
                print( item[1] )


# In[9]:


# Question 11 : 

from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print ('Start :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
    def handle_endtag(self, tag):
        print ('End   :',tag)
        
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))


# In[10]:


# Question 12

from html.parser import HTMLParser
class CustomHTMLParser(HTMLParser):
    def handle_comment(self, data):
        number_of_line = len(data.split('\n'))
        if number_of_line>1:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

parser = CustomHTMLParser()

n = int(input())

html_string = ''
for i in range(n):
    html_string += input().rstrip()+'\n'
    
parser.feed(html_string)
parser.close()


# In[11]:


# Question 13 

from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]
        
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# In[7]:


# Question 14

def upperCase(uid):
    count = 0
    for i in range(0, len(uid)):
        if uid[i].isupper():
            count+=1
    if count>=2:
        return 1
    else:
        return 0

def is_digit(uid):
    count = 0
    for i in range(0, len(uid)):
        if uid[i].isdigit():
            count+=1
    if count>=3:
        return 1
    return 0
    
def repeatation(uid):
    for i in range(0, len(uid)):
        if uid.count(uid[i])>1:
            return 0
    return 1  
    
    
T = int(input())
while T>0:
    uid = input()
    if len(uid)==10 and upperCase(uid) and is_digit(uid) and uid.isalnum() and repeatation(uid):
        print('Valid')
    else:
        print('Invalid')
    
    T-=1


# In[27]:


# Question 15 

def start_with(card):
    if 3<int(card[0])<7:
        return True
    return False 

def repeated_digits(card):
    for i in range(0, len(card)-4+1):
        if card[i] == card[i+1] == card[i+2] == card[i+3]:
            return False
    return True

def each_block_frequency(card):
    x = card.split('-')
    for i in x:
        if len(i)!=4:
            return False
    return True
    
    
T = int(input())
while T>0: 
    card = input()
    if len(card) == 16:
        if start_with(card) and card.isdigit() and repeated_digits(card):
            print('Valid')
        else:
            print('Invalid')
    elif len(card) == 19:
        if card.count('-') == 3:
            if each_block_frequency(card):
                card = card.replace('-', '')
                if start_with(card) and card.isdigit() and repeated_digits(card):
                    print('Valid')
                else:
                    print('Invalid')
            else:
                print('Invalid')          
        else:
            print('Invalid')
    else:
        print('Invalid')
    T-=1


# In[15]:


# Question 16 

import re
s=input()
print (bool(re.match(r'^[1-9][\d]{5}$',s) and len(re.findall(r'(\d)(?=\d\1)',s))<2 ))


# In[16]:


# Question 17 

import re

n, m = map(int, input().split())
a, b = [], ""
for _ in range(n):
    a.append(input())

for z in zip(*a):
    b += "".join(z)

print(re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", b))


# In[ ]:




