#!/usr/bin/env python
# coding: utf-8

# # calculator with python

# In[ ]:


def add(x,y):
    return x+y

def substract(x,y):
    return x-y

def multiplication(x,y):
    return x*y

def division(x,y):
    return x/y

print ("select operation")
print ('1.add')
print ('2.substract')
print ('3.multiplication')
print ('4.division')

while True:
    choice = input('enter your choice(1/2/3/4): ')
    
    if choice in ('1','2','3','4'):
        num1 = float(input('enter a number: '))
        num2 = float(input('enter the second number: '))
        
        if choice == '1':
            print(num1,'+',num2,' = ', add(num1,num2))
            
        elif choice == '2':
            print(num1,'-',num2,' = ',substract(num1,num2))
            
        elif choice == '3':
            print(num1,'*',num2," = ",multiplication(num1,num2))
            
        elif choice == '4':
            print(num1,'/',num2,' = ',division(num1,num2))     
        
    else:
        print('invalid input')


# In[ ]:




