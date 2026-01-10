#Question 1
#Consider the given list as [2, 7, 4, 1, 3, 6]. Write a program to count pairs of elements with sum equal to 10.
"""
list = [2, 7, 4, 1, 3, 6]
count = 0
for i in range(len(list)):
    for j in range(i+1, len(list)):
        if (list[i] + list[j]) == 10:
            count += 1
print(f"Number of pairs with sum 10 are {count}")
"""

#Question 2
#Write a program that takes a list of real numbers as input and returns the range (difference between minimum and maximum) of the list. Check for list being less than 3 elements in which case return an error message (Ex: "Range determination not possible"). Given a list [5,3,8,1,0,4], the range is 8 (8-0).
"""
x = int(input("Enter the number of elements in the list:"))
if x<3:
    print("Range determination not possible")
else:
    list = []
    for i in range(x):
        y = int(input("Enter the number to add in list:"))
        list.append(y)
    highest=list[0]
    least=list[0]
    for i in range(1, len(list)):
        if highest<list[i]:
            highest = list[i]
        if least>list[i]:
            least = list[i]
    range = highest-least
    print(f"Range is: {range}")
"""

#Question 3
#Write a program that accepts a square matrix A and a positive integer m as arguments and returns A^m.
"""
n=int(input("Enter order of matrix:"))
A=[]
print("Enter matrix elements:")
for i in range(n):
    row=[]
    for j in range(n):
        row.append(int(input()))
    A.append(row)
m=int(input("Enter power m:"))
result=A
for k in range(m-1):
    temp=[]
    for i in range(n):
        row=[]
        for j in range(n):
            sum=0
            for x in range(n):
                sum+=result[i][x]*A[x][j]
            row.append(sum)
        temp.append(row)
    result=temp
print("Resultant Matrix A^",m,":")
for i in result:
    print(i)
"""

#Question 4
#Write a program to count the highest occurring character & its occurrence count in an inpu string. Consider only alphabets. Ex: for "hippopotamus" as input string, the maximally occurring character is 'p' & occurrence count is 3.
"""
word = input("Enter a string:")
dict = {}
set = set()
for i in range(len(word)):
    if word[i].isalpha():
        if word[i] in set:
            dict[word[i]]+=1
        else:
            dict[word[i]] = 1
            set.add(word[i])
highest_count = 0
for key in dict:
    if dict[key]>highest_count:
        highest_count = dict[key]
        max_key = key
print(f"Maximally occuring character is {max_key}")
print(f"Occurence count is {highest_count}")
"""

#Question 5
#Generate a list of 25 random numbers between 1 and 10. Find the mean, median and mode for these numbers.
"""
import random
list=[]
for i in range(25):
    list.append(random.randint(1,10))
print("Generated list:",list)
sum=0
for i in list:
    sum+=i
print("The mean is:",sum/25)
list.sort()
if 25%2==0:
    median=(list[12]+list[13])/2
else:
    median=list[12]
print("The median is:",median)
dict={}
for i in list:
    if i in dict:
        dict[i]+=1
    else:
        dict[i]=1
highest_count=0
for key in dict:
    if dict[key]>highest_count:
        highest_count=dict[key]
        mode=key
print("Mode is:",mode)
print("Mode count is:",highest_count)
"""
