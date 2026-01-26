import monkdata as m
from dtree import *
import drawtree_qt5 as dt


#------------- Assignment 1 -------------------------
m1_entropy = entropy(m.monk1)
m2_entropy = entropy(m.monk2)
m3_entropy = entropy(m.monk3)
print(f"m1_entropy: {m1_entropy}\nm2_entropy: {m2_entropy}\nm3_entropy: {m3_entropy}\n")

#------------- Assignment 3 och 4 -------------------------
datasets = [m.monk1,m.monk2,m.monk3]
attributes = [0,1,2,3,4,5] #motsvarar a1,a2,...,a6
for dataset in datasets:
    for attribute in attributes:
        print(f"Average gain:{round(averageGain(dataset,m.attributes[attribute]),5)} for dataset {datasets.index(dataset)} and attribute {attribute}\n")
#Bästa attributen att splita monk1 a5 med gain på .287
#------------- Assignment 5-------------------------

t=buildTree(m.monk1, m.attributes)
dt.drawTree(t)
print(round(check(t, m.monk1),5))
print(f"{round(check(t, m.monk1test),5)}\n")

t2=buildTree(m.monk2, m.attributes)
print(round(check(t, m.monk2),5))
print(f"{round(check(t, m.monk2test),5)}\n")

t3=buildTree(m.monk3, m.attributes)
print(round(check(t, m.monk3),5))
print(f"{round(check(t, m.monk3test),5)}\n")