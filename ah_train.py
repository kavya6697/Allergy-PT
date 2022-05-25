"""
THIS FILE IS IMPORTANT
INPUT: ALLERGY FULL DATASET WITH 905 SAMPLES
OUTPUT: ALLERGY TEST DATASET, UNBALANCED TRAIN DATASET AND BPA MATRIX
MODIFICATIONS: MAKE SURE ALLERGY TEST DATASET IS HAVING ATTRIBUTE NAMES, 
               MAKE SURE THAT BPA FILE IS HAVING A/C IN TOP CORNER. 
"""

import pandas as pd
import os
import numpy as np
import math
from math import isnan
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


################################################## DATA SPLITING AND SAVING AS CSV FILES
data = pd.read_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\allergy_data2.csv")
print(data['Class'].value_counts())
listhead = list(data)
#print(listhead)
train, test = train_test_split(data,test_size=0.2)
print(train['Class'].value_counts())
print(test['Class'].value_counts())
train=np.array(train)
pd.DataFrame(train).to_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data2_ubtrain.csv")
test = np.array(test)
pd.DataFrame(test).to_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data2_test.csv")


"""
APPLIED SAMPLING IN RSTUDIO AND SAVED THREE CSV FILES FOR IDEAL, SINGLE, ALL
"""

"""
################################CODE TO GET DATA DISTRIBUTION PLOT 
plot_dist= pd.read_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data_fulltrain.csv")
samples=plot_dist['Class'].value_counts()
samples=dict(samples)
plt.bar(range(len(samples)), list(samples.values()), align='center')
plt.xticks(range(len(samples)), list(samples.keys()), fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Diagnostic Decisions", fontweight='bold')
plt.ylabel("No. of Samples", fontweight='bold')
plt.title("Fully Sampled data", fontweight='bold')
plt.show()
"""


##################################### EXTRACTING PEICES OF EVIDENCE AND DECISION ALTERNATIVES 
evi_data= pd.read_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data1_singletrain.csv")
listhead = list(evi_data)
evi_data=np.array(evi_data)
pcs=[]
for i in range(len(listhead)-1):
    pieces = set()
    for j in range(len(evi_data)):
        poa = str(listhead[i]) + " " + str(evi_data[j][i])
        pieces.add(poa)
    pcs+=pieces
cls=set()
for i in range(len(evi_data)):
    cls.add(evi_data[i][-1])
cls=list(cls)

##################################### STRUCTURE OF BPA MATRIX 
ans = [[0 for i in range(len(cls)+1)] for j in range(len(pcs)+1)] #0 matrix with one 0-row and one 0-column
ans[0][0]=0
settolist=set()
temp=-1
rset=set()
rrset=set()
ff=1
laterlist=[]
for i in range(len(evi_data[0])-1): #list1 is the entire dataset without row 0 and col 0
    temp+=1
    rset.clear()
    settolist.clear()
    for j in range(len(evi_data)):
        rset.add(evi_data[j][i])
        rrset.add(evi_data[j][i])
    settolist=list(rset)
    a=[]
    Na = False
    for i in range(len(settolist)):
        if settolist[i]==settolist[i]:
            a.append(settolist[i])
        elif settolist[i]!=settolist[i] and Na == False:
            a.append(settolist[i])
            Na = True    
    laterlist.append(a)
    m=0
    for k in range(ff,len(a)+ff):
        ans[k][0]=str(listhead[temp])+" "+ str(a[m])
        m+=1
    ff+=len(a)

################################## COMPUTING BPA VALUES
rel=ans
m=0
for i in range(1,len(ans[0])):
    ans[0][i]=cls[m]
    m+=1
conf=ans
list1=evi_data
past=[]
rost=[]
def funcConf(row,col,passing):
    m=0
    n=0
    for i in range(len(list1)):
        if str(list1[i][passing])==str(row) and str(list1[i][len(list1[0])-1])==str(col):
            m=m+1        
        if str(list1[i][passing])==str(row):
            n=n+1
    if m==0:
        return 0
    return m/n
kewl=-1
passing=0
for i in range(1,len(cls)+1): #len(ans)
    col=cls[i-1]
    kewl+=1
    forkewl=0
    sus=0
    passing=0
    for j in range(1,len(conf)): #len(ans[0])
        row=str(conf[j][0]).rsplit(' ',1)[-1]
        if forkewl==len(laterlist[sus]):
            forkewl=1
            passing+=1
            sus+=1
        else:
            forkewl+=1
        conf[j][i]=funcConf(row,col,passing)  

################################# SAVING BPA VALUES IN EXCEL 
numparr=np.array(conf)
pd.DataFrame(numparr).to_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_bpa1_singletrain.csv")