import pandas as pd
import os
import numpy as np
import math
from math import isnan
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


################################################## DATA SPLITING AND SAVING AS CSV FILES
data = pd.read_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\github\allergy_data.csv")
print(data['Class'].value_counts())
listhead = list(data)
#print(listhead)
train, test = train_test_split(data,test_size=0.2)
print(train['Class'].value_counts())
print(test['Class'].value_counts())
train=np.array(train)
pd.DataFrame(train).to_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\github\evi_data.csv")
test = np.array(test)
pd.DataFrame(test).to_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\github\que_data.csv")


plot_dist= pd.read_csv(r"E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\github\evi_data.csv")
samples=plot_dist['Class'].value_counts()
samples=dict(samples)
plt.bar(range(len(samples)), list(samples.values()), align='center')
plt.xticks(range(len(samples)), list(samples.keys()), fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Diagnostic Decisions", fontweight='bold')
plt.ylabel("No. of Samples", fontweight='bold')
plt.title("Evidential Data Samples", fontweight='bold')
plt.show()
