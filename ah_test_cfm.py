import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn

######################################################## READINING BPA VALUES FILE 
df = pd.read_csv(r'E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data2_cfm.csv')
conf_class=['RH','UT','OT', 'RH_UT', 'RH_O', 'UT_O','NORMAL'] 
actual_classes= list(df['Actual'])
pred_classes= list(df['predict'])
cfm=confusion_matrix(actual_classes,pred_classes,conf_class)
cmd_obj = ConfusionMatrixDisplay(cfm, display_labels=conf_class)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Sklearn Confusion Matrix with labels!!', 
                xlabel='Predicted Labels', 
                ylabel='Actual Labels')
plt.show()

