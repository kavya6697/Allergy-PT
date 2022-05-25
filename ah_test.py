"""
THIS FILE IS IMPORTANT
INPUT: ALLERGY TEST DATASET AND BPA MATRIX
OUTPUT: PIE GRAPH FOR DIAGNOSIS

"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

######################################################## READINING BPA VALUES FILE 
df = pd.read_csv(r'E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_bpa2_singletrain.csv')
df.set_index('A/C', inplace=True)

#################################################### OBJECTIVE KNOWLEDGE 
gain_RH=[]
gain_UT=[]
gain_OT=[]
gain_RH_UT=[]
gain_RH_O=[]
gain_UT_O=[]
gain_NORMAL=[]

for key, val in dict(df.apply(lambda x: x.argmax(), axis=1)).items():
    locals()[str('gain_'+df.columns[val])].append(key)

"""
gain_rh=(list((df.nlargest(20, ['RH']))['RH'].keys()))
#gain_rh = [sub.rsplit(' ', 1)[0] for sub in gain_rh]
gain_ut=list((df.nlargest(20, ['UT']))['UT'].keys())
#gain_ut = [sub.rsplit(' ', 1)[0] for sub in gain_ut]
gain_ot=list((df.nlargest(20, ['OT']))['OT'].keys())
#gain_ot = [sub.rsplit(' ', 1)[0] for sub in gain_ot]
gain_rh_ut=list((df.nlargest(20, ['RH_UT']))['RH_UT'].keys())
#gain_rh_ut = [sub.rsplit(' ', 1)[0] for sub in gain_rh_ut]
gain_rh_o=list((df.nlargest(20, ['RH_O']))['RH_O'].keys())
#gain_rh_o = [sub.rsplit(' ', 1)[0] for sub in gain_rh_o]
gain_ut_o=list((df.nlargest(20, ['UT_O']))['UT_O'].keys())
#gain_ut_o = [sub.rsplit(' ', 1)[0] for sub in gain_ut_o]
gain_normal=list((df.nlargest(20, ['NORMAL']))['NORMAL'].keys())
#gain_normal = [sub.rsplit(' ', 1)[0] for sub in gain_normal]

ref_obj={'RH':gain_rh[::-1],
         'UT':gain_ut[::-1],
          'OT':gain_ot[::-1],
          'RH_UT':gain_rh_ut[::-1],
          'RH_O':gain_rh_o[::-1],
          'UT_O':gain_ut_o[::-1]}        
ref_obj={'RH':gain_rh,
         'UT':gain_ut,
          'OT':gain_ot,
          'RH_UT':gain_rh_ut,
          'RH_O':gain_rh_o,
          'UT_O':gain_ut_o}
#print(ref_obj)


################################### MANUALLY ENTERING SUBJECTIVE REFERENCE

['housedust', 'cottondust', 'aspergilus', 'pollen', 'parthenium', 'cockroach', 'cat dander', 
'dos fur', 'Milk(P)', 'Milk( C )', 'curd', 'coffee', 'tea', 'beef', 'chicken', 'mutton', 
'egg', 'fish1', 'fish2', 'crab', 'prawns', 'shark', 'avaraikai', 'banana', 'beans', 
'beet root', 'brinjal', 'cabbage', 'capsicum', 'chillie', 'cauliflower', 'carrot', 
'chow-chow', 'corn', 'cucumber', 'drumstick', 'greens', 'gourds', 'kovaikai', 'kothavarai',
 'L finger', 'malli', 'mango', 'mushroom', 'nuckol', 'onion', 'peas', 'potroot', 'paneer', 
 'potato', 'pumkin', 'pudina', 'radish', 'tomato', 'tondaikai', 'vazpoo/thandu', 'yams',
 'gram', 'channa', 'dhal', 'maida', 'oats', 'ragi', 'rice', 'wheat', 'coconut', 'oil', 
 'garlic', 'ginger', 'pepper', 'tamarind', 'aginomoto', 'spices', 'coco', 'drink 1', 
 'drink2', 'fruit1', 'fruit2', 'lime', 'nuts', 'running nose', 'sneeze', 'cough', 
 'wheeze/blocks', 'headache', 'itching', 'swelling', 'red rashes', 'age', 'gender', 
 'f_history']

"""

ref={'RH':['Milk(P)','L finger','pollen','housedust','cough','sneeze'],
        'UT':['wheat','prawns','parthenium','aspergilus','redrashes','itching'],
        'OT':['mushroom','brinjal','chicken','parthenium','cockroach','swelling'],
        'RH_UT':['maida','prawns','cockroach','cottondust','itching','sneeze'],
        'RH_O':['greens','Milk(P)','cottondust','housedust','cough','wheezeBlocks'],
        'UT_O':['yams','egg','cockroach','parthenium','itching','redrashes'],
        'NORMAL':['runningnose','headache','f_history']
       }  

       
ref_mod=['housedust HR','housedust MR','housedust LR',
         'cottondust HR','cottondust MR','cottondust LR',
         'aspergilus HR','aspergilus MR','aspergilus LR',
         'pollen HR','pollen MR','pollen LR',
         'parthenium HR','parthenium MR','parthenium LR',
         'cockroach HR','cockroach MR','cockroach LR',
         'dos fur HR','dos fur MR','dos fur LR',
         'Milk(P) HR','Milk(P) MR','Milk(P) LR',
         'curd HR','curd MR','curd LR',
         'beef HR','beef MR','beef LR',
         #'egg HR','egg MR','egg LR',
         #'chicken HR','chicken MR','chicken LR',
         #'fish1 HR','fish MR','fish LR',
         'crab HR','crab MR','crab LR',
         'prawns HR','prawns MR','prawns LR',
         #'brinjal HR','brinjal MR','brinjal LR',
         #'cucumber HR','cucumber MR','cucumber LR',
         #'greens HR','greens MR','greens LR',
         #'mango HR','mango MR','mango LR',
         #'mushroom HR','mushroom MR','mushroom LR',
         #'radish HR','radish MR','radish LR',
         #'tomato HR','tomato MR','tomato LR',
         #'yams HR','yams MR','yams LR',
         #'dhal HR','dhal MR','dhal LR',
         #'maida HR','maida MR','maida LR',
         #'wheat HR','wheat MR','wheat LR', 
         #'tamarind HR','tamarind MR','tamarind LR',
         #'aginomoto HR','aginomoto MR','aginomoto LR',
         'runningnose YES','sneeze YES','cough YES','wheeze/blocks YES',
         'headache NO','itching YES','swelling NO','redrashes YES','f_history YES']


####################################################### READING TEST DATA 
test = pd.read_csv(r'E:\Ph.D\3-JOURNALS\A3-ALLERGY PT\AH CODE\ah_data2_test.csv')
x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]
print(test['Class'].value_counts())
actual_classes= list(y_test)
#print(actual_classes)
#print(test['Class'].value_counts())
list_col = list(x_test.columns)
#print(list_col)


################################################# EXTRACTING BPA VALUES FOR TEST INSTANCES 
test_bpa_full = []
for row in x_test.iterrows():
    c = 0
    test_bpa_single = []
    for a in row[1]:
        if(c == len(x_test.columns)):
            break
        s = str(list_col[c])+" "+str(a)
        if(s in df.index and s in ref_mod):
            test_bpa_single.append(df.loc[s])
        else:
            test_bpa_single.append(pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=['RH', 'OT', 'NORMAL', 'RH_UT', 'UT', 'UT_O', 'RH_O'], name=s))
        c += 1
    test_bpa_full.append(test_bpa_single)
#print(len(test_bpa_full))



#################################### DECISION PROBABILITIES WITH SUBJECTIVE WEIGHTING 
classes=['Rhinitis','Urticaria','Others', 'Rhinitis and Urticaria', 'Rhinitis and Others', 'Urticaria and Others']
conf_class=['Rhinitis','Urticaria','Others', 'Rhinitis and Urticaria', 'Rhinitis and Others', 'Urticaria and Others','Normal'] 
pred_classes=[]
match_count_exact=0
match_count_partial=0
match_count_sum_partial=0
for i in range(len(test_bpa_full)):
    #print("test instance --------",i)
    #print(actual_classes[i])
    dp_rh=0
    dp_ut=0
    dp_ot=0
    dp_rh_ut=0
    dp_rh_o=0
    dp_ut_o=0
    dp_normal=0
    for j in range(len(test_bpa_full[i])):
        if(list_col[j] in ref['RH']):
            wbpa=0
            #print(list_col[j])
            #print(test_bpa_full[i][j]['RH'])         
            #print(ref_c1['RH'].index(list_col[j])+1)
            wbpa=(test_bpa_full[i][j]['RH'])*(ref['RH'].index(list_col[j])+1)
            #print(wbpa)
            dp_rh=dp_rh+wbpa
        else:
            dp_rh=dp_rh+0
        
        if(list_col[j] in ref['UT']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['UT'])*(ref['UT'].index(list_col[j])+1)
            dp_ut=dp_ut+wbpa
        else:
            dp_ut=dp_ut+0
        
        if(list_col[j] in ref['OT']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['OT'])*(ref['OT'].index(list_col[j])+1)
            dp_ot=dp_ot+wbpa
        else:
            dp_ot=dp_ot+0
        
        if(list_col[j] in ref['RH_UT']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['RH_UT'])*(ref['RH_UT'].index(list_col[j])+1)
            dp_rh_ut=dp_rh_ut+wbpa
        else:
            dp_rh_ut=dp_rh_ut+0
        
        if(list_col[j] in ref['RH_O']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['RH_O'])*(ref['RH_O'].index(list_col[j])+1)
            dp_rh_o=dp_rh_o+wbpa
        else:
            dp_rh_o=dp_rh_o+0
            
        if(list_col[j] in ref['UT_O']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['UT_O'])*(ref['UT_O'].index(list_col[j])+1)
            dp_ut_o=dp_ut_o+wbpa
        else:
            dp_ut_o=dp_ut_o+0
        if(list_col[j] in ref['NORMAL']):
            wbpa=0
            wbpa=(test_bpa_full[i][j]['NORMAL'])*1
            dp_normal=dp_normal+wbpa
            
    #print(dp_rh,dp_ut,dp_ot,dp_rh_ut,dp_rh_o,dp_ut_o)
    dp_rh_ws=dp_rh+(dp_rh_ut/2)+(dp_rh_o/2)
    dp_ut_ws=dp_ut+(dp_rh_ut/2)+(dp_ut_o/2)
    dp_ot_ws=dp_ot+(dp_rh_o/2)+(dp_ut_o/2)
    dp_list = {'RH':dp_rh,'UT':dp_ut,'OT':dp_ot,'RH_UT':dp_rh_ut,'RH_O':dp_rh_o,'UT_O':dp_ut_o,'NORMAL':dp_normal}
    dp_list_ws = {'RH':dp_rh_ws,'UT':dp_ut_ws,'OT':dp_ot_ws,'NORMAL':dp_normal}
    #dp_list = [dp_rh,dp_ut,dp_ot,dp_rh_ut,dp_rh_o,dp_ut_o]
    #print(dp_list)
    Keymax = max(zip(dp_list.values(), dp_list.keys()))[1]
    Keymax_ws = max(zip(dp_list_ws.values(), dp_list_ws.keys()))[1]
    #print(Keymax)
    if (Keymax == actual_classes[i]):
        match_count_exact=match_count_exact+1
    if (Keymax in actual_classes[i] or actual_classes[i] in Keymax):
        match_count_partial=match_count_partial+1
    if (Keymax_ws in actual_classes[i] or actual_classes[i] in Keymax_ws):
        match_count_sum_partial=match_count_sum_partial+1
    pred_classes.append(Keymax)
    #fig = plt.figure(figsize =(10, 7))
    #plt.pie(dp_list, labels = classes, autopct='%1.0f%%')
    #plt.show()
print(pred_classes)
print('accuracy exact:',match_count_exact/len(actual_classes))
print('accuracy partial:',match_count_partial/len(actual_classes))
print('accuracy weighted:',match_count_sum_partial/len(actual_classes))
cfm=confusion_matrix(actual_classes,pred_classes)
df_cfm = pd.DataFrame(cfm, index = conf_class, columns = conf_class)
#plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
cfm_plot.figure.savefig("cfm.png")