#===================================================
#  JH - September 2024  
#  Please note that this is not an optimal implementation
#  The focus is on "understand" the algorithms
#===================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy import stats
import sys

show=1
#---------------------------------------- 1. Read Data
df  = pd.read_csv('P1_cardiacRisk.csv')
D   = df.values
M   = D.shape[1]-1  # number of characteritics
X   = D[:,0:M]      # inputs - risk factors
T   = D[:,M]        # Target - {0,1}={No risk, Risk}
N   = D.shape[0]    # number of patients


#---------------------------------------- 2. Missing values
idM = np.where(X[:,1]<0)            # missing data
idV = np.where(X[:,1]>0)            # valid data
ageMean=np.mean( X[idV,1]).round()  # mean value of valid age
X[idM,1]=ageMean                    # replace missing values by mean value


#---------------------------------------- 3. Analyse data
print(round(df.describe(), 2))

id0= np.where( T==0)                # NO risk patiets
id1= np.where( T==1)                # Risk patients
plt.subplot(121)
plt.plot( X[id0,1], X[id0,2],'b+', label='N')
plt.plot( X[id1,1], X[id1,2],'ro', label='E')

plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.subplot(122)
plt.plot( X[id0,1], X[id0,3],'b+')
plt.plot( X[id1,1], X[id1,3],'ro')
plt.xlabel('Age')
plt.ylabel('SBP')
plt.show()


if show:
    plt.xlabel('Value')
    plt.ylabel('Variable')
    plt.title('Distribution of Variables')
    plt.grid(alpha=0.3)
    plt.show()

#---------------------------------------- 3. Classification
id0= np.where( T==0)        # NO risk patients
id1= np.where( T==1)        # Risk patients

#------------------------------------- Virtual patients / Mean values of characteristics
C0 = np.zeros([1,M])
C1 = np.zeros([1,M])
for i in range(M):
    C0[0,i]= np.mean(X[id0,i])
    C1[0,i]= np.mean(X[id1,i])
    
#------------------------------------- Model Output : based on similairy with virtual patients
Y = np.zeros(N)                     # output of the classification model
for i in range(N):
    patient=X[i,0:M]
    d0 = np.linalg.norm( patient - C0)
    d1 = np.linalg.norm( patient - C1)
    if d0<d1:
        Y[i]=0
    else:
        Y[i]=1


#------------------------------------- Performance: SE/SP/F1score
TP=0
FP=0
TN=0
FN=0        
for i in range(0,N):
    if T[i]==Y[i] and T[i]==1:    #T=Y=1
        TP=TP+1
    if T[i]==Y[i] and T[i]==0:    #T=Y=0
        TN=TN+1
    if T[i]!=Y[i] and Y[i]==1:    #T=0, Y=1
        FP=FP+1
    if T[i]!=Y[i] and Y[i]==0:    #T=1, Y=0
        FN=FN+1
        
SE=TP/(TP+FN)
SP=TN/(TN+FP)
PP=TP/(TP+FP)
F1=2*SE*PP/(SE+PP)
print("\n:::::::::::  FRONTERIA DECISAO ------------------")
print(" SE - sensibilidade  >", round(SE,3))
print(" SP - Especificidade >", round(SP,3))
print(" F1score             >", round(F1,3))




