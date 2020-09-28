import pandas as pd
import numpy as np

df = pd.read_csv("IDs.csv")
print(df)

for i in pd.read_csv("IDs.csv"):
    print(i)

SMIs = []
for i in df:
    print(type(i))
    print(i)
    b = int(float(i))     
    f = open("0/" + str(b) + ".smi", 'r')
    a = f.readline()
    SMIs.append(a)    

for i in df['1.719000000000000000e+03']:
    print(i) 
    b = int(float(i))     
    f = open("0/" + str(b) + ".smi", 'r')
    a = f.readline()
    SMIs.append(a) 

print(SMIs)
np.savetxt('SMIsAvg1.csv', SMIs, fmt='%s')
