# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv", header=None)

w15=pd.Series([])
w25=pd.Series([])
w35=pd.Series([])
w45=pd.Series([])
w16=pd.Series([])
w26=pd.Series([])
w36=pd.Series([])
w46=pd.Series([])
b5=pd.Series([])
b6=pd.Series([])
w57=pd.Series([])
w67=pd.Series([])
w58=pd.Series([])
w68=pd.Series([])
w59=pd.Series([])
w69=pd.Series([])
b7=pd.Series([])
b8=pd.Series([])
b9=pd.Series([])
#inisialisasi
w15[0]=0.14
w25[0]=0.39
w35[0]=0.55
w45[0]=0.72
w16[0]=0.11
w26[0]=0.34
w36[0]=0.31
w46[0]=0.65
b5[0]=0.70
b6[0]=0.20
w57[0]=0.51
w67[0]=0.44
w58[0]=0.29
w68[0]=0.09
w59[0]=0.87
w69[0]=0.79
b7[0]=0.05
b8[0]=0.91
b9[0]=0.54
a=0.1 #learningrate
epoch=100 #epoch

#fungsi-fungsi
def y1(x1,x2,x3,x4,w1,w2,w3,w4,b):
    return (w1*x1)+(w2*x2)+(w3*x3)+(w4*x4)+b

def y2(x1,x2,w1,w2,b):
    return (w1*x1)+(w2*x2)+b
    
def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))    

def tetabaru(t,a,fact,pred):
    return t-(a*(2*(pred-fact)*(1-pred)*pred*x))

#cleaning data
#setosa = 0
#versicolor = 1
# = 2
for x in range(0,len(iris)):
    if iris.iloc[x,4] in ['Iris-setosa']:
        iris.iloc[x,4]=0
    elif iris.iloc[x,4] in ['Iris-versicolor']:
        iris.iloc[x,4]=1
    else:
        iris.iloc[x,4]=2
        
#preparation for training dataset
x1 = iris.iloc[:,0]
x2 = iris.iloc[:,1]
x3 = iris.iloc[:,2]
x4 = iris.iloc[:,3]
label = iris.iloc[:,4]
avg_error = pd.Series([])
in5 = pd.Series([])
in6 = pd.Series([])
out5 = pd.Series([])
out6 = pd.Series([])
taw5 = pd.Series([])
taw6 = pd.Series([])
in7 = pd.Series([])
in8 = pd.Series([])
in9 = pd.Series([])
out7 = pd.Series([])
out8 = pd.Series([])
out9 = pd.Series([])
taw7 = pd.Series([])
taw8 = pd.Series([])
taw9 = pd.Series([])
error = pd.Series([])


def feed(data):
    global w15,w25,w35,w45,w16,w26,w36,w46,b5,b6,w57,w67,w58,w68,w59,w69,b7,b8,b9
    for z in range(0,len(data)):
        in5[z] = y1(x1[z],x2[z],x3[z],x4[z],w15[z],w25[z],w35[z],w45[z],b5[z])
        in6[z] = y1(x1[z],x2[z],x3[z],x4[z],w16[z],w26[z],w36[z],w46[z],b6[z])
        out5[z] = sigmoid_activation(in5[z])
        out6[z] = sigmoid_activation(in6[z])
        in7[z] = y2(out5[z],out6[z],w57[z],w67[z],b7[z])
        in8[z] = y2(out5[z],out6[z],w58[z],w68[z],b8[z])
        in9[z] = y2(out5[z],out6[z],w59[z],w69[z],b9[z])
        out7[z] = sigmoid_activation(in7[z])
        out8[z] = sigmoid_activation(in8[z])
        out9[z] = sigmoid_activation(in9[z])
        if label[z]==0:
            error[z] = ((1-out7[z])**2 + (0-out8[z])**2 + (0-out9[z])**2)/3
        elif label[z]==1:
            error[z] = ((0-out7[z])**2 + (1-out8[z])**2 + (0-out9[z])**2)/3
        else:
            error[z] = ((0-out7[z])**2 + (0-out8[z])**2 + (1-out9[z])**2)/3
    avg_error = np.mean(error)
    return avg_error

def test(data):
    for z in range(0,len(data)):
        in5[z] = y1(x1[z],x2[z],x3[z],x4[z],w15[0],w25[0],w35[0],w45[0],b5[0])
        in6[z] = y1(x1[z],x2[z],x3[z],x4[z],w16[0],w26[0],w36[0],w46[0],b6[0])
        out5[z] = sigmoid_activation(in5[z])
        out6[z] = sigmoid_activation(in6[z])
        in7[z] = y2(out5[z],out6[z],w57[0],w67[0],b7[0])
        in8[z] = y2(out5[z],out6[z],w58[0],w68[0],b8[0])
        in9[z] = y2(out5[z],out6[z],w59[0],w69[0],b9[0])
        out7[z] = sigmoid_activation(in7[z])
        out8[z] = sigmoid_activation(in8[z])
        out9[z] = sigmoid_activation(in9[z])
        if label[z]==0:
            error[z] = ((1-out7[z])**2 + (0-out8[z])**2 + (0-out9[z])**2)/3
        elif label[z]==1:
            error[z] = ((0-out7[z])**2 + (1-out8[z])**2 + (0-out9[z])**2)/3
        else:
            error[z] = ((0-out7[z])**2 + (0-out8[z])**2 + (1-out9[z])**2)/3
    avg_error = np.mean(error)
    return avg_error

error_train=pd.Series([])
error_test=pd.Series([])  

fig = plt.figure()
plt.plot(np.arange(0, epoch), error_test, linestyle='-', label='test')
plt.plot(np.arange(0, epoch), error_train, color='orange', linestyle='dashed', label='train')
plt.legend()
fig.suptitle("Cost Function, alpha = %s" %(a))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
