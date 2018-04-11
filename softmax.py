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

def softmax(x,sigma):
    return (np.exp(x))/sigma

def tetabaru(t,a,taw,pred):
    return t-(a*taw*pred)

def taw_out(pred,fact):
    return (pred-fact)*(1-pred)*pred

def taw_hidd(w1,w2,w3,taw1,taw2,taw3,pred):
    return ((taw1*w1)+(taw2*w2)+(taw3*w3))*(1-pred)*pred

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
        sigma = (np.exp(in7[z])+np.exp(in8[z])+np.exp(in9[z]))
        out7[z] = softmax(in7[z],sigma)
        out8[z] = softmax(in8[z],sigma)
        out9[z] = softmax(in9[z],sigma)
        if label[z]==0:
            error[z] = ((1-out7[z])**2 + (0-out8[z])**2 + (0-out9[z])**2)/3
            taw7[z] = taw_out(out7[z],1)
            taw8[z] = taw_out(out8[z],0)
            taw9[z] = taw_out(out9[z],0)
        elif label[z]==1:
            error[z] = ((0-out7[z])**2 + (1-out8[z])**2 + (0-out9[z])**2)/3
            taw7[z] = taw_out(out7[z],0)
            taw8[z] = taw_out(out8[z],1)
            taw9[z] = taw_out(out9[z],0)
        else:
            error[z] = ((0-out7[z])**2 + (0-out8[z])**2 + (1-out9[z])**2)/3
            taw7[z] = taw_out(out7[z],0)
            taw8[z] = taw_out(out8[z],0)
            taw9[z] = taw_out(out9[z],1)
        w57[z+1]=tetabaru(w57[z],a,taw7[z],out5[z])
        w67[z+1]=tetabaru(w67[z],a,taw7[z],out6[z])
        w58[z+1]=tetabaru(w58[z],a,taw8[z],out5[z])
        w68[z+1]=tetabaru(w68[z],a,taw8[z],out6[z])
        w59[z+1]=tetabaru(w59[z],a,taw9[z],out5[z])
        w69[z+1]=tetabaru(w69[z],a,taw9[z],out6[z])
        b7[z+1]=tetabaru(b7[z],a,taw7[z],1)
        b8[z+1]=tetabaru(b8[z],a,taw8[z],1)
        b9[z+1]=tetabaru(b8[z],a,taw9[z],1)
        taw5[z]=taw_hidd(w57[z],w58[z],w59[z],taw7[z],taw8[z],taw9[z],out5[z])
        taw6[z]=taw_hidd(w67[z],w68[z],w69[z],taw7[z],taw8[z],taw9[z],out6[z])
        w15[z+1]=tetabaru(w15[z],a,taw5[z],x1[z])
        w25[z+1]=tetabaru(w25[z],a,taw5[z],x2[z])
        w35[z+1]=tetabaru(w35[z],a,taw5[z],x3[z])
        w45[z+1]=tetabaru(w45[z],a,taw5[z],x4[z])
        w16[z+1]=tetabaru(w16[z],a,taw6[z],x1[z])
        w26[z+1]=tetabaru(w26[z],a,taw6[z],x2[z])
        w36[z+1]=tetabaru(w36[z],a,taw6[z],x3[z])
        w46[z+1]=tetabaru(w46[z],a,taw6[z],x4[z])
        b5[z+1]=tetabaru(b5[z],a,taw5[z],1)
        b6[z+1]=tetabaru(b6[z],a,taw6[z],1)   
    w15[0]=w15[len(data)]
    w25[0]=w25[len(data)]
    w35[0]=w35[len(data)]
    w45[0]=w45[len(data)]
    w16[0]=w16[len(data)]
    w26[0]=w26[len(data)]
    w36[0]=w36[len(data)]
    w46[0]=w46[len(data)]
    b5[0]=b5[len(data)]
    b6[0]=b6[len(data)]
    w57[0]=w57[len(data)]
    w67[0]=w67[len(data)]
    w58[0]=w58[len(data)]
    w68[0]=w68[len(data)]
    w59[0]=w59[len(data)]
    w69[0]=w69[len(data)]
    b7[0]=b7[len(data)]
    b8[0]=b8[len(data)]
    b9[0]=b9[len(data)]
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
        sigma = (np.exp(in7[z])+np.exp(in8[z])+np.exp(in9[z]))
        out7[z] = softmax(in7[z],sigma)
        out8[z] = softmax(in8[z],sigma)
        out9[z] = softmax(in9[z],sigma)
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

for cur_epoch in range(0,epoch):
    #1
    training = iris.iloc[30:150]
    testing = iris.iloc[0:30]
    error_train[cur_epoch]=feed(training)
    error_test[cur_epoch]=test(testing)
    #2
    training = iris.iloc[0:30]
    training = training.append(iris.iloc[60:150])
    testing = iris.iloc[30:60]
    error_train[cur_epoch]=error_train[cur_epoch]+feed(training)
    error_test[cur_epoch]=error_test[cur_epoch]+test(testing)
    #3
    training = iris.iloc[0:60]
    training = training.append(iris.iloc[90:150])
    testing = iris.iloc[60:90]
    error_train[cur_epoch]=error_train[cur_epoch]+feed(training)
    error_test[cur_epoch]=error_test[cur_epoch]+test(testing)
    #4
    training = iris.iloc[0:90]
    training = training.append(iris.iloc[120:150])
    testing = iris.iloc[90:120]
    error_train[cur_epoch]=error_train[cur_epoch]+feed(training)
    error_test[cur_epoch]=error_test[cur_epoch]+test(testing)
    #5
    training = iris.iloc[0:120]
    testing = iris.iloc[120:150]
    error_train[cur_epoch]=error_train[cur_epoch]+feed(training)
    error_test[cur_epoch]=error_test[cur_epoch]+test(testing)
    #mean
    error_train[cur_epoch]=error_train[cur_epoch]/5
    error_test[cur_epoch]=error_test[cur_epoch]/5

fig = plt.figure()
plt.plot(np.arange(0, epoch), error_test, linestyle='-', label='test')
plt.plot(np.arange(0, epoch), error_train, color='orange', linestyle='dashed', label='train')
plt.legend()
fig.suptitle("Cost Function, alpha = %s" %(a))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
