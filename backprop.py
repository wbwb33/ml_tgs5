# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import re
import pandas as pd
from math import exp
import matplotlib.pyplot as plt

data = 150
a = 0.01
epoch = 100

x1 = np.empty(data)
x2 = np.empty(data)
x3 = np.empty(data)
x4 = np.empty(data)
kelas = np.empty(data, dtype='S30')

fakta1 = np.empty(data, dtype='int32')
fakta2 = np.empty(data, dtype='int32')
fakta3 = np.empty(data, dtype='int32')

fakta1[0:50] = 1
fakta1[50:150] = 0
fakta2[0:50] = 0
fakta2[50:100] = 1
fakta2[100:150] = 0
fakta3[0:100] = 0
fakta3[100:150] = 1

def get_data(dt,grp):
	f = open('iris.txt','r')
	f1 = f.readlines()
	p = re.compile(r'([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\w\-]+)')
	for i in range(data):
		dt[i] = p.match(f1[i]).group(grp)
	f.close()
	return dt

def signoid(x):
    return 1 / (1 + exp(-x))

def table_count(data):
    return data.iloc[:,0].size

def theta_bias_initiation():
    theta1 = np.array([[0.88, 0.36, 0.03, 0.57], 
                       [0.02, 0.93, 0.54, 0.32],
                       [0.57, 0.03, 0.99, 0.88]])
    bias1 = np.array([0.39,0.93,0.27])

    theta2 = np.array([[0.82, 0.3, 0.64], 
                       [0.65, 0.2, 0.52],
                       [0.7, 0.55, 0.18]])
    bias2 = np.array([0.94,0.56,0.55])

    return [theta1, bias1, theta2, bias2]

def back_prop(training,params):
    theta1 = np.array(params[0])
    bias1 = np.array(params[1])
    theta2 = np.array(params[2])
    bias2 = np.array(params[3])
    
    error1 = 0
    error2 = 0
    error3 = 0

    for i in range(table_count(training)):
      
        h11 = sum(training.iloc[i,4:8]*theta1[0,0:4]) + bias1[0]
        h12 = sum(training.iloc[i,4:8]*theta1[1,0:4]) + bias1[1]
        h13 = sum(training.iloc[i,4:8]*theta1[2,0:4]) + bias1[2]
       
        s11 = signoid(h11)
        s12 = signoid(h12)
        s13 = signoid(h13)
    
        h21 = s11 * theta2[0][0] + s12 * theta2[0][1] + s13 * theta2[0][2] + bias2[0]
        h22 = s11 * theta2[1][0] + s12 * theta2[1][1] + s13 * theta2[1][2] + bias2[1]
        h23 = s11 * theta2[2][0] + s12 * theta2[2][1] + s13 * theta2[2][2] + bias2[2]

        s21 = signoid(h21)
        s22 = signoid(h22)
        s23 = signoid(h23)

        error1 = error1 + (s21 - training.iloc[i,0])**2
        error2 = error2 + (s22 - training.iloc[i,1])**2
        error3 = error3 + (s23 - training.iloc[i,2])**2

        tau21 = 2*(s21 - training.iloc[i,0]) * (1-s21)*s21
        dw21 = tau21 * s11
        dw24 = tau21 * s12
        dw27 = tau21 * s13

    
        tau22 = 2*(s22 - training.iloc[i,1]) * (1-s22)*s22
        dw22 = tau22 * s11
        dw25 = tau22 * s12
        dw28 = tau22 * s13
        
        tau23 = 2*(s23 - training.iloc[i,2]) * (1-s23) * s23
        dw23 = tau23 * s11
        dw26 = tau23 * s12
        dw29 = tau23 * s13
        
        tau11 = (tau21 * theta2[0,0] + tau22 * theta2[1,0] + tau23 * theta2[2,0]) * (1-s11) * s11
        dw11 = tau11 * training.iloc[i,4]
        dw14 = tau11 * training.iloc[i,5]
        dw17 = tau11 * training.iloc[i,6]
        dw110 = tau11 * training.iloc[i,7]

        tau12 = (tau21 * theta2[0,1] + tau22 * theta2[1,1] + tau23 * theta2[2,1]) *(1-s12)*s12
        dw12 = tau12 * training.iloc[i,4]
        dw15 = tau12 * training.iloc[i,5]
        dw18 = tau12 * training.iloc[i,6]
        dw111 = tau12 * training.iloc[i,7]
        
        tau13 = (tau21 * theta2[0,2] + tau22 * theta2[1,2] + tau23 * theta2[2,2]) * (1-s13) * s13
        dw13 = tau13 * training.iloc[i,4]
        dw16 = tau13 * training.iloc[i,5]
        dw19 = tau13 * training.iloc[i,6]
        dw112 = tau13 * training.iloc[i,7]
    
        dw2 = np.array([[dw21, dw24, dw27],
                        [dw22, dw25, dw28],
                        [dw23, dw26, dw29]])

        dbias2 = np.array([tau21, tau22, tau23])
  
        dw1 = np.array([[dw11, dw14, dw17, dw110],
                        [dw12, dw15, dw18, dw111],
                        [dw13, dw16, dw19, dw112]])
     
        dbias1 = np.array([tau11, tau12, tau13])
      
        theta1 = theta1 - a * dw1
        bias1 = bias1 - a * dbias1
        theta2 = theta2 - a * dw2
        bias2 = bias2 - a * dbias2
        
    error = np.array([error1, error2,error3])
    error = error/table_count(training)
    error = np.mean(error)

    
    return np.array([[error],[theta1, bias1, theta2, bias2]])

def predict(signoid):
    if(signoid < 0.5):
        return 0
    else:
        return 1

def checking_accuration(data1,data2):
    checking = np.zeros(table_count(data1), dtype=bool)

    for i in range(table_count(data1)):
        
        checking[i] = np.array_equal(data1.iloc[i].values, data2.iloc[i].values)
    
    return checking

def test(testing, params):
    theta1 = np.array(params[0])
    bias1 = np.array(params[1])
    theta2 = np.array(params[2])
    bias2 = np.array(params[3])
    
    prediksi1 = np.zeros(table_count(testing), dtype='int32')
    prediksi2 = np.zeros(table_count(testing), dtype='int32')
    prediksi3 = np.zeros(table_count(testing), dtype='int32')

    error1 = 0
    error2 = 0
    error3 = 0
    
    for i in range(table_count(testing)):
    
        h11 = sum(testing.iloc[i,4:8]*theta1[0,0:4]) + bias1[0]
        h12 = sum(testing.iloc[i,4:8]*theta1[1,0:4]) + bias1[1]
        h13 = sum(testing.iloc[i,4:8]*theta1[2,0:4]) + bias1[2]
    
        s11 = signoid(h11)
        s12 = signoid(h12)
        s13 = signoid(h13)
    
        h21 = s11 * theta2[0][0] + s12 * theta2[0][1] + s13 * theta2[0][2] + bias2[0]
        h22 = s11 * theta2[1][0] + s12 * theta2[1][1] + s13 * theta2[1][2] + bias2[1]
        h23 = s11 * theta2[2][0] + s12 * theta2[2][1] + s13 * theta2[2][2] + bias2[2]
    
        s21 = signoid(h21)
        s22 = signoid(h22)
        s23 = signoid(h23)

        prediksi1[i] = predict(s21)
        prediksi2[i] = predict(s22)
        prediksi3[i] = predict(s23)
    
        error1 = error1 + (s21 - testing.iloc[i,0])**2
        error2 = error2 + (s22 - testing.iloc[i,1])**2
        error3 = error3 + (s23 - testing.iloc[i,2])**2
        
    error = np.array([error1, error2,error3])
    error = error/table_count(testing)
    error = np.mean(error)
    
    predict_table = pd.DataFrame({'prediksi 1':prediksi1,
                                  'prediksi 2':prediksi2,
                                  'prediksi 3':prediksi3})
    
    text.append(str(predict_table)+"\n")
  
    conditional = checking_accuration(testing.iloc[:,0:3], predict_table)
    
    unique, count = np.unique(conditional, return_counts=True)
    
    c = np.where(unique == True)
    if(c[0].size != 0):   
        akurasi = (float(count[c[0][0]]) / table_count(testing)) * 100
        text.append("Akurasi "+str(akurasi)+"\n")
    else:
        akurasi = 0
        text.append("Akurasi "+str(akurasi)+"\n")
    
    return np.array([error,akurasi])

f = open('output-back.txt','w')
text = []

x1 = get_data(x1,1)
x2 = get_data(x2,2)
x3 = get_data(x3,3)
x4 = get_data(x4,4)
kelas = get_data(kelas,5)

df = pd.DataFrame({'x1':x1,
                   'x2':x2,
                   'x3':x3,
                   'x4':x4,
                   'kelas':kelas,
                   'fakta 1':fakta1,
                   'fakta 2':fakta2,
                   'fakta 3':fakta3})

error_training_epoch = np.empty(epoch)
error_testing_epoch = np.empty(epoch)
akurasi_epoch = np.empty(epoch)

error_training = 0
error_testing = 0
akurasi = 0

text.append("Epoch 0\n")
text.append("=======\n")

testing = df.iloc[0:30]
training = df.iloc[30:150]

text.append(str(testing)+"\n")

training_result = back_prop(training, theta_bias_initiation())
error_training = error_training + training_result[0][0]

testing_result = test(testing,training_result[1])
error_testing = error_testing + testing_result[0]
akurasi = akurasi + testing_result[1]

for i in range(epoch):
    testing = df.iloc[30:60]
    training = df.iloc[0:30]
    training = training.append(df.iloc[60:150])

    text.append(str(testing)+"\n")
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    testing = df.iloc[60:90]
    training = df.iloc[0:60]
    training = training.append(df.iloc[90:150])

    text.append(str(testing)+"\n")
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    testing = df.iloc[90:120]
    training = df.iloc[0:90]
    training = training.append(df.iloc[120:150])

    text.append(str(testing)+"\n")
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]

    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    testing = df.iloc[120:150]
    training = df.iloc[0:120]
    
    text.append(str(testing)+"\n")
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]

    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    mean_error_training = error_training/5
    mean_error_testing = error_testing/5
    mean_akurasi = akurasi/5
    error_training_epoch[i] = mean_error_training
    error_testing_epoch[i] = mean_error_testing
    akurasi_epoch[i] = mean_akurasi
    
    if(i<epoch-1):
        text.append("Epoch "+str(i+1)+"\n")
        
        error_training = 0
        error_testing = 0
        akurasi = 0

        testing = df.iloc[0:30]
        training = df.iloc[30:150]
        text.append(str(testing)+"\n")
        
        training_result = back_prop(training, training_result[1])
        error_training = error_training + training_result[0][0]
 
        testing_result = test(testing,training_result[1])
        error_testing = error_testing + testing_result[0]
        akurasi = akurasi + testing_result[1]

print("Error Training tiap Epoch")
print(error_training_epoch)
print("Error Testing tiap Epoch")
print(error_testing_epoch)
print("Akurasi tiap Epoch")
print(akurasi_epoch)

f.writelines(text)
f.close

plt.plot(error_training_epoch,'r',error_testing_epoch,'g')

plt.show()
