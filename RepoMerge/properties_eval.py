# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:35:36 2021

@author: mmeierdo
"""


import numpy as np
import pysindy as ps 
from pysindy.feature_library import PolynomialLibrary #,  CustomLibrary, FourierLibrary, IdentityLibrary
from sklearn.preprocessing import normalize
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as score
from gen_plots import plot_function_supervised2 as pfs2
from gen_plots import plot_function_supervised2c as pfs2c

numberOfComponents = 2; var_count = 3; deg = 4; bias = False;
test_size = 0.25
seed = 296

## Vapor Concentration
DCM_100 = 320000*2; DCP_100 = 800*2; DMMP_100 = 400*2
EtOH_100 = 28800*2; MeOH_100 = 63800*2; Water_100 = 11400*2

## Refractive Index
DCM_Refractive = 1.424; DCP_Refractive = 1.457; DMMP_Refractive = 1.414
EtOH_Refractive = 1.361; MeOH_Refractive = 1.328; Water_Refractive = 1.333
Chitin_Refractive = 1.56

## Relative Polarity
DCM_Polarity = 0.309; DCP_Polarity = 0.4; DMMP_Polarity = 0
EtOH_Polarity = 0.654; MeOH_Polarity = 0.762; Water_Polarity = 1

## Computational Polar Surface Area
DCM_CPSA = 0; DCP_CPSA = 0 ; DMMP_CPSA = 35.5
EtOH_CPSA = 20.2; MeOH_CPSA = 20.2; Water_CPSA = 1

## Kovats Retention Index
DCM_Kovats = 932; DCP_Kovats = 1444; DMMP_Kovats = 1497
EtOH_Kovats = 929; MeOH_Kovats = 900; Water_Kovats = 1053

Base = [[DCM_100, DCP_100, EtOH_100, MeOH_100, Water_100],
       [DCM_Refractive, DCP_Refractive,EtOH_Refractive,MeOH_Refractive,Water_Refractive],
       [DCM_Polarity, DCP_Polarity, EtOH_Polarity, MeOH_Polarity, Water_Polarity],
       [DCM_CPSA, DCP_CPSA, EtOH_CPSA, MeOH_CPSA, Water_CPSA],
       [DCM_Kovats, DCP_Kovats, EtOH_Kovats, MeOH_Kovats, Water_Kovats]
       ]

Base = normalize(Base)

path = r"/Users/maycon/Desktop/Codes_Properties/input_params"+str(numberOfComponents)+".mat"
path = r"/Users/maycon/Desktop/Codes_Properties/all_data.mat"


data = loadmat(path)

Xd = data['XTrainLDA']
Xt = data['XTestLDA']
Y1 = data['YTrainLDA']
Yt = data['YTestLDA']
track = data['trackl']

sq = np.concatenate(track[0:len(Xd)])
seq = np.concatenate(Y1)

Fx_Train = np.empty([Y1.size,var_count])

ind = 0
for i in seq: 
    j = sq[ind]
    k = int(i)
    if var_count == 1: 
        Fx_Train[ind,0] = Base[1,k]
    if var_count == 2:
        Fx_Train[ind,0] = Base[1,k]
        Fx_Train[ind,1] = Base[4,k]
    if var_count == 3:
        Fx_Train[ind,0] = Base[1,k]
        Fx_Train[ind,1] = Base[4,k]
        Fx_Train[ind,2] = Base[0,k] * j
    if var_count == 4:
        Fx_Train[ind,0] = Base[1,k]
        Fx_Train[ind,1] = Base[4,k]
        Fx_Train[ind,2] = Base[0,k] * j
        Fx_Train[ind,3] = Base[3,k]
    if var_count == 5:
        Fx_Train[ind,0] = Base[1,k]
        Fx_Train[ind,1] = Base[4,k]
        Fx_Train[ind,2] = Base[0,k] * j
        Fx_Train[ind,3] = Base[3,k]
        Fx_Train[ind,4] = Base[2,k]
    ind += 1

sq = np.concatenate(track[len(Xd):len(Y1)+len(Yt)])
seq = np.concatenate(Yt)

Fx_Test = np.empty([Yt.size,var_count])
ind = 0
for i in seq: 
    j = sq[ind]
    k = int(i)
    if var_count == 1: 
        Fx_Test[ind,0] = Base[1,k]
    if var_count == 2:
        Fx_Test[ind,0] = Base[1,k]
        Fx_Test[ind,1] = Base[4,k]
    if var_count == 3:
        Fx_Test[ind,0] = Base[1,k]
        Fx_Test[ind,1] = Base[4,k]
        Fx_Test[ind,2] = Base[0,k] * j
    if var_count == 4:
        Fx_Test[ind,0] = Base[1,k]
        Fx_Test[ind,1] = Base[4,k]
        Fx_Test[ind,2] = Base[0,k] * j
        Fx_Test[ind,3] = Base[3,k]
    if var_count == 5:
        Fx_Test[ind,0] = Base[1,k]
        Fx_Test[ind,1] = Base[4,k]
        Fx_Test[ind,2] = Base[0,k] * j
        Fx_Test[ind,3] = Base[3,k]
        Fx_Test[ind,4] = Base[2,k]
    ind += 1

x_dot_train = Xd[:, 0]
y_dot_train = Xd[:, 1]
if numberOfComponents == 3:
    z_dot_train = Xd[:, 2]

x_test = Xt[:, 0]
y_test = Xt[:, 1]
if numberOfComponents == 3:
    z_test = Xt[:, 2]


names = ['R','K','C','S','P'] 
names = names[0:var_count]

model = ps.SINDy(feature_names = names, feature_library=PolynomialLibrary(degree=deg, include_bias=bias))
                 #feature_library=IdentityLibrary() + FourierLibrary(n_frequencies=1, include_sin=True, include_cos=True))

model.fit(x = Fx_Train, x_dot = x_dot_train)
x_pysindy_score = model.score(x = Fx_Train, x_dot = x_dot_train)
x_coef = model.coefficients()
x_labels = model.get_feature_names()
model.print('x')
x_score = model.score(x = Fx_Train, x_dot = x_dot_train) #, metric = mean_squared_error)

model.fit(x = Fx_Train, x_dot = y_dot_train)
y_pysindy_score = model.score(x = Fx_Train, x_dot = y_dot_train)
y_coef = model.coefficients()
y_labels = model.get_feature_names()
model.print('y')
y_score = model.score(x = Fx_Train, x_dot = y_dot_train)


if numberOfComponents == 3:
    model.fit(x=Fx_Train, x_dot = z_dot_train)
    z_coef = model.coefficients()
    z_labels = model.get_feature_names()
    z_score = model.score(x = Fx_Train, x_dot = z_dot_train)

if numberOfComponents == 2: 
    to_save = {"x_dot": x_dot_train, 'y_dot': y_dot_train, 'x_coef': x_coef, 'x_labels': x_labels,
           'y_coef': y_coef, 'y_labels': y_labels}
else: 
    to_save = {"x_dot": x_dot_train, 'x_coef': x_coef, 'x_labels': x_labels,
           'y_dot': y_dot_train, 'y_coef': y_coef, 'y_labels': y_labels,
           'z_dot': z_dot_train, 'z_coef': z_coef, 'z_labels': z_labels}
    
savemat("equations.mat", to_save)

x_coef = np.reshape(x_coef, len(x_coef[0]))
y_coef = np.reshape(y_coef, len(y_coef[0]))

if bias == False: 
    x_coef = np.insert(x_coef, 0, 0)
    y_coef = np.insert(y_coef, 0, 0)

if var_count == 2 and deg == 3:
    x = lambda R, K: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * R**2 + x_coef[4] * R * K + x_coef[5] * K**2 + x_coef[6]* R**3 + x_coef[7]*R**2 * K + x_coef[8]*R*K**2 + x_coef[9]*K**3;

    y = lambda R, K: y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * R**2 + y_coef[4] * R * K + y_coef[5] * K**2 + y_coef[6]* R**3 + y_coef[7]*R**2 * K + y_coef[8]*R*K**2 + y_coef[9]*K**3;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1])    
        
elif var_count == 2 and deg == 2: 
    x = lambda R, K: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * R**2 + x_coef[4] * R * K + x_coef[5] * K**2;

    y = lambda R, K: y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * R**2 + y_coef[4] * R * K + y_coef[5] * K**2;
        
    if numberOfComponents  == 3: 
        z = lambda R, K: z_coef[0] + z_coef[1] * R + z_coef[2] * K + z_coef[3] * R**2 + z_coef[4] * R * K + z_coef[5] * K**2;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1])    

elif var_count == 3 and deg == 1: 
    x = lambda R, K, C: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * C ;

    y = lambda R, K, C:  y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * C ;
        
    if numberOfComponents  == 3: 
        z = lambda R, K, C:  z_coef[0] + z_coef[1] * R + z_coef[2] * K + z_coef[3] * C;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])         

elif var_count == 3 and deg == 2: 
    x = lambda R, K, C: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * C + x_coef[4] * R**2 + x_coef[5] * R * K + x_coef[6]*R*C + x_coef[7] * K**2 + x_coef[8]*K*C + x_coef[9]*C**2;

    y = lambda R, K, C:  y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * C + y_coef[4] * R**2 + y_coef[5] * R * K + y_coef[6]*R*C + y_coef[7] * K**2 + y_coef[8]*K*C + y_coef[9]*C**2;
        
    if numberOfComponents  == 3: 
        z = lambda R, K, C:  z_coef[0] + z_coef[1] * R + z_coef[2] * K + z_coef[3] * C + z_coef[4] * R**2 + z_coef[5] * R * K + z_coef[6]*R*C + z_coef[7] * K**2 + z_coef[8]*K*C + z_coef[9]*C**2;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])  

elif var_count == 3 and deg == 3: 
    x = lambda R, K, C: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * C + x_coef[4] * R**2 + x_coef[5] * R * K + x_coef[6]*R*C + x_coef[7] * K**2 + x_coef[8]*K*C + x_coef[9]*C**2 + x_coef[10] * R**3 + x_coef[11] * R**2 * K + x_coef[12]*R**2 * C + x_coef[13] * R * K**2 + x_coef[14]*R*K*C+ x_coef[15] *R*C**2 + x_coef[16] * K**3 + x_coef[17] * K**2 * C + x_coef[18] * K * C**2 + x_coef[19] * C**3;

    y = lambda R, K, C:  y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * C + y_coef[4] * R**2 + y_coef[5] * R * K + y_coef[6]*R*C + y_coef[7] * K**2 + y_coef[8]*K*C + y_coef[9]*C**2 + y_coef[10] * R**3 + y_coef[11] * R**2 * K + y_coef[12]*R**2 * C + y_coef[13] * R * K**2 + y_coef[14]*R*K*C+ y_coef[15] *R*C**2 + y_coef[16] * K**3 + y_coef[17] * K**2 * C + y_coef[18] * K * C**2 + y_coef[19] * C**3;
        
    if numberOfComponents  == 3: 
        z = lambda R, K, C:  z_coef[0] + z_coef[1] * R + z_coef[2] * K + z_coef[3] * C + z_coef[4] * R**2 + z_coef[5] * R * K + z_coef[6]*R*C + z_coef[7] * K**2 + z_coef[8]*K*C + z_coef[9]*C**2 + z_coef[10] * R**3 + z_coef[11] * R**2 * K + z_coef[12]*R**2 * C + z_coef[13] * R * K**2 + z_coef[14]*R*K*C+ z_coef[15] *R*C**2 + z_coef[16] * K**3 + z_coef[17] * K**2 * C + z_coef[18] * K * C**2 + z_coef[19] * C**3;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])  

elif var_count == 3 and deg == 4: 
    x = lambda R, K, C: x_coef[0] + x_coef[1] * R + x_coef[2] * K + x_coef[3] * C + x_coef[4] * R**2 + x_coef[5] * R * K + x_coef[6]*R*C + x_coef[7] * K**2 + x_coef[8]*K*C + x_coef[9]*C**2 + x_coef[10] * R**3 + x_coef[11] * R**2 * K + x_coef[12]*R**2 * C + x_coef[13] * R * K**2 + x_coef[14]*R*K*C+ x_coef[15] *R*C**2 + x_coef[16] * K**3 + x_coef[17] * K**2 * C + x_coef[18] * K * C**2 + x_coef[19] * C**3 + x_coef[20] * R**4 +x_coef[21]*R**3 * K + x_coef[22]*R**3*C + x_coef[23]*R**2*K**2 + x_coef[24]*R**2*K*C+x_coef[25]*R**2*C**2 + x_coef[26]*R*K**3+x_coef[27]*R*K**2*C+x_coef[28]*R*K*C**2+x_coef[29]*R*C**3+K**4*x_coef[30]+K**3*C*x_coef[31] + K**2*C**2*x_coef[32]+K*C**3*x_coef[33]+x_coef[34]*C**4;

    y = lambda R, K, C:  y_coef[0] + y_coef[1] * R + y_coef[2] * K + y_coef[3] * C + y_coef[4] * R**2 + y_coef[5] * R * K + y_coef[6]*R*C + y_coef[7] * K**2 + y_coef[8]*K*C + y_coef[9]*C**2 + y_coef[10] * R**3 + y_coef[11] * R**2 * K + y_coef[12]*R**2 * C + y_coef[13] * R * K**2 + y_coef[14]*R*K*C+ y_coef[15] *R*C**2 + y_coef[16] * K**3 + y_coef[17] * K**2 * C + y_coef[18] * K * C**2 + y_coef[19] * C**3+ y_coef[20] * R**4 +y_coef[21]*R**3 * K + y_coef[22]*R**3*C + y_coef[23]*R**2*K**2 + y_coef[24]*R**2*K*C+y_coef[25]*R**2*C**2 + y_coef[26]*R*K**3+y_coef[27]*R*K**2*C+y_coef[28]*R*K*C**2+y_coef[29]*R*C**3+K**4*y_coef[30]+K**3*C*y_coef[31] + K**2*C**2*y_coef[32]+K*C**3*y_coef[33]+y_coef[34]*C**4;
        
    if numberOfComponents  == 3: 
        z = lambda R, K, C:  z_coef[0] + z_coef[1] * R + z_coef[2] * K + z_coef[3] * C + z_coef[4] * R**2 + z_coef[5] * R * K + z_coef[6]*R*C + z_coef[7] * K**2 + z_coef[8]*K*C + z_coef[9]*C**2 + z_coef[10] * R**3 + z_coef[11] * R**2 * K + z_coef[12]*R**2 * C + z_coef[13] * R * K**2 + z_coef[14]*R*K*C+ z_coef[15] *R*C**2 + z_coef[16] * K**3 + z_coef[17] * K**2 * C + z_coef[18] * K * C**2 + z_coef[19] * C**3+ z_coef[20] * R**4 +z_coef[21]*R**3 * K + z_coef[22]*R**3*C + z_coef[23]*R**2*K**2 + z_coef[24]*R**2*K*C+z_coef[25]*R**2*C**2 + z_coef[26]*R*K**3+z_coef[27]*R*K**2*C+z_coef[28]*R*K*C**2+z_coef[29]*R*C**3+K**4*z_coef[30]+K**3*C*z_coef[31] + K**2*C**2*z_coef[32]+K*C**3*z_coef[33]+z_coef[34]*C**4;

    x_dot_compute = np.empty([len(x_test)])
    y_dot_compute = np.empty([len(y_test)])
    for i in range(len(x_test)):
        x_dot_compute[i] = x(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2])   
        y_dot_compute[i] = y(R = Fx_Test[i,0], K = Fx_Test[i,1], C = Fx_Test[i, 2]) 



x_percentage_error = []
y_percentage_error = [] 
for i in range(len(x_test)): 
    x_percentage_error.append(100*np.absolute((x_dot_compute[i] - x_test[i])/x_test[i]))
    y_percentage_error.append(100*np.absolute((y_dot_compute[i] - y_test[i])/y_test[i]))   


plt.plot(x_percentage_error,'o', label = 'x_dot') 
plt.plot(y_percentage_error,'o', label = 'y_dot') 
plt.title('Percentage Error')
plt.xlabel('points')
plt.ylabel('%')  
plt.legend()           

x_score = score(x_test, x_dot_compute)
x_mean_percentage = np.mean(x_percentage_error)
y_score = score(y_test, y_dot_compute)
y_mean_percentage = np.mean(y_percentage_error)

print('X_Pysindy_r2: ', x_pysindy_score, 'Y_pysindy_r2: ', y_pysindy_score)
print('X_r2_score:', x_score, ' X_MPE: ',x_mean_percentage)
print('Y_r2_score:', y_score, ' Y_MPE: ',y_mean_percentage)

Built_Matrix = np.concatenate((np.reshape(x_dot_compute, (len(x_dot_compute),1)), np.reshape(y_dot_compute, (len(y_dot_compute),1))), axis = 1)


# ========================================================================== # 

from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score

kval = 4
neigh_lda = KNC(n_neighbors = kval)
neigh_lda.fit(Xd, Y1)
y_pred = neigh_lda.predict(Built_Matrix)
y_pred.astype(int)
Yt2 = np.reshape(Yt, (1,len(Yt)))
Yt2.astype(int)
KNC_accuracy_lda = accuracy_score(Yt2[0], y_pred)
print('KNC Accuracy: ',KNC_accuracy_lda)

pfs2(Xd, Y1, Built_Matrix, y_pred, var = var_count, deg = deg)
pfs2c(Xd, Y1, Built_Matrix, y_pred, var = var_count, deg = deg)


x_axis = np.linspace(0.02,0.3, 200)
y_axis = np.linspace(0.02, 0.3, 200)

r1 = Base[1][0]
r2 = Base[1][1]
r3 = Base[1][2]
r4 = Base[1][3]
r5 = Base[1][4]

k1 = Base[4][0]
k2 = Base[4][1]
k3 = Base[4][2]
k4 = Base[4][3]
k5 = Base[4][4]

r = r5
k = k5

x_ev = x(R = r, K = k, C = x_axis)
y_ev = y(R = r, K = k, C = y_axis)

plt.figure()
plt.plot(x_axis, x_ev)
plt.plot(y_axis, y_ev)

plt.figure()
plt.plot(x_ev, y_ev)

plt.figure()
plt.plot(x(R = r1, K = k1, C = x_axis), y(R = r1, K = k1, C = y_axis), c = 'r')
plt.plot(x(R = r2, K = k2, C = x_axis), y(R = r2, K = k2, C = y_axis), c = 'g')
plt.plot(x(R = r3, K = k3, C = x_axis), y(R = r3, K = k3, C = y_axis), c = 'b')
plt.plot(x(R = r4, K = k4, C = x_axis), y(R = r4, K = k4, C = y_axis), c = 'c')
plt.plot(x(R = r5, K = k5, C = x_axis), y(R = r5, K = k5, C = y_axis), c = 'm')

plt.legend(['DCM', 'DCP', 'EtOH', 'MeOH', 'Water'])
plt.title('3 Variables; Degree = '+str(deg)+'; Variable Concentration')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.xlim([-100, 100])
plt.ylim([-100, 100])

