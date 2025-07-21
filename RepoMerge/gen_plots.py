#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:33:37 2021

@author: maycon
"""
from matplotlib import pyplot as plt
import numpy as np

def plot_function_supervised2(XTrain, YTrain, XTest, YTest, var, deg):
    x_line = XTest[:,0]
    y_line = XTest[:,1]

    x_l1 = []; y_l1 = []; y_l=[]; x_l=[]; d = np.zeros(len(x_line)); dm = []
    for i in range(len(x_line)):
        d[i] = np.sqrt(x_line[i]**2 + y_line[i]**2)
    
    for i in range(len(x_line)):
        if x_line[i] not in x_l1:
            x_l1.append(x_line[i])
        if y_line[i] not in y_l1:
            y_l1.append(y_line[i])
        if d[i] not in dm:
            dm.append(d[i])
        
    x_l1 = np.sort(x_l1)
    y_l1 = np.sort(y_l1)
    dm = np.sort(dm)
    for i in range(len(x_l1)-1):
            x_l.append((x_l1[i] + x_l1[i+1])/ 2)  
            y_l.append((y_l1[i] + y_l1[i+1])/ 2)  
            
    
    fig = plt.figure(figsize=(15,15))
    #fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    '''
    for i in range(len(x_l)):
        ax.plot([x_l[i], x_l[i]],[-7,7], '--', color='k')'''
    #ax.plot([-6, 6],[0,0], '--', color='k') 
    #ax.plot([0,0],[-6,6], '--', color='k')
    '''
    for i in range(4):
        ax.add_patch(plt.Circle((0, 0), dm[i], color='k', fill=False))'''
    #ax = fig.add_subplot(projection='3d')
    '''
    ax.text(4.5,-2,"K= "+ str(932)+ " R= " + str(1.424)+"\n f(K,R)= 4.393", fontsize = "xx-large" ) 
    ax.text(2,4.5,"K= "+ str(1444)+ " R= " + str(1.457)+"\n f(K,R)= 4.616", fontsize = "xx-large" ) 
    ax.text(-4,-0.5,"K= "+ str(929)+ " R= " + str(1.361)+"\n f(K,R)= 4.329", fontsize = "xx-large" ) 
    ax.text(-1.5,-4,"K= "+ str(900)+ " R= " + str(1.328)+"\n f(K,R)= 4.282", fontsize = "xx-large" ) 
    ax.text(0.5,1,"K= "+ str(1053)+ " R= " + str(1.333)+"\n f(K,R)= 4.355", fontsize = "xx-large" ) 
    '''
    
    #ax.arrow(5.12, -2.74, 4.16,  -1.53, head_width=0.2, head_length=0.1, fc='r', ec='r')
    #ax.arrow(1.48, 4.59, 1.41,  4.5, head_width=0.2, head_length=0.1, fc='g', ec='g')
    #ax.arrow(-4.12, -0.23, -3.98, -0.18, head_width=0.2, head_length=0.1, fc='b', ec='b')
    #ax.arrow(-1.7, -3.91, -1.33, -2.7, head_width=0.2, head_length=0.1, fc='c', ec='c')
    #ax.arrow(-0.02, 1.68, -0.13, 1.62, head_width=0.2, head_length=0.1, fc='m', ec='m')
    
    ax.set_xlabel( ' Component 1', fontsize = 15)
    ax.set_ylabel( ' Component 2', fontsize = 15)
    #ax.set_zlabel( ' Component 3', fontsize = 15)
    ax.set_title(str(var)+' Variables, Deg = '+ str(deg) , fontsize = 20)
    targets = [0, 1, 2, 3, 4]
    targetsNames = ['','','','','Pc1 = Pc2', 'DCMTrain', 'DCM Pred', 'DCPTrain', 'DCP Pred','EtOHTrain',
    'EtOH Pred', 'MeOHTrain', 'MeOH Pred', 'WaterTrain', 'Water Pred']
    colors = ['r', 'g', 'b', 'c', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = np.where(YTrain == target)
        ax.scatter(XTrain[indicesToKeep[0], 0]
                   , XTrain[indicesToKeep[0],1]
                                      , c = color
                   , s = 50
                   , marker = 'o')
        indicesToKeep = np.where(YTest == target)
        ax.scatter(XTest[indicesToKeep[0], 0],
                XTest[indicesToKeep[0], 1],
                               c = color,
                s = 50, 
                marker = 'x')
        #ax.legend(targetsNames)
    '''
    plt.axline((-4.7, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((-1, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((2, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((3, 2), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((0, 0), slope=1, color="black")
    '''
    ax.legend(targetsNames, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.grid()

def plot_function_supervised2c(XTrain, YTrain, XTest, YTest, var, deg):
    x_line = XTest[:,0]
    y_line = XTest[:,1]

    x_l1 = []; y_l1 = []; y_l=[]; x_l=[]; d = np.zeros(len(x_line)); dm = []
    for i in range(len(x_line)):
        d[i] = np.sqrt(x_line[i]**2 + y_line[i]**2)
    
    for i in range(len(x_line)):
        if x_line[i] not in x_l1:
            x_l1.append(x_line[i])
        if y_line[i] not in y_l1:
            y_l1.append(y_line[i])
        if d[i] not in dm:
            dm.append(d[i])
        
    x_l1 = np.sort(x_l1)
    y_l1 = np.sort(y_l1)
    dm = np.sort(dm)
    for i in range(len(x_l1)-1):
            x_l.append((x_l1[i] + x_l1[i+1])/ 2)  
            y_l.append((y_l1[i] + y_l1[i+1])/ 2)  
            
    
    fig = plt.figure(figsize=(15,15))
    #fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    '''
    for i in range(len(x_l)):
        ax.plot([x_l[i], x_l[i]],[-7,7], '--', color='k')'''
    #ax.plot([-6, 6],[0,0], '--', color='k') 
    #ax.plot([0,0],[-6,6], '--', color='k')
    '''
    for i in range(4):
        ax.add_patch(plt.Circle((0, 0), dm[i], color='k', fill=False))'''
    #ax = fig.add_subplot(projection='3d')
    '''
    ax.text(4.5,-2,"K= "+ str(932)+ " R= " + str(1.424)+"\n f(K,R)= 4.393", fontsize = "xx-large" ) 
    ax.text(2,4.5,"K= "+ str(1444)+ " R= " + str(1.457)+"\n f(K,R)= 4.616", fontsize = "xx-large" ) 
    ax.text(-4,-0.5,"K= "+ str(929)+ " R= " + str(1.361)+"\n f(K,R)= 4.329", fontsize = "xx-large" ) 
    ax.text(-1.5,-4,"K= "+ str(900)+ " R= " + str(1.328)+"\n f(K,R)= 4.282", fontsize = "xx-large" ) 
    ax.text(0.5,1,"K= "+ str(1053)+ " R= " + str(1.333)+"\n f(K,R)= 4.355", fontsize = "xx-large" ) 
    '''
    
    #ax.arrow(5.12, -2.74, 4.16,  -1.53, head_width=0.2, head_length=0.1, fc='r', ec='r')
    #ax.arrow(1.48, 4.59, 1.41,  4.5, head_width=0.2, head_length=0.1, fc='g', ec='g')
    #ax.arrow(-4.12, -0.23, -3.98, -0.18, head_width=0.2, head_length=0.1, fc='b', ec='b')
    #ax.arrow(-1.7, -3.91, -1.33, -2.7, head_width=0.2, head_length=0.1, fc='c', ec='c')
    #ax.arrow(-0.02, 1.68, -0.13, 1.62, head_width=0.2, head_length=0.1, fc='m', ec='m')
    
    ax.set_xlabel( ' Component 1', fontsize = 15)
    ax.set_ylabel( ' Component 2', fontsize = 15)
    #ax.set_zlabel( ' Component 3', fontsize = 15)
    ax.set_title(str(var)+' Variables, Deg = '+ str(deg) , fontsize = 20)
    targets = [0, 1, 2, 3, 4]
    targetsNames = ['','','','','Pc1 = Pc2', 'DCMTrain', 'DCM Pred', 'DCPTrain', 'DCP Pred','EtOHTrain',
    'EtOH Pred', 'MeOHTrain', 'MeOH Pred', 'WaterTrain', 'Water Pred']
    colors = ['r', 'g', 'b', 'c', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = np.where(YTest == target)
        ax.scatter(XTest[indicesToKeep[0], 0],
                XTest[indicesToKeep[0], 1],
                               c = color,
                s = 50, 
                marker = 'x')
        #ax.legend(targetsNames)
    '''
    plt.axline((-4.7, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((-1, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((2, 0), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((3, 2), slope=-1, color="black", linestyle=(0, (5, 5)))
    plt.axline((0, 0), slope=1, color="black")
    '''
    ax.legend(targetsNames, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.grid()
    
def plot_function_supervised3(XTrain, YTrain, XTest, YTest):
    fig = plt.figure(figsize=(15,10))
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel( ' Component 1', fontsize = 15)
    ax.set_ylabel( ' Component 2', fontsize = 15)
    ax.set_zlabel( ' Component 3', fontsize = 15)
    ax.set_title('2 Variables, Deg = 3  ' , fontsize = 20)
    targets = [0, 1, 2, 3, 4]
    targetsNames = ['DCMTrain', 'DCMTest', 'DCPTrain', 'DCPTest','EtOHTrain',
    'EtOHTest', 'MeOHTrain', 'MeOHTest', 'WaterTrain', 'WaterTest']
    colors = ['r', 'g', 'b', 'c', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = np.where(YTrain == target)
        ax.scatter(XTrain[indicesToKeep[0], 0]
                   , XTrain[indicesToKeep[0],1],
                   XTrain[indicesToKeep[0],2]
                                      , c = color
                   , s = 50
                   , marker = 'o')
        indicesToKeep = np.where(YTest == target)
        ax.scatter(XTest[indicesToKeep[0], 0],
                XTest[indicesToKeep[0], 1],
                XTest[indicesToKeep[0], 2],
                               c = color,
                s = 50, 
                marker = 'x')
        #ax.legend(targetsNames)
    
    ax.legend(targetsNames, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.grid()