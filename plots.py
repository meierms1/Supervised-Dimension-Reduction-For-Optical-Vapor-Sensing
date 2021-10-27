
from matplotlib import pyplot as plt
import numpy as np

marker_size = 120
edge_size = 3 #edge size for hollow markers
legend_font_size = 12
axis_font_size = 30

def plot_function_unsupervised(components, Y, method, n, data_set):
    if n > 3: n = 3
 
    heads = [['DMMP', 'DCP', 'EtOH', 'MeOH', 'Water'], 
             ['DCM', 'DCP', 'EtOH', 'MeOH', 'Water'],
             ['DCP','DMMP','EtOH','MeOH','Water']
             ][data_set]
        
    if n == 2: 
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_title(method, fontsize = 20)
        
        targets = range(len(heads))
        targetsNames = heads    
        
        colors = ['r', 'g', 'b', 'c', 'm']
       
        for target, color in zip(targets,colors):
            indicesToKeep = np.where(Y == target)
            ax.scatter(components[indicesToKeep[0], 0],
                       components[indicesToKeep[0], 1],
                       c = color,
                       s = marker_size)
            
    if n == 3: 
        fig = plt.figure(figsize=(12,12))       
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_zlabel('Component 3', fontsize = axis_font_size)
        ax.set_title(method, fontsize = 20)
        
        targets = range(len(heads))
        targetsNames = heads
 
        colors = ['r', 'g', 'b', 'c', 'm']

        for target, color in zip(targets,colors):
            indicesToKeep = np.where(Y == target)
            ax.scatter(components[indicesToKeep[0], 0],
                       components[indicesToKeep[0], 1],
                       components[indicesToKeep[0], 2],
                       c = color,
                       s = marker_size)
  
    ax.legend(targetsNames, loc='best', ncol=1, fontsize = legend_font_size) 
    ax.grid()
    fig.savefig(method +str(data_set)+ '.eps', format = 'eps')

   
    
def plot_function_supervised(XTrain, YTrain, XTest, YTest, method, n, data_set):
    if n > 3: n = 3
    
    heads = [['DMMP Train','DMMP Test', 'DCP Train','DCP Test', 'EtOH Train','EtOH Test', 
              'MeOH Train','MeOH Test', 'Water Train','Water Test'],
             ['DCM Train', 'DCM Test', 'DCP Train', 'DCP Test','EtOH Train',
              'EtOH Test', 'MeOH Train', 'MeOH Test', 'Water Train', 'Water Test'],
             ['DCP Train', 'DCP Test', 'DMMP Train', 'DMMP Test','EtOH Train',
              'EtOH Test', 'MeOH Train', 'MeOH Test', 'Water Train', 'Water Test']
             ][data_set]


    if n == 2: 
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_title(method, fontsize = 20)
        
        targets = range(len(heads))
        targetsNames = heads
    
        
        colors = ['r', 'g', 'b', 'c', 'm']
       
        for target, color in zip(targets,colors):
            indicesToKeep = np.where(YTrain == target)
            ax.scatter(XTrain[indicesToKeep[0], 0],
                       XTrain[indicesToKeep[0], 1],
                       c = color,
                       s = marker_size)
            indicesToKeep = np.where(YTest == target)
            ax.scatter(XTest[indicesToKeep[0], 0],
                       XTest[indicesToKeep[0], 1],
                       edgecolors = color,
                       facecolors = 'none',
                       linewidth = edge_size,
                       s = marker_size-edge_size*10, 
                       marker = 'o')
            
    elif n == 3:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_zlabel('Component 3' , fontsize = axis_font_size)       
        ax.set_title(method, fontsize = 20)
        
        targets = range(len(heads))
        targetsNames = heads

        colors = ['r', 'g', 'b', 'c', 'm']

        for target, color in zip(targets,colors):
            indicesToKeep = np.where(YTrain == target)
            ax.scatter(XTrain[indicesToKeep[0], 0],
                       XTrain[indicesToKeep[0], 1],
                       XTrain[indicesToKeep[0], 2],
                       c = color,
                       s = marker_size)
            indicesToKeep = np.where(YTest == target)
            ax.scatter(XTest[indicesToKeep[0], 0],
                       XTest[indicesToKeep[0], 1],
                       XTest[indicesToKeep[0], 2],
                       edgecolors = color,
                       facecolors = 'none',
                       linewidth = edge_size,
                       s = marker_size-edge_size*10, 
                       marker = 'o')
          
    ax.legend(targetsNames, loc='best', ncol=1, fontsize = legend_font_size) 
    ax.grid()
    fig.savefig(method +str(data_set) + '.eps', format = 'eps')
 
def plot_function_supervised7(XTrain, YTrain, XTest, YTest, method, data_set, n = 2):
    if n > 3: n = 3
    heads = ['DCM Train', 'DCM Test', 'DCP Train', 'DCP Test','EtOH Train',
              'EtOH Test', 'MeOH Train', 'MeOH Test', 'Water Train', 'Water Test']

    if n == 2: 
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_title(method, fontsize = 20)
        
        targets = range(2*len(heads))
        targetsNames = heads      
        colors = ['r','r','g','g','b','b','c','c','m','m'] 
              
        for target, color in zip(targets,colors):
            if target % 2 == 0:
                indicesToKeep = np.where(YTrain == target)
                ax.scatter(XTrain[indicesToKeep[0], 0],
                           XTrain[indicesToKeep[0], 1],
                           c = color,
                           s = marker_size,
                           marker = 'o')
                indicesToKeep = np.where(YTest == target)
                ax.scatter(XTest[indicesToKeep[0], 0],
                           XTest[indicesToKeep[0], 1],
                           edgecolors = color,
                           facecolors = 'none',
                           linewidth = edge_size,
                           s = marker_size-edge_size*10, 
                           marker = 'o')
            else:
                indicesToKeep = np.where(YTrain == target)
                ax.scatter(XTrain[indicesToKeep[0], 0],
                           XTrain[indicesToKeep[0], 1],
                           c = color,
                           s = marker_size,
                           marker = 's',
                           label = '_nolegend_')
                indicesToKeep = np.where(YTest == target)
                ax.scatter(XTest[indicesToKeep[0], 0],
                           XTest[indicesToKeep[0], 1],
                           edgecolors = color,
                           facecolors = 'none',
                           linewidth = edge_size,
                           s = marker_size-edge_size*10, 
                           marker = 's',
                           label = '_nolegend_')
            
    elif n == 3:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_zlabel('Component 3' , fontsize = axis_font_size)
        ax.set_title(method, fontsize = 20)
        
        targets = range(2*len(heads))
        targetsNames = heads

        colors = ['r','r','g','g','b','b','c','c','m','m']
        
        for target, color in zip(targets,colors):
            if target % 2 == 0:
                indicesToKeep = np.where(YTrain == target)
                ax.scatter(XTrain[indicesToKeep[0], 0],
                           XTrain[indicesToKeep[0], 1],
                           XTrain[indicesToKeep[0], 2],
                           c = color,
                           s = marker_size,
                           marker = 'o')
                indicesToKeep = np.where(YTest == target)
                ax.scatter(XTest[indicesToKeep[0], 0],
                           XTest[indicesToKeep[0], 1],
                           XTest[indicesToKeep[0], 2],
                           edgecolor = color,
                           facecolor = 'none',
                           linewidth = edge_size,
                           s = marker_size-edge_size*10, 
                           marker = 'o',
                           label = '_nolegend_')
            else:
                indicesToKeep = np.where(YTrain == target)
                ax.scatter(XTrain[indicesToKeep[0], 0],
                           XTrain[indicesToKeep[0], 1],
                           XTrain[indicesToKeep[0], 2],
                           c = color,
                           s = marker_size,
                           marker = 's')
                indicesToKeep = np.where(YTest == target)
                ax.scatter(XTest[indicesToKeep[0], 0],
                           XTest[indicesToKeep[0], 1],
                           XTest[indicesToKeep[0], 2],
                           edgecolor = color,
                           facecolor = 'none',
                           linewidth = edge_size,
                           s = marker_size-edge_size*10, 
                           marker = 's',
                           label = '_nolegend_'
                           )
                
    ax.legend(targetsNames, loc='best', ncol=1, fontsize = legend_font_size)
    ax.grid()
    fig.savefig(method +str(data_set) + '.eps', format = 'eps')

def plot_function_unsupervised7(components, Y, method, data_set, n = 2):
    if n > 3: n = 3
    heads = ['DCM', 'DCP','EtOH','MeOH', 'Water']
           
    if n == 2: 
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        
        targets = range(2*len(heads))
        targetsNames = heads
           
        colors = ['r','r','g','g','b','b','c','c','m','m']

        for target, color in zip(targets,colors):
            if target % 2 == 0:
                indicesToKeep = np.where(Y == target)
                ax.scatter(components[indicesToKeep[0], 0],
                           components[indicesToKeep[0], 1],
                           c = color,                   
                           s = marker_size,
                           marker = 'o')
            else:
                indicesToKeep = np.where(Y == target)
                ax.scatter(components[indicesToKeep[0], 0],
                           components[indicesToKeep[0], 1],
                           c = color,
                           s = marker_size,
                           marker = 's',
                           label = '_nolegend_'
                           )
         
    if n == 3: 
        fig = plt.figure(figsize=(12,12))       
        ax = fig.add_subplot(projection='3d')
        
        ax.set_xlabel('Component 1', fontsize = axis_font_size)
        ax.set_ylabel('Component 2', fontsize = axis_font_size)
        ax.set_zlabel('Component 3', fontsize = axis_font_size)
        
        targets = range(len(heads))
        targetsNames = heads
 
        colors = ['r','r','g','g','b','b','c','c','m','m']

        for target, color in zip(targets,colors):
            if target % 2 == 0:
                indicesToKeep = np.where(Y == target)
                ax.scatter(components[indicesToKeep[0], 0],
                           components[indicesToKeep[0], 1],
                           components[indicesToKeep[0], 2],
                           c = color,
                           s = marker_size)
            else:
                indicesToKeep = np.where(Y == target)
                ax.scatter(components[indicesToKeep[0], 0],
                           components[indicesToKeep[0], 1],
                           components[indicesToKeep[0], 2],
                           c = color,
                           s = marker_size,
                           marker = 's',
                           label = '_nolegend_')
  
    ax.legend(targetsNames, loc='best', ncol=1, fontsize = legend_font_size)
    ax.grid()
    fig.savefig(method +str(data_set)+ '.eps', format = 'eps')        
 