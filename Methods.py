import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNC
  
def SPCA(data_matrix, test_matrix, L, number_of_components):
    """Supervised Principal Component Analysis
    algorithm that returns the reduced training
    and test matrices."""
    n = data_matrix.shape[0]
    H = np.eye(n,dtype=np.int8) - (1/n)*np.ones((n,n),dtype=np.int8)
    A = np.dot(np.dot(np.dot(np.dot(np.transpose(data_matrix), H), L), H), data_matrix) # data_matrix'*H*L*H*data_matrix
    w, v = LA.eig(A)
    idx = np.argsort(-w)
    coeff = v.real[:,idx]
    components = coeff[:,0:number_of_components]
    
    X_train_reduced = np.dot(data_matrix, components)
    X_test_reduced = np.dot(test_matrix, components)
    return X_train_reduced, X_test_reduced


def RPCA(data_matrix, test_matrix, L, number_of_components):
    """Regression Principal Component Analysis algorithm
    that returns the reduced training and test matrices."""
    n = data_matrix.shape[0]
    H = np.eye(n) - (1/n)*np.ones((n,n))
    A = np.dot(np.dot(np.dot(np.dot(np.transpose(data_matrix), H), L), H), data_matrix) # data_matrix'*H*L*H*data_matrix
    B = np.dot(np.transpose(data_matrix), data_matrix)        
    LA.cholesky(B)
    w, v = SLA.eig(A,B)
    w = np.real(w)
    idx = np.argsort(-w)
    coeff = v.real[:,idx]
    components = coeff[:,0:number_of_components]
    components,r = LA.qr(components)
        
    X_train_reduced = np.dot(data_matrix, components)
    X_test_reduced = np.dot(test_matrix, components)
    
    return X_train_reduced, X_test_reduced


def opt_param(XTrain, YTrain):
    '''This function is used to find the best k value for each seed for all methods
    and data-sets. It uses cross validation over the training data to avoid any
    contamination of the test data.
    It returns the optimal value for k.
    '''    
    tune_param = [
        {'n_neighbors' : np.linspace(1, 15, 15).astype(int), 'metric': ['minkowski']}
        ]  
    clf = GridSearchCV(KNC(), param_grid = tune_param, cv = 5)
    clf.fit(XTrain, YTrain)
    k = clf.best_params_
       
    return int(k['n_neighbors'])

    





    
    
    
    
    
    