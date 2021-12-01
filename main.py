'''
@author: Maycon Meier
'''
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
from Methods import SPCA,RPCA
from plots import plot_function_unsupervised as pfu, plot_function_supervised as pfs
from plots import plot_function_unsupervised7 as pfu7, plot_function_supervised7 as pfs7

''' All changeble parameters are listed and described in the inputfile.py; 
    No changes should be made in the main file. '''
    
from inputfile import testFraction,numberOfComponents, data_set, split_seed, path


''' import the data from the .xlsx file '''
data_set = data_set - 1
#df = pd.read_excel(path, sheet_name=data_set) #Use this for .xlsx file
df = pd.read_csv(path)
spectraData = df.values

X = spectraData[:,2:spectraData.shape[1]]

# Range control is used to discriminated between the different amount of 
# data-points per vapor in each data set. It controls the target matrix.

range_control = [3,12,10][data_set] 

''' Create the target matrix '''
speciesCount = 0;
Y = np.empty([X.shape[1], 1])
for columnCount in range(0, X.shape[1]):
    i = np.remainder(columnCount, range_control)
    if (i < range_control):
        Y[columnCount] = speciesCount
    if (i == range_control - 1):
        speciesCount += 1
        
if data_set == 2:
    r2 = 5
    speciesCount = 0;
    Y2 = np.empty([X.shape[1], 1])
    for columnCount in range(0, X.shape[1]):
        i = np.remainder(columnCount, r2)
        if (i < r2):
            Y2[columnCount] = speciesCount
        if (i == r2 - 1):
            speciesCount += 1


''' Preprocessing of the data matrix'''
X = np.transpose(X)
X = StandardScaler().fit_transform(X)


############################# Supervised Method ###############################

''' Train and test split of the data using scikit learn'''

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=testFraction,
            random_state = split_seed, stratify = np.reshape((Y), (1,-1))[0])
if data_set == 2:
    _, _,YTrain2, YTest2 = train_test_split(X, Y2, test_size=testFraction, 
            random_state = split_seed, stratify = np.reshape((Y), (1,-1))[0])


'''  LDA Implement '''

numberOfTrainingPoints = XTrain.shape[0]     
    
totalDataPoints = X.shape[0]
XNew = np.concatenate((XTrain, XTest)) 

pcomp = [5,20,20][data_set]
        

pca = PCA(n_components=pcomp)
XReduced = pca.fit_transform(XNew)
            
XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];

lda = LinearDiscriminantAnalysis(n_components = numberOfComponents, priors
                                     = None, shrinkage = None, solver = 'eigen')
lda.fit(XReducedTrain,YTrain)
print('LDA Variance:',lda.explained_variance_ratio_)
XTrainLDA = lda.transform(XReducedTrain)
XTestLDA = lda.transform(XReducedTest)   

if data_set == 2:
    pfs7(XTrain=XTrainLDA, YTrain=YTrain2, XTest = XTestLDA, YTest=YTest2, 
         method='LDA' , n=numberOfComponents, data_set=data_set)
else:
    pfs(XTrain=XTrainLDA, YTrain=YTrain, XTest = XTestLDA, YTest=YTest, 
        method='LDA' , n=numberOfComponents, data_set=data_set)



'''  PLS Implement ''' 

pls = PLSRegression(n_components=numberOfComponents)
pls.fit(XTrain, YTrain)
XTrainPLS = pls.transform(XTrain)
XTestPLS = pls.transform(XTest)

if data_set == 2:
    pfs7(XTrain=XTrainPLS, YTrain=YTrain2, XTest = XTestPLS, YTest=YTest2, 
         method='PLS', n=numberOfComponents, data_set=data_set)
else:
    pfs(XTrain=XTrainPLS, YTrain=YTrain, XTest = XTestPLS, YTest=YTest, 
        method='PLS', n=numberOfComponents, data_set=data_set)


''' SPCA Implement '''

numberOfTrainingPoints = XTrain.shape[0]
L = np.zeros((numberOfTrainingPoints, numberOfTrainingPoints), dtype=np.int8)
for data_index_row in range(numberOfTrainingPoints):
    for data_index_column in range(numberOfTrainingPoints):
        L[data_index_row, data_index_column] = np.equal(YTrain[data_index_row],
                                                        YTrain[data_index_column])
L= L + np.eye(numberOfTrainingPoints, dtype = np.int8)
[XTrainSPCA, XTestSPCA] = SPCA(XTrain, XTest, L, numberOfComponents)

if data_set == 2:
    pfs7(XTrain=XTrainSPCA, YTrain=YTrain2, XTest = XTestSPCA, YTest=YTest2,
         method='SPCA' , n=numberOfComponents, data_set=data_set)
else:
    pfs(XTrain=XTrainSPCA, YTrain=YTrain, XTest = XTestSPCA, YTest=YTest, 
        method='SPCA' , n=numberOfComponents, data_set=data_set)


''' LSR-PCA Implement  '''

totalDataPoints = X.shape[0]
XNew = np.concatenate((XTrain,XTest))
pca = PCA(n_components = pcomp)
XReduced = pca.fit_transform(XNew)
XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];
[XTrainRPCA, XTestRPCA] = RPCA(XReducedTrain, XReducedTest, L, numberOfComponents)

if data_set == 2:
    pfs7(XTrain=XTrainRPCA, YTrain=YTrain2, XTest = XTestRPCA, YTest=YTest2,
         method='LSR-PCA' , n=numberOfComponents, data_set=data_set)   
else:
    pfs(XTrain=XTrainRPCA, YTrain=YTrain, XTest = XTestRPCA, YTest=YTest, 
        method='LSR-PCA' , n=numberOfComponents, data_set=data_set)



########################### Unsupervised Method ###############################

'''  PCA Implement '''

pca = PCA(n_components=numberOfComponents)
principalComponents = pca.fit_transform(X)
print('PCA Variance:', pca.explained_variance_ratio_)

if data_set == 2:
    pfu7(principalComponents, Y2, method='PCA' , 
         n=numberOfComponents, data_set=data_set)
else:
    pfu(principalComponents, Y, method='PCA' , 
        n=numberOfComponents, data_set=data_set)



''' Kernel PCA Implement '''
    
kpca = KernelPCA(n_components=numberOfComponents, kernel='rbf',
                     gamma=None, eigen_solver = 'arpack', max_iter = 6000)
XKPCA = kpca.fit_transform(X) 

if data_set == 2:
    pfu7(XKPCA, Y2, method='KPCA', n=numberOfComponents, data_set=data_set)
else:
    pfu(XKPCA, Y, method='KPCA', n=numberOfComponents, data_set=data_set)
