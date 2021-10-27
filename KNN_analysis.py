import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
from Methods import SPCA,RPCA, opt_param
from sklearn.metrics import accuracy_score
from tabulate import tabulate as tb

from inputfile import path, testFraction, numberOfComponents, data_set, loop_size

split_seed = None

name_tosave = 'sheet' + str(data_set)

''' Start by reading the right excel tab and storage the data. We Also need to 
create a vector Y to classify the data in its different vapors.
'''

df = pd.read_excel(path, sheet_name=data_set)
spectraData = df.values

X = spectraData[:,2:spectraData.shape[1]]

range_control = [3,12,10][data_set]        
    
speciesCount = 0;
Y = np.empty([X.shape[1], 1])
for columnCount in range(0, X.shape[1]):
    i = np.remainder(columnCount, range_control)
    if (i < range_control):
        Y[columnCount] = speciesCount
    if (i == range_control - 1):
        speciesCount += 1
        
''' Now we need to treat the data. 
Regression methos are implemented such that matrix A has n rows of data samples 
and m colums of data features.
Therefore the matrix X here need to be transposed before the data is preprocessed. ''' 

X = np.transpose(X)
X = StandardScaler().fit_transform(X)

''' Now we need to split the data. I'm writing method 1 and 2. Method 1 simply uses the sklearn slipt function. 
Method 2 used a personalized function that breakes the data such that all different chemicals have at least one test data points.
'''
KNC_accuracy_pca = []; KNC_accuracy_lda = []; KNC_accuracy_kpca = []; 
KNC_accuracy_pls = []; KNC_accuracy_rpca = []; KNC_accuracy_spca = []; 


for i in range(loop_size):
    
    print(i)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=testFraction,
                        random_state = split_seed, stratify = np.reshape((Y), (1,-1))[0])
    numberOfTrainingPoints = XTrain.shape[0]
    
    ''' 
    LDA Implement
    '''
    totalDataPoints = X.shape[0]
    XNew = np.concatenate((XTrain, XTest)) 
    
    pcomp = [5, 20, 20][data_set]         


    pca = PCA(n_components=pcomp)
    XReduced = pca.fit_transform(XNew)
            
    XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
    XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];
            
    lda = LinearDiscriminantAnalysis(n_components = numberOfComponents, priors
                                     = None, shrinkage = None, solver = 'eigen')
    lda.fit(XReducedTrain,YTrain)
    XTrainLDA = lda.transform(XReducedTrain)
    XTestLDA = lda.transform(XReducedTest)
   
    k_val = opt_param(XTrainLDA,YTrain)
    neigh_lda = KNC(n_neighbors = k_val)
    neigh_lda.fit(XTrainLDA, YTrain)
    y_pred_lda1 = neigh_lda.predict(XTestLDA)

    KNC_accuracy_lda.append(accuracy_score(YTest, y_pred_lda1))
    
    
    ''' 
    pls 
    ''' 
    pls = PLSRegression(n_components=numberOfComponents)
    pls.fit(XTrain, YTrain)
    XTrainPLS = pls.transform(XTrain)
    XTestPLS = pls.transform(XTest)
    ###################################################
    
    k_val = opt_param(XTrainPLS,YTrain)
    neigh_pls = KNC(n_neighbors = k_val)
    neigh_pls.fit(XTrainPLS, YTrain)
    y_pred_pls1 = neigh_pls.predict(XTestPLS)
    
    KNC_accuracy_pls.append(accuracy_score(YTest, y_pred_pls1))

    ''' 
    pca 
    '''
    pca = PCA(n_components=numberOfComponents)
    principalComponents = pca.fit_transform(X)
    
    
    XTrainPCA, XTestPCA, YTrainPCA, YTestPCA = train_test_split(principalComponents,
                                Y, test_size=testFraction, stratify = np.reshape((Y), (1,-1))[0])

    ##########################################################
    k_val = opt_param(XTrainPCA,YTrainPCA)
    neigh_pca = KNC(n_neighbors = k_val)
    neigh_pca.fit(XTrainPCA, YTrainPCA)
    y_pred_pca1 = neigh_pca.predict(XTestPCA)

    KNC_accuracy_pca.append(accuracy_score(YTestPCA, y_pred_pca1))

    clf_pca2 = DecisionTreeClassifier(max_depth=4)
    clf_pca2.fit(XTrainPCA, YTrainPCA)
    y_pred_pca2 = clf_pca2.predict(XTestPCA)


    ''' 
    spca 
    '''
    numberOfTrainingPoints = XTrain.shape[0]
    L = np.zeros((numberOfTrainingPoints, numberOfTrainingPoints), dtype=np.int8)
    for data_index_row in range(numberOfTrainingPoints):
        for data_index_column in range(numberOfTrainingPoints):
            L[data_index_row, data_index_column] = np.equal(YTrain[data_index_row],YTrain[data_index_column])
    L= L + np.eye(numberOfTrainingPoints, dtype = np.int8)
    [XTrainSPCA, XTestSPCA] = SPCA(XTrain, XTest, L, numberOfComponents)
    ###################################################
    k_val = opt_param(XTrainSPCA,YTrain)
    neigh_spca = KNC(n_neighbors = k_val)
    neigh_spca.fit(XTrainSPCA, YTrain)
    y_pred_spca1 = neigh_spca.predict(XTestSPCA)

    KNC_accuracy_spca.append(accuracy_score(YTest, y_pred_spca1))


    ''' 
    rpca 
    '''
    totalDataPoints = X.shape[0]
    XNew = np.concatenate((XTrain,XTest))
    pca = PCA(n_components = pcomp)
    XReduced = pca.fit_transform(XNew)
    XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
    XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];
    [XTrainRPCA, XTestRPCA] = RPCA(XReducedTrain, XReducedTest, L, numberOfComponents)
    
    ###############################################################
    k_val = opt_param(XTrainRPCA,YTrain)
    neigh_rpca = KNC(n_neighbors = k_val)
    neigh_rpca.fit(XTrainRPCA, YTrain)
    y_pred_rpca1 = neigh_rpca.predict(XTestRPCA)

    KNC_accuracy_rpca.append(accuracy_score(YTest, y_pred_rpca1))

    ''' 
    kpca
    '''
    
    kpca = KernelPCA(n_components=numberOfComponents, kernel='rbf',
                     gamma=None, eigen_solver = 'arpack', max_iter = 6000)
    XKPCA = kpca.fit_transform(X) 

    XTrainKPCA, XTestKPCA, YTrainKPCA, YTestKPCA = train_test_split(XKPCA, Y,
                        test_size=testFraction, stratify = np.reshape((Y), (1,-1))[0])

    ###################################################
    k_val = opt_param(XTrainKPCA,YTrainKPCA)
    neigh_kpca = KNC(n_neighbors = k_val)
    neigh_kpca.fit(XTrainKPCA, YTrainKPCA)
    y_pred_kpca1 = neigh_kpca.predict(XTestKPCA)

    KNC_accuracy_kpca.append(accuracy_score(YTestKPCA, y_pred_kpca1))

############################ LOOP END ########################################

''' calculate the mean of the results '''
mean_KNC_accuracy_lda = np.mean(KNC_accuracy_lda)
mean_KNC_accuracy_pca = np.mean(KNC_accuracy_pca)
mean_KNC_accuracy_rpca = np.mean(KNC_accuracy_rpca)
mean_KNC_accuracy_spca = np.mean(KNC_accuracy_spca)
mean_KNC_accuracy_kpca = np.mean(KNC_accuracy_kpca)
mean_KNC_accuracy_pls = np.mean(KNC_accuracy_pls)


''' calculate the standard deviation of the results'''
std_KNC_accuracy_lda = np.std(KNC_accuracy_lda)
std_KNC_accuracy_pca = np.std(KNC_accuracy_pca)
std_KNC_accuracy_rpca = np.std(KNC_accuracy_rpca)
std_KNC_accuracy_spca = np.std(KNC_accuracy_spca)
std_KNC_accuracy_kpca = np.std(KNC_accuracy_kpca)
std_KNC_accuracy_pls = np.std(KNC_accuracy_pls)


''' calculate the max value''' 
max_KNC_accuracy_lda = max(KNC_accuracy_lda)
max_KNC_accuracy_pca = max(KNC_accuracy_pca)
max_KNC_accuracy_rpca = max(KNC_accuracy_rpca)
max_KNC_accuracy_spca = max(KNC_accuracy_spca)
max_KNC_accuracy_kpca = max(KNC_accuracy_kpca)
max_KNC_accuracy_pls = max(KNC_accuracy_pls)

''' calculate the min value '''
min_KNC_accuracy_lda = min(KNC_accuracy_lda)
min_KNC_accuracy_pca = min(KNC_accuracy_pca)
min_KNC_accuracy_rpca = min(KNC_accuracy_rpca)
min_KNC_accuracy_spca = min(KNC_accuracy_spca)
min_KNC_accuracy_kpca = min(KNC_accuracy_kpca)
min_KNC_accuracy_pls = min(KNC_accuracy_pls)

''' Combine Everything:'''
mean_KNC = [mean_KNC_accuracy_lda, mean_KNC_accuracy_pca, mean_KNC_accuracy_rpca, mean_KNC_accuracy_spca, mean_KNC_accuracy_kpca, mean_KNC_accuracy_pls]
std_KNC = [std_KNC_accuracy_lda, std_KNC_accuracy_pca, std_KNC_accuracy_rpca, std_KNC_accuracy_spca, std_KNC_accuracy_kpca, std_KNC_accuracy_pls]
max_KNC = [max_KNC_accuracy_lda, max_KNC_accuracy_pca, max_KNC_accuracy_rpca, max_KNC_accuracy_spca, max_KNC_accuracy_kpca, max_KNC_accuracy_pls]
min_KNC = [min_KNC_accuracy_lda, min_KNC_accuracy_pca, min_KNC_accuracy_rpca, min_KNC_accuracy_spca, min_KNC_accuracy_kpca, min_KNC_accuracy_pls]

xaxis =['LDA','PCA','LSR-PCA','SPCA','KPCA','PLS']

''' KNC plot '''
plt.figure()
plt.errorbar(xaxis, mean_KNC, yerr = std_KNC, linestyle = 'None', marker = 'o')
plt.title('KNearestNeighboor Accuracy')
plt.savefig(name_tosave + '_Accuracy' + '.eps')


tb_head = ['Method', 'Mean', 'STD', 'Max', 'Min'] 
tb_body = [
    ['LDA', mean_KNC[0], std_KNC[0], max_KNC[0],min_KNC[0]],
    ['LSR-PCA', mean_KNC[2], std_KNC[2], max_KNC[2],min_KNC[2]],
    ['PLS', mean_KNC[5], std_KNC[5], max_KNC[5],min_KNC[5]],
    ['SPCA', mean_KNC[3], std_KNC[3], max_KNC[3],min_KNC[3]],
    ['KPCA', mean_KNC[4], std_KNC[4], max_KNC[4],min_KNC[4]],
    ['PCA', mean_KNC[1], std_KNC[1], max_KNC[1],min_KNC[1]]
    ]
tbf = tb(tb_body, headers = tb_head, tablefmt="fancy_grid")
tbl = tb(tb_body, headers = tb_head, tablefmt="latex")
print(tbf)

