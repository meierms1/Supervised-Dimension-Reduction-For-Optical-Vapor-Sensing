"""
@author: Maycon Meier
"""

''' User input parameters


testFraction: This variable is used to set the amount of test points when using 
            sklearn split function;

numberOfComponents: The number of components returned from the dimension reduction 
            methods and the dimension of the plots. If it is set to a value higher
            than 3, plots will show the 3 dimension plots. 
            
            Note: Number of components for LDA is limited to n-1, where n is the
            number of classes in the data set.
             
data_set: Assign 0 for Data-set 1; 
          Assign 1 for data-set 2; 
          Assign 2 for data-set 3.
          
loop_size: Number of random seed interaction for the KNN Accuracy Analysis code.
    
split_seed: to allow reproducibility of the results, the split data can be 
            controlled with a non-random seed.


path: local where the .xlsx data file is stored. 

''' 

testFraction = 0.2
numberOfComponents = 2
data_set = 1 #Choose the tab you want to read from the xlsx data file [0 to 2]
loop_size = 50 
split_seed = 86  # [92,86,96]

path =  r"/Users/maycon/Dropbox/UCCS/Clean/DATA.xlsx"



