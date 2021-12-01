"""
@author: Maycon Meier
"""

''' User input parameters


testFraction: This variable is used to set the amount of test points when using 
            sklearn split function; Dataset 1 requires a minimum of 30% test points;
            Datasets 2 and 3 can run with 20% test points.

numberOfComponents: The number of components returned from the dimension reduction 
            methods and the dimension of the plots. If it is set to a value higher
            than 3, plots will show the 3 dimension plots. 
            
            Note: Number of components for LDA is limited to n-1, where n is the
            number of classes in the data set.
             
data_set: Assign 1 for Data-set 1; 
          Assign 2 for data-set 2; 
          Assign 3 for data-set 3.
          
loop_size: Number of random seed interaction for the KNN Accuracy Analysis code.
    
split_seed: to allow reproducibility of the results, the split data can be 
            controlled with a non-random seed.


path: local where the .csv data file is stored. To run the .xlsx file instead of 
      the .csv file, comment line 26 in the main.py file. 

''' 

testFraction = 0.2 # Test fraction has to be >= the number of classes. [0.3, 0.2, 0.2]
numberOfComponents = 2
data_set = 1 #Choose the dataset [1 to 3]
loop_size = 50 
split_seed = 86  # [92,86,96]

path = r"Dataset%d.csv"%data_set


