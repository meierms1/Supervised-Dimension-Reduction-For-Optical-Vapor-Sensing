# Supervised Dimension Reduction for Optical Vapor Sensing

A comprehensive Python implementation comparing supervised and unsupervised dimension reduction methods for optical vapor sensing applications. This project evaluates the performance of various dimensionality reduction techniques on spectral data for chemical vapor classification.

## Overview

This project implements and compares multiple dimension reduction algorithms for analyzing optical spectra of chemical vapors. The system processes spectral data from optical sensors and applies various machine learning techniques to reduce dimensionality while preserving classification accuracy.

### Key Features

- **Multiple Dimension Reduction Methods**: Implementation of both supervised and unsupervised techniques
- **Comparative Analysis**: Performance evaluation across different datasets and methods
- **Visualization Tools**: 2D and 3D plotting capabilities for dimension-reduced data
- **Cross-validation**: K-nearest neighbors analysis with optimized parameters
- **Flexible Configuration**: Easy parameter adjustment through configuration files

## Datasets

The project includes three spectral datasets containing measurements of different chemical vapors:

### Dataset 1
- **Vapors**: DMMP, DCP, EtOH, MeOH, Water
- **Concentrations**: 0.15P, 0.25P, 0.50P for each vapor
- **Spectral Range**: 449-699 nm (1008 wavelength points)
- **Data Points per Vapor**: 3 concentration levels

### Dataset 2  
- **Vapors**: DCM, DCP, EtOH, MeOH, Water
- **Additional grouping**: 5 data points per vapor group
- **Similar spectral characteristics to Dataset 1**

### Dataset 3
- **Vapors**: DCP, DMMP, EtOH, MeOH, Water
- **Data Points per Vapor**: 10 measurements
- **Extended measurement set for improved statistical analysis**

## Methods Implemented

### Supervised Methods

1. **Linear Discriminant Analysis (LDA)**
   - Maximizes class separability
   - Reduces to n-1 dimensions (where n = number of classes)
   - Includes PCA preprocessing for high-dimensional data

2. **Partial Least Squares (PLS) Regression**
   - Finds directions that maximize covariance between features and targets
   - Suitable for high-dimensional, correlated data

3. **Supervised Principal Component Analysis (SPCA)**
   - Custom implementation incorporating class information
   - Uses similarity matrix L based on class labels
   - Optimizes for within-class similarity

4. **Least Squares Regression PCA (LSR-PCA)**
   - Combines PCA preprocessing with regression-based dimension reduction
   - Generalized eigenvalue problem approach
   - Enhanced class separation through supervised learning

### Unsupervised Methods

1. **Principal Component Analysis (PCA)**
   - Standard dimensionality reduction preserving maximum variance
   - Baseline comparison method

2. **Kernel PCA (KPCA)**
   - Non-linear dimension reduction using RBF kernel
   - Captures non-linear relationships in spectral data

## Technical Implementation

### Core Architecture

```
main.py              # Primary execution script
├── inputfile.py     # Configuration parameters
├── Methods.py       # Custom algorithm implementations (SPCA, RPCA)
├── plots.py         # Visualization functions
└── KNN_analysis.py  # Classification performance evaluation
```

### Algorithm Details

#### SPCA Implementation
```python
def SPCA(data_matrix, test_matrix, L, number_of_components):
    """
    Supervised Principal Component Analysis
    
    Parameters:
    - data_matrix: Training data (n_samples × n_features)
    - test_matrix: Test data 
    - L: Similarity matrix based on class labels
    - number_of_components: Target dimensionality
    
    Returns:
    - X_train_reduced, X_test_reduced: Transformed datasets
    """
```

The algorithm constructs matrix A = X^T H L H X where:
- H: Centering matrix (I - (1/n)11^T)
- L: Class similarity matrix
- X: Data matrix

#### LSR-PCA Implementation
Uses generalized eigenvalue decomposition to solve:
Av = λBv, where A incorporates class information and B = X^T X

### Data Preprocessing

1. **Standardization**: Z-score normalization using sklearn's StandardScaler
2. **Train-Test Split**: Stratified splitting to maintain class proportions
3. **PCA Preprocessing**: Applied before LDA and LSR-PCA for computational efficiency

### Visualization

The project includes comprehensive plotting functions:

- **2D Scatter Plots**: For 2-component reductions
- **3D Visualizations**: For 3+ component reductions  
- **Color-coded Classes**: Distinct colors for each vapor type
- **Method Comparison**: Side-by-side plots for different algorithms

## Configuration

### Input Parameters (`inputfile.py`)

```python
testFraction = 0.2          # Test set proportion (0.2-0.3 recommended)
numberOfComponents = 2       # Target dimensionality (≤ n_classes-1 for LDA)
data_set = 1                # Dataset selection (1, 2, or 3)
loop_size = 50              # Iterations for KNN analysis
split_seed = 86             # Reproducibility seed
path = "Dataset1.csv"       # Data file path
```

### Dataset-Specific Settings

- **Dataset 1**: Minimum 30% test fraction required
- **Datasets 2 & 3**: 20% test fraction acceptable
- **Component Limits**: LDA limited to n_classes-1 components
- **Preprocessing Components**: [5, 20, 20] for datasets [1, 2, 3]

## Performance Evaluation

### K-Nearest Neighbors Analysis

The `KNN_analysis.py` script performs comprehensive performance evaluation:

1. **Cross-validation**: Grid search for optimal k values
2. **Multiple Random Seeds**: Statistical significance testing
3. **Accuracy Metrics**: Classification accuracy for each method
4. **Comparative Analysis**: Performance ranking across methods

### Metrics Reported

- **Explained Variance Ratio**: For PCA and LDA
- **Classification Accuracy**: KNN performance on test sets
- **Statistical Significance**: Multi-seed evaluation results

## Usage

### Basic Execution

1. **Configure Parameters**: Edit `inputfile.py` with desired settings
2. **Run Main Analysis**: 
   ```bash
   python main.py
   ```
3. **Performance Evaluation**: 
   ```bash
   python KNN_analysis.py
   ```

### Expected Outputs

- **Visualization Plots**: 2D/3D scatter plots for each method
- **Console Output**: Variance ratios and performance metrics
- **Comparative Results**: Method performance rankings

## Dependencies

```python
numpy >= 1.19.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
scipy >= 1.7.0
tabulate >= 0.8.0
```

## File Structure

```
├── README.md              # Project documentation
├── main.py               # Primary analysis script
├── inputfile.py          # Configuration parameters
├── Methods.py            # Custom algorithm implementations
├── plots.py              # Visualization functions
├── KNN_analysis.py       # Performance evaluation
├── DATA.xlsx            # Original Excel data file
├── Dataset1.csv         # CSV export of Dataset 1
├── Dataset2.csv         # CSV export of Dataset 2
└── Dataset3.csv         # CSV export of Dataset 3
```

## Research Context

This implementation supports research in:

- **Chemical Sensing**: Optical detection of volatile organic compounds
- **Spectral Analysis**: Dimensionality reduction for hyperspectral data
- **Machine Learning**: Supervised vs. unsupervised learning comparison
- **Sensor Applications**: Portable chemical detection systems

## Author

**Maycon Meier**

## License

This project is available for research and educational purposes. Please cite appropriately if used in academic work.

## Contributing

Contributions are welcome! Please ensure:

1. Code follows existing style conventions
2. New methods include appropriate documentation
3. Test cases are provided for new algorithms
4. Performance comparisons are updated accordingly

## Future Enhancements

- Implementation of additional supervised methods (e.g., Fisher's LDA variants)
- Deep learning approaches for non-linear dimension reduction
- Real-time processing capabilities for sensor applications
- Extended evaluation metrics beyond classification accuracy
