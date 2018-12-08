---
title: Models
notebook: LC_Models.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

import warnings
warnings.filterwarnings('ignore')

% matplotlib inline
```


## Data Preparation

### Stratified Sampling
We split training  and test dataset by stratifing on the loan grade.



```python
df_clean = pd.read_csv("data/df_clean.csv")
```




```python
df_train, df_test = train_test_split(df_clean, 
                                     test_size = .2, 
                                     stratify = df_clean['grade'], 
                                     random_state=90)

print("Shape of the training set: {}".format(df_train.shape))
print("Shape of the training set: {}".format(df_test.shape))
```


    Shape of the training set: (1704488, 89)
    Shape of the training set: (426122, 89)
    



```python
def split_columns(df, target_col, drop_columns):
    # Get the response variable
    y_train = df[[target_col]]

    # Drop the required columns
    X_train = df.drop(drop_columns, axis=1)
    
    return X_train, y_train
```




```python
X_train, y_train = split_columns(df_train, target_col='response', drop_columns=['response'])
X_test, y_test = split_columns(df_test, target_col='response', drop_columns=['response'])
```


### Standardization
We standardize all the predictors that are not dummy variables. 



```python
def scale_datasets(train_data, test_data, cols_to_scale):
    """
    This function will be used to standardize columns in your datasets. It
    also allows you to pass in a test dataset, which will be standardized
    using the stats from the training data. 
    
    :param: train: The training dataset
    :param: test: The test dataset, which will be standardized using stats 
                  from the training data. 
    :param: cols_to_scale: List containing the column names to be standardized
    :return: (DataFrame, DataFrame) Standardized test and training DataFrames
    """
    
    train = train_data.copy()
    test = test_data.copy()
    
    # Fit the scaler on the training data
    scaler = StandardScaler().fit(train[cols_to_scale])

    # Scale both the test and training data. 
    train[cols_to_scale] = scaler.transform(train[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])
    
    return train, test
```




```python
cols_to_scale = ["loan_amnt"                     
                , "term"                          
                , "int_rate"                      
                , "grade"                         
                , "emp_length"                    
                , "annual_inc"                    
                , "dti"                           
                , "delinq_2yrs"                   
                , "earliest_cr_line"              
                , "inq_last_6mths"                
                , "open_acc"                      
                , "pub_rec"                       
                , "revol_util"                                  
                , "acc_now_delinq"                
                , "tot_coll_amt"                  
                , "tot_cur_bal"                                         
                , "fico"] 
X_train, X_test = scale_datasets(X_train, X_test, cols_to_scale)
```


## Classification of Good and Bad Loans

### Logistic Regression 

### kNN 

### LDA

### QDA

### SVM

### Single Decision Tree

### Random Forest

### AddBoost

### Neural Network
