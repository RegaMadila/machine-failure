import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col='machine_failure', test_size=0.2, random_state=42):

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = X.drop(columns=[c for c in leakage_cols if c in X.columns], errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_categorical_indices(X):
    import numpy as np
    return np.where(X.dtypes != float)[0]
