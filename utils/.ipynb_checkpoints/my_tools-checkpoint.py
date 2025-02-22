import numpy as np
import pandas as pd

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

# Calculate IQR (Interquartile Range)
def get_outliers(df, feature=None):
    #Make sure all operations run correctly
    try:
        #Define the first quantile
        Q1 = df[feature].quantile(0.25)
        #Define the second quantile
        Q3 = df[feature].quantile(0.75)
        #get the InterQuantile range
        IQR = Q3 - Q1
        
        # Define outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        return outliers
    except Exception as e:
        print(f"Error occur when you input {e}")


def outlier_pct(data_length=1, outlier_length=1):
    pct = round((outlier_length / data_length) * 100, 2)
    print(f"Length of Dataset: {data_length}")
    print(f"Length of Outlier: {outlier_length}")
    print(f"The outlier is {pct}% of the dataset")
    pass