import numpy as np
import pandas as pd

class MyTools:
    def __init__(self, df):
        self.df = df

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return  np.sqrt(mse)
        
    # Calculate IQR (Interquartile Range)
    def get_outliers(self, feature=None):
        #Make sure all operations run correctly
        try:
            #Define the first quantile
            Q1 = self.df[feature].quantile(0.25)
            #Define the second quantile
            Q3 = self.df[feature].quantile(0.75)
            #get the InterQuantile range
            IQR = Q3 - Q1
            
            # Define outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            mask = (self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)
            # Identify outliers
            outliers = self.df[mask]
            
            return outliers, mask
        except Exception as e:
            print(f"Error occur when you input {e}")
        
    #Get the percentage of outlier    
    def outlier_pct(self, outlier_length=1):
        pct = round((outlier_length / len(self.df)) * 100, 2)
        print(f"Length of Dataset: {len(self.df)}")
        print(f"Length of Outlier: {outlier_length}")
        print(f"The outlier is {pct}% of the dataset")
        pass
        