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





# # Pre-allocate memory for predictions
# y_pred_wfv = pd.Series(np.empty(len(y_test)), index=y_test.index, dtype=float)
# history = y_train.copy()

# for i in range(len(y_test)):
#     # Fit the ARIMA model
#     model = ARIMA(history, order=(3, 0, 0)).fit()
    
#     # Forecast the next value
#     next_pred = model.forecast()
    
#     # Store the prediction
#     y_pred_wfv.iloc[i] = next_pred.iloc[0]
    
#     # Append the true value from y_test to history using pd.concat
#     history = pd.concat([history, y_test.iloc[[i]]])

# # Display the predictions
# print(y_pred_wfv)



    
# #Update of the walk forward validation for garch model
# # Create empty list to hold predictions
# predictions = []

# # Calculate size of test data (20%)
# test_size = len(y_test)

# # Walk forward
# for i in range(test_size):
#     # Create test data
#     y_train_wf = y_train.iloc[: -(test_size - i)]

#     # Train model
#     model_new = ARIMA(y_train_wf, order=(3, 0, 0)).fit()

#     # Generate next prediction (volatility, not variance)
#     next_pred = model_new.predict()

#     # Append prediction to list
#     predictions.append(next_pred)

# # Create Series from predictions list
# y_test_wfv = pd.Series(predictions, index=y_train.tail(test_size).index)

# print("y_test_wfv type:", type(y_test_wfv))
# y_test_wfv.head()
        