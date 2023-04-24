import numpy as np
import pandas as pd
sample= [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_zscore(sample)
print("Outliers from Z-scores method: ", sample_outliers)

Output": 
Outliers from Z-scores method:  [101]

titanic_df = pd.read_csv('train.csv')
titanic_outlier = detect_outliers_zscore(titanic_df['Age'])
print("Outliers from Z-scores method: ", titanic_outlier)

OUTPUT:
Outliers from Z-scores method:  [101, 80.0, 74.0]

























# import numpy as np
# sample= [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
# outliers = []
# def detect_outliers_zscore(data):
#     thres = 3
#     mean = np.mean(data)
#     std = np.std(data)
#     # print(mean, std)
#     for i in data:
#         z_score = (i-mean)/std
#         if (np.abs(z_score) > thres):
#             outliers.append(i)
#     return outliers# Driver code
# sample_outliers = detect_outliers_zscore(sample)
# print("Outliers from Z-scores method: ", sample_outliers)
