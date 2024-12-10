import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data_train = np.array([
    [32134,290475,48776,14861,2777,32393,72980,22226,307602,242706,4135592,20103,31434,17598,81127,3703,1117565,1684351],
    [29888,213291,64733,8029,3543,0,57799,1201,323635,372573,5049694,14957,26345,15176,65393,6045,945863,1571497],
    [1612,155582,66136,6655,1625,0,45209,18580,390958,308901,4377991,12675,23587,20522,81587,4548,909309,1475917],
    [26132,52913,73207,9549,3830,0,108083,24539,261523,375459,4460637,22761,53551,54185,92965,8411,1061444,1813744],
    [18121,55558,77127,39945,2409,0,85434,24813,281591,372044,4231197,106304,41595,35767,92425,7458,1040050,1546896],
    [6034,110205,77818,36450,1491,0,59699,17431,268681,369933,4062671,105334,39141,15756,67707,6756,1237864,1249338]])

data_train = data_train.transpose()
  
df_train = pd.DataFrame(
    data_train,
    columns=["2004", "2005", "2006", "2007", "2008", "2009"]
)


df_train_min_max_scaled = df_train.copy()  # Salin DataFrame asli

# Aplikasikan normalisasi Min-Max Scaling
for column in df_train_min_max_scaled.columns:
    df_train_min_max_scaled[column] = (((df_train_min_max_scaled[column] - df_train_min_max_scaled[column].min()) * 0.8) / (df_train_min_max_scaled[column].max() - df_train_min_max_scaled[column].min())) + 0.1
    
print("DATA TRAINING : \n",df_train_min_max_scaled)
# Menampilkan DataFrame
data_testing = np.array([
    [5853, 160989, 82063, 36109, 525, 0, 45307, 21340, 250500, 342897, 4385510, 100886, 35449, 7222, 31583, 4822, 960548, 869666],  
    [6380, 0, 99244, 0, 0, 0, 48094, 0, 209827, 168467, 4091990, 3657, 22337, 0, 35256, 246, 948357, 739554],                       
    [4707, 0, 112141, 5122, 0, 30, 41466, 16, 233432, 255099, 3160592, 6675, 18460, 802, 28400, 2197, 849772, 623201],              
    [453, 139521, 44244, 6098, 0, 14, 17959, 70, 31878, 272201, 3082766, 5659, 8904, 292, 0, 2019, 753926, 486877],                 
    [453, 110144, 71864, 18223, 0, 14, 23307, 471, 22486, 262685, 3742865, 9775, 10566, 2585, 0, 1317, 651929, 518357],             
    [2173, 14165, 112541, 15145, 0, 35, 34300, 777, 78972, 327803, 4169157, 11238, 16272, 5159, 0, 3324, 513774, 574545]])

data_testing = data_testing.transpose()
    
df_testing = pd.DataFrame(
    data_testing,
    columns=["2010", "2011", "2012", "2013", "2014","2015"]
)

df_testing_min_max_scaled = df_testing.copy()  # Salin DataFrame asli

# Aplikasikan normalisasi Min-Max Scaling
for column in df_testing_min_max_scaled.columns:
    df_testing_min_max_scaled[column] = (((df_testing_min_max_scaled[column] - df_testing_min_max_scaled[column].min()) * 0.8) / (df_testing_min_max_scaled[column].max() - df_testing_min_max_scaled[column].min())) + 0.1
    
print("DATA TESTING : \n",df_testing_min_max_scaled)