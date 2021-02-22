import datetime
import math
import numpy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler


def calculate_entropy_purity(matrix_c):
    entropy = []
    puritym = []
    e_inner = 0
    for binc in range(matrix_c.shape[0]):
        for predictc in range(matrix_c.shape[0]):
            if matrix_c[binc][predictc] > 0:
                e_inner += -(matrix_c[binc][predictc] / sum(matrix_c[binc])) * math.log2(
                    (matrix_c[binc][predictc] / sum(matrix_c[binc])))
                # print('inner entropy', e_inner)
            else:
                e_inner += 0
        entropy.append(e_inner * sum(matrix_c[binc]) / sum(sum(matrix_c)))
        e_inner = 0
        puritym.append((max(matrix_c[binc]) / sum(matrix_c[binc])) * sum(matrix_c[binc]) / sum(sum(matrix_c)))
    return sum(entropy), sum(puritym)


def feature_extraction(df):
    features_extracted = pd.DataFrame()
    features_extracted['RootMean'] = df.apply(lambda r: np.sqrt((r ** 2).sum() / r.size), axis=1)
    list_ = [i for i in range(5, 25)]
    features_extracted['time_diff'] = df[list_].idxmax(axis=1, skipna=True) * 5 - 30
    features_extracted['cgm_diff'] = (df.max(axis=1, skipna=True)) - (df.min(axis=1, skipna=True))
    features_extracted['cgm_diffNorm'] = (df.max(axis=1, skipna=True)) - (df.min(axis=1, skipna=True)) / df.min(axis=1, skipna=True)
    features_extracted['std'] = df.std(axis=1)
    features_extracted['mean'] = df.mean(axis=1)
    return features_extracted


def find_SSE(X,labels,flag=0):
    df= pd.Series(labels)
    X_new =np.insert(X,2, df.to_numpy(), axis=1)
    clusters_p = numpy.unique(labels)
    if flag:
        clusters_p = clusters_p[1:]
    pos_mean=[np.mean(X_new[X_new[:,2] == i, 0:2], axis=0) for i in clusters_p]
    dist =0
    cid=0
    distance_sum = [(0,0)]
    for p in X_new:
        for idx in clusters_p: # Not considering noise points
            if p[2] == idx:
                cid = idx
                dist += (p[0] - pos_mean[idx][0]) ** 2 + (p[1] - pos_mean[idx][1]) ** 2
        distance_sum.append((cid,dist))
    sse_np = np.array(distance_sum)
    sse_mean=[np.mean(sse_np[sse_np[:,0]==i, 1:], axis=0) for i in clusters_p[1:]]
    return sum(sse_mean)/len(sse_mean)

insulin_data = pd.read_csv('InsulinData.csv', usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
insulin_data = insulin_data[insulin_data['BWZ Carb Input (grams)'] > 0]
insulin_data['Timestamp'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
insulin_data = insulin_data.rename(columns={"BWZ Carb Input (grams)": "CarbInput"})
insulin_data.drop(columns={'Time', 'Date'}, inplace=True)
insulin_data = insulin_data.sort_values('Timestamp')
insulin_data.reset_index(drop=True, inplace=True)

n = math.floor((insulin_data['CarbInput'].max() - insulin_data['CarbInput'].min()) / 20.0)

min_ = insulin_data['CarbInput'].min()
max_ = insulin_data['CarbInput'].max()

vals = [i for i in range(int(min_), int(max_), 20)]
conditions = [(insulin_data['CarbInput'] >= vals[0]) & (insulin_data['CarbInput'] < vals[1]),
              (insulin_data['CarbInput'] >= vals[1]) & (insulin_data['CarbInput'] < vals[2]),
              (insulin_data['CarbInput'] >= vals[2]) & (insulin_data['CarbInput'] < vals[3]),
              (insulin_data['CarbInput'] >= vals[3]) & (insulin_data['CarbInput'] < vals[4]),
              (insulin_data['CarbInput'] >= vals[4]) & (insulin_data['CarbInput'] < vals[5]),
              (insulin_data['CarbInput'] >= vals[5]) & (insulin_data['CarbInput'] <= max_)]

values = [0, 1, 2, 3, 4, 5]
insulin_data['ClusterId'] = np.select(conditions, values)

insulin_data['difference'] = insulin_data['Timestamp'] - insulin_data['Timestamp'].shift(1)

time_interval = datetime.timedelta(hours=2)
for_30min = datetime.timedelta(minutes=30)

insulin_data = insulin_data[insulin_data['difference'] >= time_interval]
timestamps_meal = insulin_data['Timestamp']

cgm_data = pd.read_csv('CGMData.csv', usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
cgm_data['Timestamp'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])

# Dropping unnecessary columns and cleaning data with less than 80% of entries in a day
cgm_data = cgm_data.drop(columns=['Date', 'Time'])
cgm_data.dropna(inplace=True)

meal_data = pd.DataFrame()
meal_data['dummy'] = [i for i in range(1, 31)]

counter = 0
for r in timestamps_meal:
    mask = (cgm_data['Timestamp'] > (r - for_30min)) & (cgm_data['Timestamp'] <= (r + time_interval))
    df = cgm_data['Sensor Glucose (mg/dL)'][mask].values
    # df.append(r)
    meal_data[counter] = pd.Series(df)
    counter += 1

meal_data = meal_data.drop(columns='dummy')
meal_data = meal_data.T
meal_afterna = meal_data.dropna(axis=0)

insulin_data.reset_index(drop=True, inplace=True)

# Merge insulin and feature data
Merged_df = insulin_data.merge(meal_afterna, how='inner', left_index=True, right_index=True)
Merged_df.reset_index(drop=True, inplace=True)

list_ = [i for i in range(30)]
features = feature_extraction(Merged_df[list_])
features.dropna(axis=0,inplace=True)

features['CarbInput'] = Merged_df['CarbInput']
y = Merged_df['ClusterId']

X = StandardScaler().fit_transform(features[['mean','CarbInput']])

kmeans =KMeans(init='k-means++', n_clusters=3 , random_state=3)
model = kmeans.fit(X)
label_k = kmeans.labels_

dbsc = DBSCAN(eps=0.23, min_samples=5).fit(X)
label_d =dbsc.labels_

kmeans_conf1 = confusion_matrix(y, model.labels_)
dbsc_con = confusion_matrix(y, dbsc.labels_)
metrics = []

metrics.append(find_SSE(X,label_k,0)[0])
metrics.append(calculate_entropy_purity(kmeans_conf1))
metrics.append(find_SSE(X,label_d,1)[0])
metrics.append(calculate_entropy_purity(dbsc_con[1:6,0:6]))

for_csv = metrics[0],metrics[2],metrics[1][0],metrics[3][0],metrics[1][1],metrics[3][1]
pd.DataFrame(for_csv).T.to_csv('Result.csv',index=False,header=False)
