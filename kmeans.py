#Importing necessary libraries

import pandas as pd
import numpy as np
from sklearn import datasets
import random
from sklearn import preprocessing


#Function to cluster df rows into k clusters

def cluster_kmeans(df, k):
    
    no_clusters = k
    len_col = len(df.columns)
    
    #Initialising k random points from index of dataframe as centroids
    init_centroids = random.sample(list(df.index), k=no_clusters)
    
    #Creating a list of centroid points. This will have a length of k
    #Each list item will have n elements where n is the number of attributes in df
    cent_list = []
    for x in init_centroids:
        cent_list.append(df.loc[x])
    
    #This will contain sse values of each iteration
    #Values will be appended after each loop below
    sse_list = []

    
    #While loop will run till the SSE condition inside is met
    while 1 > 0:
        
        #For each row, we calculate the dist between that row and each centroid in cent_list
        #Cluster for each row is allocated based on minimum euclidean distance
        for index, row in df.iterrows():
            distances = []
            for y in cent_list:
                dist = 0
                for x in range(0,len_col):
                    dist = dist + (row[x] - y[x])**2

                distances.append((dist)**(0.5))
            clust = int(distances.index(min(distances)))
            df.at[index,'cluster'] = clust

        #Calculating SSE
        sse = 0
        for index,rows in df.iterrows():
            val = rows['cluster']
            dist = 0
            for x in range(0,len_col):
                dist = dist + (rows[x] - cent_list[int(val)][x])**2

            sse = sse + dist
            
        #Condition to prevent error during first iteration
        if len(sse_list) > 0:
            
            #Stopping condition: Change in SSE value is less than 2%
            #Return df and SSE
            if abs(sse - sse_list[-1]) < 0.02*sse_list[-1]:
                df_new = pd.DataFrame(df['cluster'])
                df_new['cluster'] = df_new['cluster'].astype(int)
                return(df_new, sse)
                

        sse_list.append(sse)

        
        #Calculating new centroid for each cluster
        for i in range(0,k):
            df_temp = df.loc[df['cluster'] == i]
            cent_new = []
            for x in range(0,len_col):
                cent_new.append(df_temp.iloc[:,x].mean(axis=0))

            cent_list[i] = cent_new
            
            


#Testing the output on a sample dataset
iris = datasets.load_iris()
df1 = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'])


#Standardising values for Iris dataset
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df1)
df2 = pd.DataFrame(scaled_df, columns = df1.columns)

no_clusters = 4

#Testing function on standardised Iris dataset
cluster_kmeans(df2, no_clusters)
