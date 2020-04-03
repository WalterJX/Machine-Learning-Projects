import pandas as pd

def cutDataframe(df):#this function can separate the dataframe in to 3 dataframe, according to the outcome value
    i=0
    df = df.sort_values(by=['outcome(Cluster Index)'])
    while (df.iat[i,5]==0):
        i+=1
    df_0 = df.iloc[:i,:]
    j=i
    while(df.iat[i,5]==1):
        i+=1
    df_1 = df.iloc[j:i,:]
    df_2 = df.iloc[i:,:]
    return df_0, df_1, df_2

def calDisSquare(point_1, point_2): #calculate the distance's square
    dist = 0
    for i in range(1,5):
        num1 = point_1.iat[0,i]
        num2 = point_2.iat[0,i]
        dist += (num1 - num2) ** 2
    return dist

for m in range(0,20):#considering the initial cluster center may influence the result, run 20 times 
    df = pd.read_excel('Iris.xls',sheet_name='Sheet1')
    samples = df.sample(n=3) # randomly take 3 samples
    
    j=100000
    j_old=100001
    while(j_old-j>0.00001): #start to iterate
        j_old = j
        for i in range(0,df.shape[0]):
            min = 1000
            flag=-1
            for k in range(0,3):#assign each point to one of the cluster, according to the distance to each senter
                dist = calDisSquare(df.iloc[i:i+1,:],samples.iloc[k:k+1,:])
                if(dist<min):
                    min = dist
                    flag = k
            df.loc[i,'outcome(Cluster Index)'] = flag
        df_0, df_1, df_2 = cutDataframe(df)
    
        #update cluster centers
        df_0 = df_0.append(df_0.mean(),ignore_index=True)
        samplesNew = df_0.iloc[df_0.shape[0]-1:df_0.shape[0],:]
        df_0 = df_0.drop([df_0.shape[0]-1])
        
        samplesNew = samplesNew.append(df_1.mean(),ignore_index = True)
        samplesNew = samplesNew.append(df_2.mean().T,ignore_index = True)
        samples = samplesNew
        
        #calculate J
        sum = 0
        for i in range(0,df.shape[0]):
            k = df.iat[i,5]
            point = df.iloc[i:i+1,:]
            sample = samples.iloc[k:k+1,:]
            dist = calDisSquare(point, sample)
            sum += dist
        j = sum
        print(j)#show J in each iteration
    print('\n')#separate the results each time
        
    
        