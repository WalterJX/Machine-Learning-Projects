import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle

total_amount=768
sample_amount = int(input("Please input n: "))
test_amount = total_amount-sample_amount*2

df = pd.read_csv('pima-indians-diabetes.csv',header = None);
df.columns = ["attr1","attr2","attr3","attr4","attr5","attr6","attr7","attr8","diabetes"]
df = df.sort_values(by=['diabetes'])#we need same numbers of samples from both healthy and diabetes people, so first we need to separate them
temp=0
while (df.iat[temp,8]!=1):
    temp+=1
df_healthy = df.iloc[:temp,:]
df_diabetes = df.iloc[temp:total_amount,:]

sum=0
for j in range(0,1000):
    df_healthy = shuffle(df_healthy)
    df_daibetes = shuffle(df_diabetes)
    train_X1 = df_healthy.iloc[:sample_amount,0:8]
    train_X2 = df_diabetes.iloc[:sample_amount,0:8]
    train_X = pd.concat([train_X1, train_X2]) #combine the samples from two dataframe as train samples
    train_Y1 = df_healthy.iloc[:sample_amount,8:9]
    train_Y2 = df_diabetes.iloc[:sample_amount,8:9]
    train_Y = pd.concat([train_Y1,train_Y2]) # combine
    test_X = pd.concat([df_healthy.iloc[sample_amount:,0:8],df_diabetes.iloc[sample_amount:,0:8]])
    test_Y = pd.concat([df_healthy.iloc[sample_amount:,8:9],df_diabetes.iloc[sample_amount:,8:9]])
    
    regr_model = linear_model.LinearRegression()
    regr_model.fit(train_X,train_Y) # fit model
    correct_amount=0
    
    pre = regr_model.predict(test_X)
    correct = 0;
    for i in range(0,test_amount):#check whether the predictions are right, calculate the accuracy
        res = -1;
        if(pre[i:i+1]>=0.5):
            res = 1
        else:
            res=0
        if(res==test_Y.iat[i,0]):
            correct+=1
    #print(correct/test_amount)
    sum += correct/test_amount
accuracy = sum/1000 # average accuracy in 1000 independent experiments
print ('when n is '+str(sample_amount)+', average accuracy is '+str(accuracy)) # print the result

        
    