import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

data={
    'studyhour':[1,2,3,4,5],
    'testscore':[40,50,60,70,80]
}
df=pd.DataFrame(data)
standared_scaler=StandardScaler()
standared_scaled=standared_scaler.fit_transform(df)

# print(standared_scaled)

print('StandardScaler output')
print(pd.DataFrame(standared_scaled,columns=['studyhour','testscore']))

minmax_scaler=MinMaxScaler()
minmax_scaled=minmax_scaler.fit_transform(df)

print('MinMaxScaler output')
print(pd.DataFrame(minmax_scaled,columns=['studyhour','testscore']))

x=df[['studyhour']]
y=df[['testscore']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print('train data')
print(x_train)

print('test data')
print(x_test)

print('train data')
print(y_train)

print('test data')
print(y_test)
