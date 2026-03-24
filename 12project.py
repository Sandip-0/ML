import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data=pd.read_csv('suicide.csv')

model=LinearRegression()
le=LabelEncoder()


data['country']=le.fit_transform(data['country'])
data['sex']=le.fit_transform(data['sex'])
data['age_group']=le.fit_transform(data['age_group'])

x=data[['country','sex','age_group','year']]
y=data['suicide_rate']

model.fit(x,y)
predict_score=model.predict(x)


mae=mean_absolute_error(y,predict_score)
mse=mean_squared_error(y,predict_score)
r2=r2_score(y,predict_score)
rmse=np.sqrt(mse)


print("Mean Absolute ERROR (MAE): ", round(mae,2))
print("Mean Squared ERROR (MSE): ", round(mse,2))
print("Root Mean Sauared ERROR (PMSE). ",round(rmse,2))
print("R^2 Score (model acurecy): ",round(r2,4)) # closer to 1 better


