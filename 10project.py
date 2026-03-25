from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data=pd.read_csv("data/student.csv")
x=data[['Hours']] # it should 2d
y=data['Score']

model=LinearRegression()

model.fit(x,y)
predict_score=model.predict(x)

mae=mean_absolute_error(y,predict_score)
mse=mean_squared_error(y,predict_score)
rmse=np.sqrt(mse)

print("Mean Absolute ERROR (MAE): ", mae)
print("Mean Squared ERROR (MSE): ", mse)
print("Root Mean Sauared ERROR (PMSE). ",rmse)

new_prediction = model.predict([[7]])
print("Predicted Score for 7 hour", new_prediction)