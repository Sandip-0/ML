import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

data=pd.read_csv("data/Student_Performance.csv")
model=LinearRegression()

x=data[['study_hours']]
y=data['overall_score']

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

# #histogram
plt.figure(figsize=(10,6)) # width , hight
plt.hist(data["overall_score"],bins=30,color='skyblue',edgecolor='black')
plt.title("Distribution of FINAL EXAM SCORES")
plt.xlabel("Final Exam Score")
plt.ylabel("Number of Students")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# Scatter + Regression line 
plt.figure(figsize=(10,6))
plt.plot(x,predict_score, color="red", label="Predicted Scores (Regression Line)")
plt.scatter(x, y, color='blue', label='Actual Scores')
plt.title("Model Prediction VS Actual Score")
plt.xlabel("Study Hours per week" ) 
plt.ylabel("Final Output")
plt.grid (True,color="gray",linestyle=":")
plt.legend()
# plt.tight_layout()
plt.show()