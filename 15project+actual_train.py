from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("data/suicide.csv")

le = LabelEncoder()
model = RandomForestRegressor(n_estimators=100)

data['country'] = le.fit_transform(data['country'])
data['sex'] = le.fit_transform(data['sex'])
data['age_group'] = le.fit_transform(data['age_group'])

X = data[['country','sex','age_group','year']]
y = data['suicide_rate']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


model.fit(X_train,y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,pred)

print("MAE:",round(mae,2))
print("MSE:",round(mse,2))
print("RMSE:",round(rmse,2))
print("R2:",round(r2,4))



plt.figure(figsize=(8,6))
plt.scatter(y_test, pred)

plt.xlabel("Actual Suicide Rate")
plt.ylabel("Predicted Suicide Rate")
plt.title("Actual vs Predicted Suicide Rate")

# plt.show()


errors = y_test - pred
plt.figure(figsize=(8,6))
plt.scatter(pred, errors)
plt.axhline(0)

plt.xlabel("Predicted Values")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Plot")

# plt.show()

plot_data = pd.read_csv("data/suicide.csv")
pivot = plot_data.pivot_table(values='suicide_rate', index='year', columns='country')
pivot.plot(figsize=(12,6))
# plt.figure(figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Suicide Rate")
plt.title("Suicide Rate Trend by Country")

plt.legend(title="Country", bbox_to_anchor=(1.05,1))
plt.tight_layout()

plt.show()