# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_absolute_error,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load dataset
df=pd.read_csv("data/student_success_dataset.csv")


# Step 3: Encoding categorical variables
le=LabelEncoder()
df['Internet']=le.fit_transform(df['Internet'])
df['Passed']=le.fit_transform(df['Passed'])


# Step 4: Feature selection and scaling
features=['SleepHours','StudyHours','Attendance','PastScore']
scaler=StandardScaler()
df_scaled=df.copy()
X=scaler.fit_transform(df[features])
y=df_scaled['Passed']


# Step 5: Split data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# Step 6: Train model
model=LogisticRegression()
model.fit(X_train,y_train)


# Step 7: Evaluation
y_pred=model.predict(X_test)
print('Classification report')
print(classification_report(y_test,y_pred))


# Step 8: Confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


print("predict your reasult")
try:
    study_hours=float(input("enter your study hours: "))
    Sleep_hours=float(input("enter your Sleep hours: "))
    attendance=float(input("enter your Attendance: "))
    Past_Score=float(input("enter your PastScore: "))

    user_input_df=pd.DataFrame([{
        'SleepHours':Sleep_hours,
        'StudyHours':study_hours,
        'Attendance':attendance,
        'PastScore':Past_Score
    }])
    user_input_scaled=scaler.fit_transform(user_input_df)
    prediction=model.predict(user_input_scaled)[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Prediction Based on input: {result}")
except Exception as e:
    print(f"Excetion is {e}")