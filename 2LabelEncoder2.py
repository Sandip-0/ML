import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import numpy as np

df=pd.read_csv("data/data.csv")
le=LabelEncoder()
df_label=df.copy()
# df_label['gender_e']=le.fit_transform(df_label["gender"])
# df_label['pass_e']=le.fit_transform(df_label["pass"])
print('new')
# print(df_label)


place_e=pd.get_dummies(df_label,columns=['place'])
place_e=place_e.replace({False:0,True:1,'Yes':1,'No':0,'Female':0,'Male':1})
print('one hot encoded data')
print(place_e)