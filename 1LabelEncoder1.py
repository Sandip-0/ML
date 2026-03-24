import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import numpy as np
df=pd.read_csv("data.csv")

le=LabelEncoder()
df_label=df.copy()
df_label['gender_encoded']=le.fit_transform(df_label["gender"])
df_label['pass_encoded']=le.fit_transform(df_label["pass"])
print('new')
# print(df_label)
# print(df_label[['name','gender_encoded','pass_encoded']])

place_encoded=pd.get_dummies(df_label,columns=['place'])
print('one hot encoded data')
# place_encoded=place_encoded.replace(False,0)
# place_encoded=place_encoded.replace(True,1)

place_encoded=place_encoded.replace({False:0,'No':0,'Yes':1,True:1,})
print(place_encoded)