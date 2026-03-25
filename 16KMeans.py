import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data={
    'Customer': ['Riya', 'Aman', 'Faizan','Neha','Imran','Sneha'],
    'Age': [20, 30, 40, 22, 38, 25],
    'Spending': [100, 200, 300, 110, 290, 130]
}
df=pd.DataFrame(data)
X=df[['Age','Spending']]
model=KMeans(n_clusters=2,random_state=42,n_init=10)
df['Group']=model.fit_predict(X)
plt.figure(figsize=(6,5))
for g in df['Group'].unique(): #[0,1]=n_clusters=2
    group_data=df[df['Group']==g]
    plt.scatter(group_data['Age'],group_data['Spending'],label=f'group {g}')

plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title("Customer sigment (K-Means) ")
plt.legend()
plt.grid()
plt.show()