import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = {
    'Age': [25, 30, 35, 40, 45, 50],
    'Income': [30000, 40000, 50000, 60000, 70000, 800001],
    'Spending': [70, 60, 50, 40, 30, 20],
    'Savings': [1000, 5000, 8000, 10000, 15000,20000]
}

df=pd.DataFrame(data)

scaler=StandardScaler() # Z_SCALE =(MEAN-x)/std 
scaler_data=scaler.fit_transform(df)
pca=PCA(n_components=2) # all column summerize into 2 column
pca_result=pca.fit_transform(scaler_data)
pca_df=pd.DataFrame(pca_result,columns=['PCA1','PCA2'])
explained_variance = pca

explained_variance=pca.explained_variance_ratio_
print("Variance captured by each PCA Component:")
print(np.round(explained_variance * 100, 2))
plt.scatter(pca_df ['PCA1'], pca_df['PCA2'],color='black')
plt.title("PCA Projection (2D View)") 
plt.xlabel('PCA1 Main Pattern')
plt.ylabel('PCA2 Minor Pattern')
plt.grid(True)
plt.show()