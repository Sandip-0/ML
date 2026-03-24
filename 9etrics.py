'''
MEAN ABSOLUTE ERROR
1- take the mistake difference
2- remove the minus sign
3- add
4- divide

MSE Mean Squared Error
1- Mistakes square them
2- add
3- divide total

RMSE
Root Mean Squared Error
'''

from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
# real ans
real_score=[90, 60, 80, 100] #90+60+80+100=330
# model ans
predict_score=[85, 70, 70,95] #85+70+70+95=320

mae=mean_absolute_error(real_score,predict_score) # diff=5+10+10+5=30/4=7.5
mse=mean_squared_error(real_score,predict_score) # 5^2+10^2+10^2+5^2=25+100+100+25=250/4=62.5
rmse = np.sqrt(mse) # 7.9

print("MAE: On average off by: ", mae)
print("MSE: Squared Mistake Value: ", mse)
print ("RMSE: Final Realistic error: ", rmse)