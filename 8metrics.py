from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.metrics import confusion_matrix
x_true=[1,0,1,0,1,0,1,1]
y_predict=[1,0,1,0,1,1,0,1]
print("accuracy score",accuracy_score(x_true,y_predict))
print("precision score",precision_score(x_true,y_predict))
print("recall score",recall_score(x_true,y_predict))
print("f1 score",f1_score(x_true,y_predict))
print("\n")

#  TN(true negative)  | FP(false positive)
#  FN(false negative) | TP(true positive)
print("confusion score:\n",confusion_matrix(x_true,y_predict))