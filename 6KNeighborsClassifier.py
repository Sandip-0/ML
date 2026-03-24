from sklearn.neighbors import KNeighborsClassifier
x =[
    [180, 7.1],
    [200, 7.5],
    [250, 8.1],
    [300, 8.5],
    [330, 9.1],
    [360, 9.51]
] 
y=[0,0,0,1,1,0] #0-apple , 1-orrange
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

weight=float(input("enter fruit weight in grams : "))
size=float(input("enter fruit size in cm : "))

m_predict=model.predict([[weight,size]])[0]  # list under contant so we put 0, like index
if(m_predict==1):
    print("This is likely an orange")
else:
    print("This is likely an apple")