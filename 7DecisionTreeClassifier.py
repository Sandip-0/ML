from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
x=[
    [7,2], #apple
    [8,3], #apple
    [9,8], #orange
    [10,9] #orange
]
y=[0,0,1,1] #0=apple ,1=orange(color shade wise)
model.fit(x,y)

size=float(input("enter fruit size in cm : "))
shade=float(input("enter fruit color shade(0-10): "))

m_pridict=model.predict([[size,shade]])[0]

if(m_pridict==0):
    print("This is likely an apple")
else:
    print("This is likely an orange")