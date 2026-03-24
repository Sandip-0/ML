from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
x=[[1],[2],[3],[4],[5]]
y=[0,0,1,1,1]

model.fit(x,y)
hour=float(input('enter study hour : '))

result=model.predict([[hour]])[0] # list under contant so we put 0, like index

if (result==1):
    print(f"based on your study hour {hour}, you may PASS")
else :
    print(f"based on your study hour {hour}, you may FAIL")