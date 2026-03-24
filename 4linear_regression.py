'''
linear regression
1- finds a pattern in old data
2- straight line
3- line
y=m*x + b
[[value]] 2d list
[[6]]
X= [[11, [21, [31, [41, [511]]
1- its not about the line, its about the story
2- 5 rows use start
3- accuracy is not always the goal
'''
from sklearn.linear_model import LinearRegression

model=LinearRegression()
x=[[1],[2],[3],[4],[5]] # it should be 2d
y=[40,50,65,75,90]
model.fit(x,y)

hour=float(input('enter study hour : '))
predict_marks=model.predict([[hour]])
print(f"based your study hour {hour} , you may score around {predict_marks}")