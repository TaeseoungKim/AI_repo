# -*- coding: utf-8 -*-

# - 최근접 이웃 알고리즘을 사용한 회귀분석
# - 최근접 이웃 알고리즘의 단점 (한계)

import numpy as np
X = np.arange(1,11)
print(X)

X = X.reshape(-1,1) # 
print(X)

# 종속변수 y의 값이 연속된 수치형이다
# - 회귀분석용 종속변수의 선언
y = np.arange(10, 101, 10)
print(y)

# 회귀분석을 위한 최근접 이웃 클래스
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X,y)


# 최근접 이웃 알고리즘의 회귀 분석은 입력된 Xdhk
# 가장 인접한 이웃을 검색한 후, 해당 이웃들의 값의 평균을 반환
# 7과 8의 70, 80 값의 평균을 반환
pred = model.predict([[7.3]])
print(f'7.3 -> {pred}') # 결과는 n_neighbors가 2이므로 70과 80의 평균


pred = model.predict([[3.1]])
print(f'3.1 -> {pred}')

# 최근접 이웃 알고리즘은 fit 메소드에 의해서 입력된
# 데이터의 범위에 종속된다.
# fit 메소드에서 입력된 X의 범위 이상의 값을 올바르게 예측할 수 없다.
# 즉, data의 값이 범위의 한계가 없다면 최근접 이웃은 제대로 작동x
pred = model.predict([[59.5]])
print(f'59.5 -> {pred}')

pred = model.predict([[1000000]])
print(f'1000000 -> {pred}')

# 연속된 수치형을 선형모델을 사용하여 예측할 수 있는
# 머신러닝 클래스 활용
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

pred = model.predict([[7.3]])
print(f'7.3 -> {pred}')

pred = model.predict([[3.1]])
print(f'3.1 -> {pred}')

pred = model.predict([[59.5]])
print(f'59.5 -> {pred}')

pred = model.predict([[1000000]])
print(f'1000000 -> {pred}')



