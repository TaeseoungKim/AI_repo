# -*- coding: utf-8 -*-

import pandas as pd
X = pd.DataFrame()

X['rate'] = [0.7, 0.8, 0.77]
print(X)


X['price'] = [10000, 5000, 9700]
print(X)

y = pd.Series([0,1,0])


# 최근접 이웃 알고리즘의 학습 및 예측 방법
# - 학습 : fit 메소드에 입력된 데이터를 단순 저장
# - 예측 : fit 메소드에 의해서 저장된 데이터와 예측하고자 하는 신규 데이터와의 유클리드 거리를 계산하여
#               가장 가까운 n_neighbors 개수를 판단하여
#               n_neighbors의 y의 값을 사용하여 보팅을 진행



# 즉, 최근접 이웃의 단점은 스케일의 차이가 영향을 많이 준다.
# 위와 같이 rate와 price의 범위 차이가 너무 크다보니 스케일이 적은 데이터는 일반적으로 무시된다

# 데이터 전처리 (스케일링), 최소~최대의 범위를 0~1로 바꿔준다
# - MinMax, Standard, Robust
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

print(X)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X,y)

# (원래 제곱까지 하는데 여기선 생략)
# 0 0.70 10000 -> 0.01, 6000 -> 더한다 6000.01
# 0 0.80 5000 -> 0.09, 1000 -> 1000.09
# 0 0.77 9700 -> 0.06, 5700 -> 5700.06

pred = model.predict([[0.71, 4000]])
print("pred = ",pred)

new_data = [[0.71, 4000]]
new_data = scaler.transform(new_data)
print(new_data)

pred = model.predict(new_data)
print("pred = ",pred)