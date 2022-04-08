#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:19:32 2022

@author: macbook
"""


# - 결정트리(분류) Decision Tree(
# Decision Tree는 data를 분류할 때, 무조건 하나의 특성만 가지고 분류한다.
# 그러므로 Decision Tree는 스케일처리(전처리)가 필요없다. => 원본 데이터로 바로 사용할 수 있다.
import pandas as pd 
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data,
                 columns=data.feature_names)
y = pd.Series(data.target)

X.info()
X.isnull().sum()
X.describe(include='all')

y.head()
y.value_counts()
y.value_counts() / len(y)

from sklearn.model_selection import train_test_split
splits = train_test_split(X,y,
                          test_size=0.3,
                          random_state=11,
                          stratify=y)

X_train = splits[0]
X_test=splits[1]
y_train=splits[2]
y_test=splits[-1]

X_train.head()
X_test.head()

X_train.shape
X_test.shape

y_train.shape
y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)

# 가중치를 넓힌다..?

from sklearn.tree import DecisionTreeClassifier
# penalty를 줄 수 있다. (ex, l1, l2 )
# class_weight 



model = DecisionTreeClassifier(max_depth=3, #max_depth=Nonde => 끝까지 내려간다.
                               class_weight='balanced',
                               random_state=0)



model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)

# 특성의 중요도 값 확인
print(f'cpef_ : {model.feature_importances_}')

# 분류 모델의 평가 방법
# 1. 정확도
#   - 전체 데이터에서 정답의 비율을 의미
#   - 암이 아닌 사람을 맞추는 것이 아닌, 암인 사람을 맞추는 것이 중요하다
from sklearn.metrics import accuracy_score

# 2. 정밀도
#   - 모델이 예측한 결과에서 정답인 비율
#   - 각 클래스 별로 분류된 값이 반환
#   - 만약 모델에서 암인 사람의 정밀도가 10%라면, 모델이 예측한 10명 중, 1명만 암이였던거임 -> 다틀림
from sklearn.metrics import precision_score

# 3. 재현율
#   - 실제 데이터 중 모델이 정답으로 예측한 비율
#   - (각 클래스 별로 분류된 값이 반환)
#   - 정밀도가 높은 모델은 재현율은 낮다.
#   - 재현율이 높다 -> 조금 틀리더라도 
from sklearn.metrics import recall_score


# 혼동행렬
from sklearn.metrics import confusion_matrix
pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm

# class_weight : 악성이 아닌 쪽에다 가중치를 준다면, 정말 악성인 것만 고른다.

# array([[144,   4],
#       [  4, 246]])


# 정밀도(0) : 125 / (125+10)
# 재현율(0) : 125 / (125+23)

# question : 1의 정밀도와 재현율은? => 시험에 나옴
# 정밀도(1) : 240 / (23 + 240)
# 재현율(1) : 240 / (10 + 240)
# 1이 많으므로 1을 많이 맞추는게 페널티가 적다 => 시험에 나옴



