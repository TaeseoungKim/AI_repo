#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 데이터를 분석(머신러닝)
# 1. 데이터를 로딩
import pandas as pd

# 유방암 데이터 셋 (악성, 악성x)
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
# print(data)

# 설명변수 : 특정 정답(종속변수)을 유추하기 위해서 정의된 데이터 셋
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. 데이터 탐색 (EDA)

# - 데이터의 개수, 컬럼의 개수
# - 각 컬럼 데이터에 대해서 결측데이터 존재 여부
# - 각 컬럼 데이터의 자료형
# (반드시 수치형의 데이터로만 머신러닝이 가능!!)
print(X.info())


# pandas는 컬럼 수가 너무 많으면 다 출력하지 않고 일부만 보여준다 (특징)
# - 출력 컬럼의 개수를 제어
pd.options.display.max_columns = 30

# - 출력 행의 개수를 제어
# pd.options.display.max_rows = 30


# 데이터를 구성하는 각 컬럼들에 대해서 기초 통계 정보를 확인
# - 데이터의 개수
# - 평균, 표준편차
# - 최소, 최대값
# - 4분위 수

# 데이터의 스케일 부분을 중점적으로 체크
# (스케일 : 값의 범위)
# - 각 컬럼 별 스케일의 오차가 존재하는 경우
# 머신러닝 알고리즘의 종류에 따라서 스케일 처리가 필요함
print(X.describe()) 


# 종속변수 - y
print(y)

# 종속변수의 값이 범주형인 경우
# 범주형 값의 확인 및 개수 체크가 필요하다. (0,1로 구성된 줄 알았는데 아닐 수 있으므로,)
print(y.value_counts())

# 범주형 종속변수의 경우 값의 비율이 중요하다
# - 일반적인 경우 실제 검출(예측)하고자 하는 데이터의 개수는 상대적으로 소량
# - 데이터의 비율이 많이 차이가 있는 경우
# (오버샘플링/언더샘플링)을 사용하여 데이터의 불균형 문제를 해결
print(y.value_counts() / len(y))




# 전처리는 데이터 분할을 한 뒤에 한다.
# 3. 데이터 분할
# - 학습, 테스트 데이터 셋으로 분할
# - 7 : 3, 8 : 2

# - 데이터 분할을 위한 함수 : train_test_split
from sklearn.model_selection import train_test_split

# train_size = 테스트 데이터에 어떤 비율을 줄 것 인지 (잘 안씀)
# test_size = 테스트 데이터에 어떤 비율을 줄 것 인지
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                    train_size=학습데이터비율,test_size=테스트데이터비율,
#                                                    stratify=정답데이터,
#                                                    random_state=임의의 정수값)

# stratify : 데이터가 분류(범주)형 데이터 셋인 경우에만 사용
#                   각 범주형 값의 비율을 유지하면서 데이터를 분할시키는 역활

# random_state : train_test_split 함수의 분할 결과를 항상 동일하게 분할됨을 보장하는 옵션
# 즉, stratify는 0과 1의 비율을 유지, random_state는 학습용,테스트 데이터의 비율을 유지하는 것인가..?


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)



# stratify=y를 해줌으로써, y 데이터의 비율을 맞춰준다..?


# 분할된 데이터의 개수 확인
# - 398 171
print(len(X_train), len(X_test))


# 분할된 데이터의 부분 확인
# -249 58 476 529 422
print(X_train.head())


# 분할된 학습, 테스트 데이터에 대한
# 종속변수의 비율을 확인
print(y_train.value_counts() / len(y_train))
print(y_test.value_counts() / len(y_test))


# 4. 데이터 전처리
# - 스케일 처리
# - 인코딩 처리
# - 차원 축소
# - 특성 공학... 나중에..


# 데이터를 분할하기전에 전처리를 한다면?
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 전체 데이터를 사용하여 스케일링 처리하게 된다면?
# - 전체 데이터가 0 ~ 1로 압축
# (이상치가 존재하지 않음. 모든 값은 범위 내에 있음)
# scaler.fit(X)

# 데이터를 분할한 후에 전처리를 하면?
# - 학습 데이터만 - ~ 1로 압축
# (테스트 데이터는 학습 데이터의 스케일링 범위에 따라서 음수 값, 혹은 1보다 큰값이 있을 수도 있음)

scaler.fit(X_train)
X_train = scaler.transform(X_train)

# 테스트 데이터는 학습 데이터로 fitting된 스케일러를 사용하여
# 변환하는 작업만 수행한다!
X_test = scaler.transform(X_test)


# 5. 머신러닝 모델 구축
from sklearn.neighbors import KNeighborsClassifier
# - 머신러닝 모델 객체 생성
# - (하이퍼 파라메터를 제어하는 부분)
model = KNeighborsClassifier()

# - 생성된 머신러닝 모델 객체를 학습
# (fit 메소드를 사용하는 부분)

# - 사이킷런의 모든 머신러닝의 클래스는 fit 메소드의 매개변수로 X,y를 입력받음
# - (X는 반드시 2차원, y는 1차원으로 가정함)
model.fit(X_train,y_train)

# - 머신러닝 모델의 평가
# - score 메소를 사용
# - score(X,y) : X를 사용하여 예측된 값과 y를 비교하여 평가 결과를 반환
# - 주의사항!!!
# - 머신러닝 클래스의 종류가 분류형이라면 score 메소드의 결과는 정확도
# (전체 데이터에서 정답으로 예측한 비율)
# - 머신러닝 클래스의 종류가 회귀형이라면 score 메소드의 결과는 R2 스코어(결정계수)
# (- 값 ~ 1 사이의 값)
score = model.score(X_train, y_train)
print(f'score = {score}')

score = model.score(X_test, y_test)
print(f'score = {score}')

# 학습된 머신러닝 모델을 사용하여 예측
# - predict 메소드 사용
# - model.predict(설명변수 - X)
# - 예측할 데이터 X는 반드시 2차원으로 입력되어야 함

pred = model.predict(X_test[:3]) # X_test.iloc[:3] 이 3개의 행, 데이터를 가지고 예측을 한 값을 Pred에 저장
print(f'pred = {pred}')
print(f'{y_test[:3]}')

# 학습된 머신러닝 모델을 사용하여 확률값을 예측할 수 있다.
# n_jobs 옵션 : 병렬처리를 하는 옵션,  (default로 하면 cpu core하나만 사용)
proba = model.predict_proba(X_test[:100]) # 위에서 전처리를 하며 numpy타입으로 변했기 때문에 iloc를 사용하지 못함
print(proba)
print(f'{y_test[:3]}')










