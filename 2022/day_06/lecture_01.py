# -*- coding: utf-8 -*-

# 앙상블
# - 다수개의 모델을 결합하여 하나의 예측을 할 수 있는 결합 모델
# - 다수결의 원칙이 적용 (분류)
# - 평균 원칙이 적용 (회귀)

# 앙상블은 가장 좋은 평가 성적을 반환하지 않음
# - 앙상블은 하나의 모델 객체를 사용하지 않고
#   다수개의 모델 결과를 사용하여 다수결/평균을 취하므로
#   앙상블을 구성하는 많은 모델에서 가장 좋은 성적의
#   모델보다는 항상 평가가 떨어질 수 있음
#   즉, 앙상블 model은 best를 찾는 것이 아닌, 일반화 성능을 극대화하기 위한 모델
#   (테스트 데이터에 대한 성능)

#   앙상블을 구현하는 방법
#   1. 취합
#   - 앙상블을 구성하고 있는 각각의 모델이 독립적으로 동작
#   - 각각의 모델이 독립적으로 학습하고 예측한 결과를 반환하여
#   최종적으로 취합된 결과를 다수결/평균으로 예측함
#   - Voting, Bagging, RandomForest
#   - **취합 기반의 앙상을 내부의 모델들은 각각 일정 수준이상의 예측 성능을 달성해야함**
#   - 학습과 예측의 속도가 빠르다(병렬처리가 가능한 구조): 병렬처리 중요

#   2. 부스팅
#   - 앙상블을 구성하고 있는 각각의 모델이 선형으로 결합되어
#       점진적으로 학습의 성능을 향상시켜 나가는 방법
#   - 결과 = 첫번째 모델이 예측 결과 * 가중치 
#                       + 두번째 모델이 예측한 결과 * 가중치 + ... N번째 모델이 예측한 결과 * 가중치
#   - AdaBoosting, GradientBoosting, XGBoost, LightGBM(속도가 굉장히 빠름)
#   - 부스팅 기반의 앙상블 내부 모델들은 강한 제약 조건을 설정하여 점진적으로 성능이 향상될 수 있도록 제어한다.
#   - 취약점: 학습 속도가 최악이다. (병렬이 아닌 순차적으로 처리해야 하므)
# 아래 내용 시험에 나온다 함
# : 취합을 통해 앙상블을 구성할 때에는, 각 모델을 최대한 과적합을 해서 각 모델의 최고의 성능을 내도록한다 => 어차피 평균을 낼거기 때문


import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=0)

# 앙상블 기반의 클래스를 로딩
from sklearn.ensemble import VotingClassifier


# 모델을 제어하기 위해 hyper param이 필요하다?
# param들
# estimators : 쓰고자 하는 모델들
# voting : hard=각 모델들이 동등한 투표권을 가짐, soft=예측한 결과의 확률 값을 기반으로 보팅 => hard가 더 좋다함
# weights : 속도 어쩌구라 했는데..

# 앙상블을 구성하는 각 모델의 클래스를 로딩
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 밑 모델 세개는 "필수적인" param들만 넣었음
m1 = KNeighborsClassifier(n_jobs=-1)
m2 = LogisticRegression(ramdom_state=1,n_jobs=-1)
m3 = DecisionTreeClassifier(max_depth=3,ramdom_state=1)

estimators = [('knn',m1), ('lr',m2), ('dt',m3)]
model = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Score (train): {score}')

score = model.score(X_test, y_test)
print(f'Score (train): {score}')

pred = model.predict(X_test[:1])
print(f'predict (ensemble): {pred}')

# 1번째 모델이 예측한 값
pred = model.estimators_[0].predict(X_test[:1])
print(f'predict (ensemble): {pred}')




























