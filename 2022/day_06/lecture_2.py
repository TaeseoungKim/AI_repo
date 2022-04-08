# -*- coding: utf-8 -*-

# 보팅: 각 모델들에게 같은 데이터를 주고 취합하는 개념
# 배깅: 각 다른 모델들에게 각각 다른 데이터를 준다
# 배깅의 random sampling
# 배깅의 bootstraping 
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
from sklearn.ensemble import BaggingClassifier



# 앙상블을 구성하는 각 모델의 클래스를 로딩
from sklearn.tree import DecisionTreeClassifier

# - 배깅의 경우 앙상블을 구성하는 머신러닝 모델은 1개를 사용한다
# - 다만 각각의 모델이 학습하는 데이터는 무작위 추출 방법(부트스트래핑)으로 처리하여 
#       다수개의 모델이 **서로 다른 관점의 학습**을 진행할 수 있도록 처리

# 취합(voting)을 할 때는 제약을 걸지않고 과적합을 했었지만, 


base_estimator = DecisionTreeClassifier(random_state=1)
model = BaggingClassifier(base_estimator=base_estimator,
                         n_estimators=10,
                         max_samples=0.5,
                         max_features=0.5,
                         random_state=1,
                         n_jobs=-1)

#배깅의 param
# base_estimator: 알지?
# n_estimators: 몇개의 모델을 쓸지
# max_samples: 각 모델들에게 할당할 데이터
# max_features: 헷갈릴 수 있지만 잘 알아야댐, 컬럼제어, 즉, 가져갈 수 있는 컬럼의 비
# bootstrap: 이거 시험에 나온다

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




























