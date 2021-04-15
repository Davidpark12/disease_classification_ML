import pandas as pd
import numpy as np
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
train = pd.read_csv("dataset_park_median.csv")
train.head()
print(train.head())

y = train["Classification"]
#print(y.head())

train = train.drop(["Classification", "Unnamed: 0"] , 1)
#print(train)

#print(y.value_counts())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["S"] = le.fit_transform(train["S"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)

from xgboost import XGBClassifier
xgb_cf = XGBClassifier(nthread=-1, n_estimators=2000, colsample_bytree=0.75, subsample=0.85, max_depth=2
                       , reg_alpha=0.1, learning_rate=0.02)
# 최적화 n_estimator=2000이다.
# nthread는 기본값이 1이다. random forest에는 njobs와 동일. 사용할 수 있는 모든 cpu를 한개로 쓰게 된다. 기본은 4개인데..3배 정도가 느려진다.
# 4나 -1이 모든 cpu core를 다 사용해서 학습을 하겠다는 얘기므로, 속도가 4배가 빨라 진다.
# n_estimator는 lightgbm과 마찬가지로 tree 수이다. random forest는 n_estimator이다.
# max_depth는 얼마나 깊어질껀지.. 여기서 2개의 층까지만 깊어진다. lightgbm의 num_leaves에 해당
# reg_alpha는 regression에서의 lasso인 L1, Ridge인 L2에 해당. 즉 중요하게 생각하는 feature importance를 낮춘다. reg_alpha는 lasso 해당하며 기본값은 0이다.
# 즉 이걸로 0으로 보낸 애들이 있다는 말이며, 중요한 feature의 가중치를 낮추어서 다른 feature에게 가중치를 나누어주며, 과적합을 직접적으로 막아준다.
# colsample_bytree가 과적합을 간접적으로 막아준다면, reg_alpaha(L1)와 reg_lambda(L2에 해당)는 직접적으로 막아준다 하겠다.
# Learning rate는 학습율, 학습양으로 단위 나무에서 학습할때 공이 내려오는 속도, 관성에 해당하며, 이게 크면 굉장히 빨리 내려오게 된다.
# 그렇게 되면 최적의 지점에 한번에 안착을 못하고 진자 운동을 해버리기게 때문에 최적을 찾기가 힘들어 진다. 기본값은 0.1이다.
# 이 옵션을 줄여주게 되면, 속도가 느려진다. 너무 느려지게 되면, 중간에 local minimum 에 빠져서 global minimum 을 찾지 못하게 된다. 일반적으로
# 머신러닝에서 0.01~0.025가 보통 최적이 나온다. learinig_rate를 중요한 option이지만, 보통 낮추는데 이 경우는 n_estimator(나무갯수, 학습횟수)를
# 늘려야 한다

pred = []
for train_index, valid_index in skf.split(X_train, y_train):
    X_train2, X_valid2 = X_train.iloc[train_index], X_train.iloc[valid_index]
    # 80%인 4000개 중에서 train_index는 3200개에 해당(4/5), valid_index는 20%인 나머지 800개이다(4000개의 1/5))
    # n_split이 5이므로 80%에서 80%가 된 것임
    y_train2, y_valid2 = y_train.iloc[train_index], y_train.iloc[valid_index]
    # 위와 동일함, 모델을 총 5번을 돌려서 그것의 평균을 낸다.
    xgb_cf.fit(X_train2, y_train2, eval_set=[(X_valid2, y_valid2)], early_stopping_rounds=20, eval_metric="mlogloss")
    # early_stopping이 5라고하면, validation loss가 계속 찍히는데 5번동안 늘어나면 이상적이지만, 5번 늘어나다가 6번부터 validation loss가 감소할 수가 있다.
    pred.append(xgb_cf.predict_proba(X_test))
    # 5번마다의 최적의 나무의 개수를 가지고, 각각 달라진 나무의 개수를 가지고 실제 test를 가지고 예측을 한다. 총 5개의 예측값이 만들어진다.
    # predict_proba가 각각 1029개의 데이터의 정답값에 대해서 39개를 확률값으로 계산한다.

#print(np.mean(pred, axis=0), np.mean(pred, axis=0).shape)
#print(y.unique())
#print(pd.DataFrame(np.mean(pred, axis=0), columns=y.unique()).idxmax(1))
pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y.unique()).idxmax(axis=1)
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(pred_df, y_test))
print(np.mean(pred, axis=0))

class_probs = np.mean(pred, axis=0)
#(1029, 39)이 다섯개가 만들어지고 각각의 리스트 안에 들어있음. 각각 질환 클래스에 대한 확률값을 예측한
#1029개의 데이터가 필요하므로 5개중 평균 행렬이 만들어짐
labels = le.fit_transform(y_test)
# y_test는 정답값이 문자(영어)이므로
top1 = 0.0
top5 = 0.0

top5_list = []
for i, l in enumerate(labels):
    class_prob = class_probs[i]
    top_values = (-class_prob).argsort()[:5]
    #argsort는 작은것들부터 정리가 된다. 그래서 앞이 -class_prob를 하면 큰게 앞에 포진된다.
    # if top_values[0] == l:
    #     top1 += 1.0
    if np.isin(np.array([l]), top_values):
        top5_list.append(l)
        top5 += 1.0
    else:
        top5_list.append(-1)


print("top5 acc", top5/len(labels))

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
print(accuracy_score(top5_list, le.fit_transform(y_test)))

print(classification_report(top5_list, le.fit_transform(y_test)))

