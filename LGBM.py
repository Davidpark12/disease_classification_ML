import pandas as pd
import numpy as np


pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
#train = pd.read_csv("dataset_park_median.csv")
train = pd.read_csv("dataset_park_new.csv", encoding="CP949" )
train.head()
print(train.head())
print(train.dtypes)

y = train["Classification"]
#print(y.head())

#train = train.drop(["Classification", "Unnamed: 0"] , 1)
train = train.drop(["Classification"] , 1)
#print(train)


#print(y.value_counts())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["S"] = le.fit_transform(list(train["S"]))
#train = train.replace({"180.0 이상":180, "120 이상": 120, "180 이상":180, "120.0 이상": 120})
cate_cols = train.dtypes[train.dtypes=="object"].index
train[cate_cols] = train[cate_cols].astype("float32")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)

from lightgbm import LGBMClassifier

lgb_cf = LGBMClassifier(colsample_bytree=0.85,subsample=0.9, num_leaves=6, min_child_samples=30, n_estimators=300)
# n_estimator=300으로 최종 설정
# #colsample_bytree는 기본값이 1이므로, 나무가 만들어질때마다, 어떤 feature를 선택할떄마다 100% 칼럼을 모두 쓰게 된다.
# # "그러나 이렇게 다 쓰게 되면, 각각 나무에 들어가는 column에 들어가는 상대적으로 중요하지 않는 칼럼들을 오히려 학습을 하지 않게 된다."
# # "하는것만 하게 된다. 그러므로 거기에만 과적합이 되게 된다. 경쟁에서 밀린 칼럼은 학습을 못하게 되므로 colsample_bytree 0.7~0.8, 칼럼이 숫자가 많으면
# # 0.5~0.4로 낮추어도 된다.즉 colsample_bytree에서 0.85씩 feature를 random하게 나무가 만들어질때마다 다른 질문을 하게 되서 만들어지게 된다.""
# # subsample은 기본값은 1이다. 즉 tree를 만들때 마다, random하게 샘플을 넣어주게 된다. 이렇게 되면 모델 학습시에 결국 다른 질문을 하게 되고,나무들 사이에
# #다양성이 확보되게 된다.
# # num_leave는 기본값이 31이다. 입사귀의 숫자이다. 이 말은 입사귀를 31를 개를 만들어야 하니까 과대적합이 일어나고 있었으며, 본 연구에서는 오히려 입사귀를 줄이는 게 성능향상에 도음이 되었으며 대신 300개 정도 많은 트리를 심었다. 어자피 early stopping에서
# # 멈추었고 우리 경우 100개에서 멈추었다 확인할 것...
# # min_child_sample는 하나의 입사귀를 만들때 최소 자식노드의 data가 최소 30개의 sample이 있어야지 split가 된다. 원래 기본값는 20개임.
# #learning_rate=0.1가 기본값이며 원래 최적화는 learning_rate=0.02)
# # n_estimator는 random forest는 10이고, Lightbgm, xgboost는 기본값이 100이다.

from lightgbm import LGBMClassifier
#lgb_cf = LGBMClassifier(colsample_bytree=0.7, subsample=0.8, num_leaves=24, learning_rate=0.1, n_estimators=50)

pred = []
for train_index, valid_index in skf.split(X_train, y_train):
    X_train2, X_valid2 = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train2, y_valid2 = y_train.iloc[train_index], y_train.iloc[valid_index]
    lgb_cf.fit(X_train2, y_train2, eval_set=(X_valid2, y_valid2), early_stopping_rounds=20)
    pred.append(lgb_cf.predict_proba(X_test))
print(np.mean(pred, axis=0))
#pd.DataFrame(pred)
#pd.DataFrame(np.mean(pred, axis=0)).to_csv("lgbm_final.csv", index= False)

#print(np.mean(pred, axis=0), np.mean(pred, axis=0).shape)
#print(y.unique())
#print(pd.DataFrame(np.mean(pred, axis=0), columns=y.unique()).idxmax(1))
pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y.unique()).idxmax(1)

class_probs = np.mean(pred, axis=0)
#(1029, 39)이 다섯개가 만들어지고 각각의 리스트 안에 들어있음. 각각 질환 클래스에 대한 확률값을 예측한
#1029개의 데이터가 필요하므로 5개중 평균 행렬이 만들어짐
labels = le.fit_transform(y_test)
# y_test는 정답값이 문자(영어)이므로 A0A같은 것들이 A0A=0,A0B=1,A0C=2의 숫자로 바꾼다
top1 = 0.0
top5 = 0.0
top5_list = []
for i, l in enumerate(labels):
    class_prob = class_probs[i]
    top_values = (-class_prob).argsort()[:5]
    #argsort는 작은것들부터 정리가 된다. 그래서 앞이 -class_prob를 하면 큰게 앞에 포진된다.
    if top_values[0] == l:
        top1 += 1.0
    if np.isin(np.array([l]), top_values):
        top5_list.append(l)
        top5 += 1.0
    else:
        top5_list.append(-1)

print("top1 acc", top1/len(labels))
print("top5 acc", top5/len(labels))


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
print(accuracy_score(top5_list, y_test))
# print(f1_score(pred_df, y_test, average = "micro"))
# print(f1_score(pred_df, y_test, average = "macro"))
# print(f1_score(pred_df, y_test, average = "weighted"))
print(classification_report(top5_list, le.fit_transform((y_test))))
print(top5_list)
print(le.fit_transform(y_test))
# import shap
# import matplotlib.pyplot as plt
# explainer = shap.TreeExplainer(lgb_cf).shap_values(X_train2)
# shap.summary_plot(explainer, X_train2)
# #plt.show()

