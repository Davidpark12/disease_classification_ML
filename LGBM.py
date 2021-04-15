import pandas as pd
import numpy as np


pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

train = pd.read_csv("dataset_park_new.csv", encoding="CP949" )
train.head()
print(train.head())
print(train.dtypes)

y = train["Classification"]


train = train.drop(["Classification"] , 1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["S"] = le.fit_transform(list(train["S"]))

cate_cols = train.dtypes[train.dtypes=="object"].index
train[cate_cols] = train[cate_cols].astype("float32")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)

from lightgbm import LGBMClassifier

lgb_cf = LGBMClassifier(colsample_bytree=0.85,subsample=0.9, num_leaves=6, min_child_samples=30, n_estimators=300)

pred = []
for train_index, valid_index in skf.split(X_train, y_train):
    X_train2, X_valid2 = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train2, y_valid2 = y_train.iloc[train_index], y_train.iloc[valid_index]
    lgb_cf.fit(X_train2, y_train2, eval_set=(X_valid2, y_valid2), early_stopping_rounds=20)
    pred.append(lgb_cf.predict_proba(X_test))
print(np.mean(pred, axis=0))

pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y.unique()).idxmax(1)

class_probs = np.mean(pred, axis=0)

labels = le.fit_transform(y_test)

top1 = 0.0
top5 = 0.0
top5_list = []
for i, l in enumerate(labels):
    class_prob = class_probs[i]
    top_values = (-class_prob).argsort()[:5]

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

print(classification_report(top5_list, le.fit_transform((y_test))))
print(top5_list)
print(le.fit_transform(y_test))


