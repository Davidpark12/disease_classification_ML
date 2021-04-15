import pandas as pd
import numpy as np

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
#train = pd.read_csv("dataset_park_new_2.csv", encoding="CP949" )
train = pd.read_csv("dataset_park_new_2.csv", encoding="CP949" )
#train = pd.read_csv("dataset_park_new_2_MODY(25)_final_2.csv", encoding="CP949" )
print(train.isnull().sum().sort_values(ascending=False))
#print(len(train["Classification"].value_counts()))
print(train["Classification"].value_counts())

train["FE"] = train["FE"].fillna(train["FE"].median())
print(train.isnull().sum().sort_values(ascending=False))

y = train["Classification"]
train = train.drop("Classification", 1)
#print(train.dtypes)

train = pd.get_dummies(train)


print(train.dtypes)
for i in train.columns:
    train[str(i)] = train[str(i)].fillna(train[str(i)].median())
print(train.isnull().sum().sort_values(ascending=False))


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_label = pd.Series(y.copy())
y = le.fit_transform(y)
y = pd.Series(y)

from sklearn.preprocessing import StandardScaler
#모든 column 들의 범위를 -1~1로 만들어 준다. 평균이 0이고 분산이 1인 데이터의 분포
sc = StandardScaler()
train[train.columns] = sc.fit_transform(train)

print('-'*10)
print(train.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)
#print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
#print(y_valid.value_counts())

y_train_label, y_valid_label = train_test_split(y_label, test_size=0.2, random_state=38, stratify=y_label)
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score


pred = []
i = 0
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)
for train_index, valid_index in skf.split(X_train, y_train):
    i += 1
    X_train2, X_valid2 = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train2, y_valid2 = y_train.iloc[train_index], y_train.iloc[valid_index]
    dlmodel = Sequential()
    dlmodel.add(Dense(1024, input_dim=train.shape[1], activation="relu"))
    dlmodel.add(Dropout(0.5))
    # dlmodel.add(Dense(256, activation="relu"))
    # dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(256, activation="relu"))
    dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(39, activation="softmax"))
    dlmodel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc", f1score])
    # earlystop = EarlyStopping(patience=10, verbose=1, monitor="val_f1score", mode="max") # monitor기본값은 val_loss이다.
    # modelcheck = ModelCheckpoint(str(i)+"best.h5", save_best_only=True, verbose=1, monitor="val_f1score", mode="max")
    # reducelr = ReduceLROnPlateau(patience=7, verbose=1, monitor="val_f1score", mode="max")
    earlystop = EarlyStopping(patience=10, verbose=1)  # monitor기본값은 val_loss이다.
    modelcheck = ModelCheckpoint(str(i) + "best.h5", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(patience=7, verbose=1)
    dlmodel.fit(X_train2, y_train2, validation_data=(X_valid2, y_valid2), epochs=150, batch_size=128,
                callbacks=[earlystop, modelcheck, reducelr])

    dlmodel.load_weights(str(i)+"best.h5")
    pred.append(dlmodel.predict(X_test))
#pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=pd.Series(y.unique()).sort_values()).idxmax(1)
pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y_label.unique()).idxmax(1)

print(np.mean(pred, axis=0))
#pd.DataFrame(pred)
#pd.DataFrame(np.mean(pred, axis=0)).to_csv("Deeplearning_final_TOP5_1.csv", index= False)

from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(pred_df, y_valid_label))


class_probs = np.mean(pred, axis=0)

labels = le.fit_transform(y_test)

top1 = 0.0
top5 = 0.0
top5_list = []
for i, l in enumerate(labels):
    class_prob = class_probs[i]
    top_values = (-class_prob).argsort()[:5]

    if np.isin(np.array([l]), top_values):
        top5_list.append(l)
        top5 += 1.0
    else:
        top5_list.append(top_values[0])

print("top5 acc", top5/len(labels))

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
print(accuracy_score(top5_list, le.fit_transform(y_test)))

print(classification_report(top5_list, le.fit_transform(y_test)))

print(top5_list)

import matplotlib.pyplot as plt
color = plt.get_cmap("prism", 50)

print(pd.Series(y_label.unique()).sort_values().values)
