import pandas as pd
import numpy as np

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

train = pd.read_csv("dataset_park_new_2.csv", encoding="CP949" )

print(train.isnull().sum().sort_values(ascending=False))

print(train["Classification"].value_counts())

train["FE"] = train["FE"].fillna(train["FE"].median())
print(train.isnull().sum().sort_values(ascending=False))

y = train["Classification"]
train = train.drop("Classification", 1)

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
sc = StandardScaler()
train[train.columns] = sc.fit_transform(train)

print('-'*10)
print(train.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)

y_train_label, y_valid_label = train_test_split(y_label, test_size=0.2, random_state=38, stratify=y_label)
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K


def recall(y_target, y_pred):

    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    count_true_positive_false_negative = K.sum(y_target_yn)

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    return recall


def precision(y_target, y_pred):

    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))

    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    count_true_positive_false_positive = K.sum(y_pred_yn)

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())


    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)

    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())


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
    dlmodel.add(Dense(256, activation="relu"))
    dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(39, activation="softmax"))
    dlmodel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc", f1score])

    earlystop = EarlyStopping(patience=10, verbose=1)
    modelcheck = ModelCheckpoint(str(i) + "best.h5", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(patience=7, verbose=1)
    dlmodel.fit(X_train2, y_train2, validation_data=(X_valid2, y_valid2), epochs=150, batch_size=128,
                callbacks=[earlystop, modelcheck, reducelr])

    dlmodel.load_weights(str(i)+"best.h5")
    pred.append(dlmodel.predict(X_test))

pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y_label.unique()).idxmax(1)

print(np.mean(pred, axis=0))

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
