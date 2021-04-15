import pandas as pd
import numpy as np
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
train = pd.read_csv("dataset_park_median.csv")

train.head()
print(train.head())

y = train["Classification"]

train = train.drop(["Classification", "Unnamed: 0"] , 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["S"] = le.fit_transform(train["S"])
y_label = pd.Series(y.copy())
y = le.fit_transform(y)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)

y_train_label, y_valid_label = train_test_split(y_label, test_size=0.2, random_state=38, stratify=y_label)
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



pred = []
i = 0
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)
for train_index, valid_index in skf.split(X_train, y_train):
    i += 1
    X_train2, X_valid2 = X_train[train_index], X_train[valid_index]
    y_train2, y_valid2 = y_train[train_index], y_train[valid_index]
    dlmodel = Sequential()
    dlmodel.add(Dense(380, input_dim=train.shape[1], activation="relu"))
    dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(128, activation="relu"))
    dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(39, activation="softmax"))
    dlmodel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    earlystop = EarlyStopping(patience=10, verbose=1)
    modelcheck = ModelCheckpoint(str(i)+"best.h5", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(patience=7, verbose=1)
    dlmodel.fit(X_train2, y_train2, validation_data=(X_valid2, y_valid2), epochs=50, batch_size=32,
                callbacks=[earlystop, modelcheck, reducelr])
    dlmodel.load_weights(str(i)+"best.h5")
    pred.append(dlmodel.predict(X_test))
pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y_label.unique()).idxmax(1)

print(np.mean(pred, axis=0))
pd.DataFrame(np.mean(pred, axis=0)).to_csv("Deeplearning_final_TOP5_1.csv", index= False)

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


