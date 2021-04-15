import pandas as pd
import numpy as np
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
train = pd.read_csv("dataset_park_median.csv")
#train = pd.read_csv("dataset_park_new_2.csv", encoding="CP949" )
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
y_label = pd.Series(y.copy())
y = le.fit_transform(y)


from sklearn.preprocessing import StandardScaler
#모든 column 들의 범위를 -1~1로 만들어 준다. 평균이 0이고 분산이 1인 데이터의 분포
sc = StandardScaler()
train = sc.fit_transform(train)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=38, stratify=y)
#print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
#print(y_valid.value_counts())

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
    # dlmodel.add(Dense(256, activation="relu"))
    # dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(128, activation="relu"))
    dlmodel.add(Dropout(0.5))
    dlmodel.add(Dense(39, activation="softmax"))
    # dlmodel.add(Dense(39, activation="sigmoid"))
    dlmodel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    earlystop = EarlyStopping(patience=10, verbose=1)
    modelcheck = ModelCheckpoint(str(i)+"best.h5", save_best_only=True, verbose=1)
    reducelr = ReduceLROnPlateau(patience=7, verbose=1)
    dlmodel.fit(X_train2, y_train2, validation_data=(X_valid2, y_valid2), epochs=50, batch_size=32,
                callbacks=[earlystop, modelcheck, reducelr])
    #epochs최적화는 50이다. 어자피 중간에 멈춤
    #batch_size = 기본값이 32개가 기본임.
    dlmodel.load_weights(str(i)+"best.h5")
    pred.append(dlmodel.predict(X_test))
#pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=pd.Series(y.unique()).sort_values()).idxmax(1)
pred_df = pd.DataFrame(np.mean(pred, axis=0), columns=y_label.unique()).idxmax(1)

print(np.mean(pred, axis=0))
#pd.DataFrame(pred)
pd.DataFrame(np.mean(pred, axis=0)).to_csv("Deeplearning_final_TOP5_1.csv", index= False)

from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(pred_df, y_valid_label))
#print(pred_df)
#print(y_valid_label)

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
        #top5_list.append(-1)
        top5_list.append(top_values[0])

print("top5 acc", top5/len(labels))

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
print(accuracy_score(top5_list, le.fit_transform(y_test)))

print(classification_report(top5_list, le.fit_transform(y_test)))

# import shap
# import matplotlib.pyplot as plt
# explainer = shap.TreeExplainer(xgb_cf).shap_values(X_train2)
# shap.summary_plot(explainer, X_train2)
'''pred_df = pd.DataFrame(pred, columns=y_label.unique()).idxmax(1)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
print(pred_df)
print(confusion_matrix(pred_df, y_valid2))

c_map = pd.DataFrame(confusion_matrix(y_valid2, pred_df), index=pd.Series(y_label.unique()).sort_values().values,
                     columns=pd.Series(y_label.unique()).sort_values().values)
_ , axis = plt.subplots(1,1, figsize = (12, 12))
sns.heatmap(c_map, annot=True, cmap="YlGnBu")
plt.title("Multiclass_accuracy: {0:.4f}".format(accuracy_score(pred_df, y_valid2)))
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.show()'''
