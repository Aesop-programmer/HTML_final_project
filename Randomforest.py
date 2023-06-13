import numpy as np
import csv
import pandas as pd
import copy
import math
from ycimpute.imputer import knnimput
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def valid(test_y, y):
    Ein = 0
    for i in range(len(y)):
        Ein = Ein + abs(int(float(y[i]))-int(float(test_y[i])))
    return (Ein/len(y))


def normalize_knn(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if(X[i][j] != np.nan):
                X[i][j] = (X[i][j]-mean[j])/std[j]
    X = np.array(X)
    X = knnimput.KNN(k=15).complete(X)
    return X


def com(feature, composer, data):

    for i in range(1, len(data)):
        for j in range(len(composer)-1):
            feature[i-1].append(0)
    for i in range(1, len(data)):
        one_hot = composer.index(data[i][-2])
        if(one_hot != 0):
            feature[i-1][-one_hot] = 1
        else:
            for j in range(1, len(composer)):
                feature[i-1][-j] = np.nan
    return feature


with open('train.csv', newline='', encoding="utf-8") as csvfile:
    train = csv.reader(csvfile)
    train = list(train)

composer = []
for i in range(1, len(train)):
    if(train[i][-2] not in composer):
        composer.append(train[i][-2])

artistic = []
for i in range(1, len(train)):
    if(train[i][-1] not in artistic):
        artistic.append(train[i][-1])
artistic_weight = []
for i in range(len(artistic)):
    weight = []
    if(artistic[i] == ''):
        artistic_weight.append(np.nan)
    else:
        for j in range(1, len(train)):
            if(train[j][-1] == artistic[i]):
                weight.append(float(train[j][0]))
        artistic_weight.append(sum(weight)/len(weight))
pick = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 23]

feature = []
for i in range(1, len(train)):
    feature.append([])
    for j in pick:
        feature[i-1].append(train[i][j])
for i in range(len(feature)):
    for j in range(len(feature[i])):
        if(feature[i][j] != '' and feature[i][j] != 'False' and feature[i][j] != 'True'):
            feature[i][j] = float(feature[i][j])
        elif(feature[i][j] == 'False'):
            feature[i][j] = 0
        elif(feature[i][j] == 'True'):
            feature[i][j] = 1
        else:
            feature[i][j] = np.nan

feature = com(feature, composer, train)

for i in range(1, len(train)):
    if(train[i][14] == 'compilation'):
        feature[i-1].append(1)
        feature[i-1].append(0)
        feature[i-1].append(0)
    elif(train[i][14] == 'album'):
        feature[i-1].append(0)
        feature[i-1].append(1)
        feature[i-1].append(0)
    elif(train[i][14] == 'single'):
        feature[i-1].append(0)
        feature[i-1].append(0)
        feature[i-1].append(1)
    else:
        feature[i-1].append(np.nan)
        feature[i-1].append(np.nan)
        feature[i-1].append(np.nan)

y_train = []
for i in range(1, len(train)):
    y_train.append(int(float(train[i][0])))


with open('test.csv', newline='', encoding="utf-8") as csvfile:
    test = csv.reader(csvfile)
    test = list(test)

pick = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 22]

# 可以選要用那些特徵，0是我們要predict的
# 只選數值或true fale 但且沒有選id
feature_test = []
for i in range(1, len(test)):
    feature_test.append([])
    for j in pick:
        feature_test[i-1].append(test[i][j])

for i in range(len(feature_test)):
    for j in range(len(feature_test[i])):
        if(feature_test[i][j] != '' and feature_test[i][j] != 'False' and feature_test[i][j] != 'True'):
            feature_test[i][j] = float(feature_test[i][j])
        elif(feature_test[i][j] == 'False'):
            feature_test[i][j] = 0
        elif(feature_test[i][j] == 'True'):
            feature_test[i][j] = 1
        else:
            feature_test[i][j] = np.nan
feature_test = com(feature_test, composer, test)

for i in range(1, len(test)):
    if(test[i][13] == 'compilation'):
        feature_test[i-1].append(1)
        feature_test[i-1].append(0)
        feature_test[i-1].append(0)
    elif(test[i][13] == 'album'):
        feature_test[i-1].append(0)
        feature_test[i-1].append(1)
        feature_test[i-1].append(0)
    elif(test[i][13] == 'single'):
        feature_test[i-1].append(0)
        feature_test[i-1].append(0)
        feature_test[i-1].append(1)
    else:
        feature_test[i-1].append(np.nan)
        feature_test[i-1].append(np.nan)
        feature_test[i-1].append(np.nan)


for i in range(len(feature)):
    feature[i].append(artistic_weight[artistic.index(train[i+1][-1])])
for i in range(len(feature_test)):
    feature_test[i].append(artistic_weight[artistic.index(test[i+1][-1])])

feature = feature + feature_test
print(len(feature[0]))
print(feature[0])

feature = np.array(feature)

feature = normalize_knn(feature)
feature_train = feature[0:len(train)-1]
feature_test = feature[len(train)-1:len(feature)]
print("end")
# feature_train 就是training 的feature
# y_train是 training的y
# feature_test 就是test的


# 最上層模型
model_classifier = RandomForestClassifier(
    n_estimators=1111, oob_score=True, random_state=910123, max_features=10)
model_classifier.fit(feature_train, y_train)

y = model_classifier.oob_decision_function_


y_median = []
for i in range(len(y)):
    thresholde = 0
    for j in range(10):
        thresholde = y[i][j] + thresholde
        if(thresholde > 0.5):
            y_median.append(j)
            break
y = y_median
print("oob=", valid(y, y_train),)

feature_train = feature_train.tolist()
for i in range(len(feature_train)):
    feature_train[i].append(y_median[i])
feature_train = np.array(feature_train)
model_classifier_2 = RandomForestClassifier(
    n_estimators=520, oob_score=True, random_state=910123, max_features=10)
model_classifier_2.fit(feature_train, y_train)
y = model_classifier_2.oob_decision_function_
y_median = []
for i in range(len(y)):
    thresholde = 0
    for j in range(10):
        thresholde = y[i][j] + thresholde
        if(thresholde > 0.5):
            y_median.append(j)
            break
y = y_median
print("oob_level2=", valid(y, y_train), "n_estimaotr=", j)
# 預測實際值
y = model_classifier.predict_proba(feature_test)
y_median = []
for i in range(len(y)):
    thresholde = 0
    for j in range(10):
        thresholde = y[i][j] + thresholde
        if(thresholde > 0.5):
            y_median.append(
                j)
            break
y = y_median

feature_test = feature_test.tolist()
for i in range(len(feature_test)):
    feature_test[i].append(y_median[i])
feature_test = np.array(feature_test)
y = model_classifier_2.predict_proba(feature_test)

y_median = []
for i in range(len(y)):
    thresholde = 0
    for j in range(10):
        thresholde = y[i][j] + thresholde
        if(thresholde > 0.5):
            y_median.append(
                j)
            break
y = y_median


with open("predict_classifier_median_with_string_alice_two_level_Alice_530.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "Danceability"])
    for i in range(len(y)):
        writer.writerow([17170+i, y[i]])
