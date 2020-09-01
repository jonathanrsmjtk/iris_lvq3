import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier #K-NN Classifier for comparison
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier for comparison
from sklearn.model_selection import KFold

#Directory depends on where you saved / fork this
df = pd.read_csv("/Volumes/Hard Disk/Programming/lvq3_iris/19-420-bundle-archive/Iris.csv")

#drop Id column as it doesn't needed
df = df.drop("Id", axis=1)

#Rename columns
df = df.rename(columns={"SepalLengthCm": "sl", "SepalWidthCm": "sw", "PetalLengthCm": "pl", 
                   "PetalWidthCm":"pw", "Species":"y"})

#Plot using seaborn, but this is only worked on Jupyter.
# sns.set(style = "white", color_codes = True)
# print(sns.FacetGrid(df, hue = "y", size = 5) \
#    .map(plt.scatter, "sl", "sw") \
#    .add_legend())
# print(sns.FacetGrid(df, hue = "y", size = 5) \
#    .map(plt.scatter, "pl", "pw") \
#    .add_legend())

#Data splitting, for random split or not
def dataRandomSet(data, random=None):
    if random != None:
        train_x, test_x, train_y, test_y = train_test_split(df.drop(["y"], axis=1), 
                                                                        df["y"], random_state=random, test_size=0.3)
        test_y = test_y.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        train_x = train_x.reset_index(drop=True)
        test_x = test_x.reset_index(drop=True)

    else:
        train_y = df.loc[0:120, "y"].reset_index(drop=True)
        test_y = df.loc[120: , "y"].reset_index(drop=True)
        train_x = df.iloc[0:120].drop("y", axis=1).reset_index(drop=True)
        test_x = df.iloc[120:].drop("y", axis=1).reset_index(drop=True)
    return train_x, train_y, test_x, test_y

#Train using LVQ 3
def lvq3_train(data, kelas, a, b, max_ep, min_a, e):
    X = data.values
    y = np.array(kelas)
    c, train_idx = np.unique(y, True)
    r = c
    W = X[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(zip(X, y)) if i not in train_idx])
    X = train[:, 0]
    y = train[:, 1]
    ep = 0
#     print("Aktivasi", W)
#     print("Dipakai", X)
    
    while ep < max_ep and a > min_a:
        for i, x in enumerate(X):
            d = [math.sqrt(sum((w - x) ** 2)) for w in W]
            min_1 = np.argmin(d)

            min_2 = 0
            dc = float(np.amin(d))
            dr = 0
            min_2 = d.index(sorted(d)[1])
            dr = float(d[min_2])
            if c[min_1] == y[i] and c[min_1] != r[min_2]:
                W[min_1] = W[min_1] + a * (x - W[min_1])

            elif c[min_1] != r[min_2] and y[i] == r[min_2]:
                if dc != 0 and dr != 0:

                    if min((dc/dr),(dr/dc)) > (1-e) / (1+e):
                        W[min_1] = W[min_1] - a * (x - W[min_1])
                        W[min_2] = W[min_2] + a * (x - W[min_2])
            elif c[min_1] == r[min_2] and y[i] == r[min_2]:
                W[min_1] = W[min_1] + e * a * (x - W[min_1])
                W[min_2] = W[min_2] + e * a * (x- W[min_2])
        a = a * b
        ep += 1
    return W, c

#Test Using LVQ 3
def lvq3_test(x, W):
    
    W, c = W
    d = [math.sqrt(sum((w - x) ** 2)) for w in W]

    return c[np.argmin(d)]

#Evaluation
def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels,
           preds, average = 'weighted')))
    print("Recall Score: {}".format(recall_score(labels, preds,
           average = 'weighted')))
    print("Accuracy Score: {}".format(accuracy_score(labels,
           preds)))
    print("F1 Score: {}".format(f1_score(labels, preds, average =
           'weighted')))

train_x, train_y, test_x, test_y = dataRandomSet(df, 42) #Set random state to 42. You can set it later.
print(train_x.head()) #Print train data after splitted and shuffled

#Get final weight from train
W = lvq3_train(train_x, train_y, 0.3, 0.2, 100, 0.001, 0.3)

#Test and predict
predicted = []
for i in test_x.values:
    predicted.append(lvq3_test(i, W))
print_metrics(test_y, predicted)

#Comparison using K-Fold
def kfold_comparison(train_x, test_x, train_y, test_y):
    lvq3_acc = []
    rfc_acc = []
    knn_acc = []
    kf= KFold(n_splits=5, shuffle=False)
    print(kf)  
    i=1       
    for train_index, test_index in kf.split(df):

        x = train_x.append(test_x).reset_index(drop=True)
        y = train_y.append(test_y).reset_index(drop=True)
        train_x = x.iloc[train_index]
        test_x = x.iloc[test_index]
        train_y = y.loc[train_index]
        test_y = y.loc[test_index]
        W = lvq3_train(train_x, train_y, 0.3, 0.2, 100, 0.001, 0.3)
        predicted = []
        for h in test_x.values:
            predicted.append(lvq3_test(h, W))
        lvq3_acc.append(metrics.accuracy_score(test_y, predicted))
        
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(train_x, train_y)
        predicted = rfc.predict(test_x)
        rfc_acc.append(metrics.accuracy_score(test_y, predicted))

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_x, train_y)
        predicted = knn.predict(test_x) # 0:Overcast, 2:Mild
        knn_acc.append(metrics.accuracy_score(test_y, predicted))
        i+=1
    
    return lvq3_acc, rfc_acc, knn_acc

lvq3_acc, rfc_acc, knn_acc = kfold_comparison(train_x, test_x, train_y, test_y)

#Show plot of comparisons
N = 5

ind = np.arange(N) 
width = 0.18
plt.bar(ind - width, knn_acc, width, label='K-NN')
plt.bar(ind, rfc_acc, width,
    label='Random Forest')
plt.bar(ind + width, lvq3_acc, width,
    label='LVQ3')

plt.ylabel('Accuracy')
plt.xlabel('Fold')
plt.title('K-Fold Cross Validation')

plt.xticks(ind + width / 2, [i+1 for i in range(N)])
plt.legend(loc='best')
plt.show()