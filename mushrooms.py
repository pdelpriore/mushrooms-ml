import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

mushrooms = pd.read_csv("./dataset/mushrooms.csv")

# pd.set_option("display.max_column", mushrooms.shape[1])

# print(mushrooms.head())

# print(mushrooms.describe())

# sns.heatmap(mushrooms.isna())

# print(mushrooms.isna().sum())

# print(np.sum(mushrooms["stalk-root"] == "?", axis=0))

# print(mushrooms.shape)

# mushrooms = mushrooms[mushrooms["stalk-root"] != "?"]

# mushrooms = mushrooms.drop("stalk-root", axis=1)

def impute(data):
    return data.drop(["stalk-root", "veil-type"], axis=1)

def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label="train score")
    plt.plot(N, val_score.mean(axis=1), label="validation score")
    plt.legend()

df_imputed = impute(mushrooms)

X = df_imputed.drop("class", axis=1)
y = df_imputed["class"]

X_encoder = OneHotEncoder()
y_encoder = LabelBinarizer()

X_encoder.fit(X)
X_encoded = X_encoder.transform(X)

y_encoded = y_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=0)

"""
model_list = {
    "LinearSVC": LinearSVC(random_state=0),
    "SVC": SVC(random_state=0),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Ada Boost": AdaBoostClassifier(random_state=0)
    }

for name, model in model_list.items():
    print(name)
    evaluate(model, X_train, y_train.ravel(), X_test, y_test)
"""

model = KNeighborsClassifier()
model.fit(X_train, y_train.ravel())

# print(model.score(X_test, y_test))





