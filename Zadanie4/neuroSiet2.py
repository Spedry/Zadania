import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

classifier = LogisticRegression(random_state=0)
classifier2 = keras.Sequential([
    keras.layers.Dense(6, input_dim=2, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
sc = StandardScaler()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

dataset = ""
X = ""
Y = ""


def import_dataset():
    print("Reading table")
    return pd.read_csv('Data/Social_Network_Ads.csv')


def parse_x():
    print("Parsing X")
    return dataset.iloc[:, 0:2].values


def parse_y():
    print("Parsing Y")
    return dataset.iloc[:, 2].values


def split(X, Y, size, rstate):
    print("Splitting into train and test datasets")
    return train_test_split(X, Y, test_size=size, random_state=rstate)


def scaling_fit_x(x):
    print("Fit scaling X")
    return sc.fit_transform(x)


def scaling_x(x):
    print("Scaling X")
    return sc.transform(x)


def fit(X_train, Y_train):
    print("Fitting classifier")
    classifier.fit(X_train, Y_train)


def compile(X_train, Y_train, X_test, Y_test):
    classifier2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier2.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))


def conf1(X_test, Y_test):
    y_pred1 = classifier.predict(X_test)
    return confusion_matrix(Y_test, y_pred1)


def conf2(X_test, Y_test):
    y_pred2 = classifier2.predict(X_test)
    y_pred2 = (y_pred2 > 0.5)
    return confusion_matrix(Y_test, y_pred2)


def visualize(cm1, cm2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(cm1, cmap=plt.cm.Blues, annot=True, fmt='g', ax=ax1)
    ax1.set_xlabel('Predpovedaná trieda')
    ax1.set_ylabel('Skutočná trieda')
    ax1.set_title('Matica zámeny - Logistic Regression (Churn_Modelling.csv)')

    sns.heatmap(cm2, cmap=plt.cm.Blues, annot=True, fmt='g', ax=ax2)
    ax2.set_xlabel('Predpovedaná trieda')
    ax2.set_ylabel('Skutočná trieda')
    ax2.set_title('Matica zámeny - Neural Network (Churn_Modelling.csv)')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset = import_dataset()
    print(dataset)

    X = parse_x()
    print(X)
    Y = parse_y()
    print(Y)

    X_train, X_test, Y_train, Y_test = split(X, Y, 0.25, 0)
    print(X_train, X_test, Y_train, Y_test)
    X_train = scaling_fit_x(X_train)
    print(X_train)
    X_test = scaling_x(X_test)
    print(X_test)

    fit(X_train, Y_train)

    history = compile(X_train, Y_train, X_test, Y_test)

    cm1 = conf1(X_test, Y_test)
    cm2 = conf2(X_test, Y_test)

    visualize(cm1, cm2)
