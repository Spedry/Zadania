import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

classifier = LogisticRegression(random_state=0)
sc = StandardScaler()

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


def predict(X_test):
    print("Predicting")
    return classifier.predict(X_test)


def matrix(Y_test, Y_pred):
    print("Creating matrix")
    return confusion_matrix(Y_test, Y_pred)


def accuracy(Y_test, Y_pred):
    print("Creating matrix")
    return accuracy_score(Y_test, Y_pred)


def visualize(X, Y, set):
    X_set, y_set = X, Y
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                 cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.xlim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)): plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                                                         c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Logistic Regression (' +set+ ' set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
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
    print(predict(sc.transform(
        [[30, 87000]]
    )))
    Y_pred = predict(X_test)
    print(Y_pred)
    print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))
    cm = matrix(Y_test, Y_pred)
    print(cm)
    accuracy = accuracy(Y_test, Y_pred)
    print(accuracy)
    visualize(X_train, Y_train, 'Training')
    visualize(X_test, Y_test, 'Testing')
