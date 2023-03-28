import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

dataset = ""
X = ""
Y = ""


def import_dataset():
    print("Reading table")
    return pd.read_csv('./Data/dataset_c3.csv')


def parse_x():
    print("Parsing X")
    return dataset.iloc[:, 1:2].values


def parse_y():
    print("Parsing Y")
    return dataset.iloc[:, -1].values


def split(X, Y, size):
    print("Splitting into train and test datasets")
    return train_test_split(X, Y, test_size=size, random_state=1)


def regression(X_train, Y_train):
    print("Fitting regressor")
    regressor.fit(X_train, Y_train)


def predict(X_test):
    print("Predicting")
    return regressor.predict(X_test)


def visualize(X_train, Y_train, set):
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Známka na základe hodín učenia sa (' + set + ')')
    plt.xlabel('Počet hodín učenia sa')
    plt.ylabel('Známka')
    plt.show()


if __name__ == '__main__':
    dataset = import_dataset()
    print(dataset)
    X = parse_x()
    print(X)
    Y = parse_y()
    print(Y)
    X_train, X_test, Y_train, Y_test = split(X, Y, 0.2)
    print(X_train, X_test, Y_train, Y_test)
    regression(X_train, Y_train)
    print(regressor)
    Y_pred = predict(X_test)
    print(Y_pred)
    visualize(X_train, Y_train, "Training set")
    visualize(X_test, Y_test, "Test set")
