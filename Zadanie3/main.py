import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
le = LabelEncoder()
sc_X = StandardScaler()

dataset = ""
X = ""
Y = ""

def import_dataset():
    print("Reading table")
    return pd.read_csv('./Data/Data.csv')


def parse_x():
    print("Parsing X")
    return dataset.iloc[:, :-1].values


def parse_y():
    print("Parsing Y")
    return dataset.iloc[:, 3].values


def fill(X):
    print("Filling")
    imputer.fit(X[:, 1:3])
    return imputer.transform(X[:, 1:3])


def transform(X):
    print("Transforming country")
    return np.array(ct.fit_transform(X))


def encode(Y):
    print("Encoding Y")
    return le.fit_transform(Y)


def split(X, Y):
    print("Splitting into train and test datasets")
    return train_test_split(X, Y, test_size=0.2, random_state=1)


def scaling_fit_x(x):
    print("Fit scaling X")
    return sc_X.fit_transform(x[:, 3:])


def scaling_x(x):
    print("Scaling X")
    return sc_X.transform(x[:, 3:])


if __name__ == '__main__':
    dataset = import_dataset()
    print(dataset)
    X = parse_x()
    print(X)
    Y = parse_y()
    print(Y)
    X[:, 1:3] = fill(X)
    print(X)
    X = transform(X)
    print(X)
    Y = encode(Y)
    print(Y)
    X_train, X_test, Y_train, Y_test = split(X, Y)
    print(X_train, X_test, Y_train, Y_test)
    X_train[:, 3:] = scaling_fit_x(X_train)
    print(X_train)
    X_test[:, 3:] = scaling_x(X_test)
    print(X_test)