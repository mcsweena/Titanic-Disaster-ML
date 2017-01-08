import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import itemfreq
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier
sns.set(style="white", color_codes=True)


def imports():
    train = pd.read_csv("train.csv", dtype={"Age": np.float64})
    test = pd.read_csv("test.csv", dtype={"Age": np.float64}, index_col=None)

    return train, test


def reduce_dataset_size(train, test):
    train = train[["Pclass", "Sex", "Age", "Embarked", "Survived"]]
    pass_id = test[['PassengerId']]
    test = test[["Pclass", "Sex", "Age", "Embarked"]]

    return train, test, pass_id


def factorise(train, test):
    """
    The pandas factorize function assigns each unique value in a
    series to a sequential, 0-based index, and calculates which
    index each series entry belongs to.

    """

    train["Sex"] = pd.factorize(train["Sex"])[0]
    test["Sex"] = pd.factorize(test["Sex"])[0]

    return train, test


def factorise_dep(train, test):
    train["Embarked"].fillna('S', inplace=True)
    test["Embarked"].fillna('S', inplace=True)

    train['Embarked'] = pd.factorize(train['Embarked'])[0]
    test['Embarked'] = pd.factorize(test['Embarked'])[0]

    return train, test


def fillnas(train, test):
    train.fillna(train.mean(), inplace=True)
    test.fillna(test.mean(), inplace=True)

    return train, test


def process_data(train, test):
    train_data = pd.DataFrame.as_matrix(train[['Pclass', 'Sex', 'Age', 'Embarked']])
    train_target = pd.DataFrame.as_matrix(train[['Survived']]).ravel()
    test_data = pd.DataFrame.as_matrix(test[['Pclass', 'Sex', 'Age', 'Embarked']])

    return train_data, train_target, test_data


def classify(train_data, train_target, test_data):
    clf = GaussianNB()
    clf.fit(train_data, train_target)
    predictions = clf.predict(test_data).astype(int)
    return predictions


def results(predictions, pass_id):
    test_results = pass_id
    test_results["Survived"] = predictions
    test_results.to_csv('titanic_prediction.csv', index=False)


def main():
    train, test = imports()
    train, test, pass_id = reduce_dataset_size(train, test)
    train, test = factorise(train, test)
    train, test = factorise_dep(train, test)
    train, test = fillnas(train, test)
    train_data, train_target, test_data = process_data(train, test)

    predictions = classify(train_data, train_target, test_data)

    print("\nTest set survival:")
    print(itemfreq(predictions))

    results(predictions, pass_id)
    print("\nComplete!")

if __name__ == "__main__":
    main()
