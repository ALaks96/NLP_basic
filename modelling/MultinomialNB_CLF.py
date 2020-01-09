import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def fit_clf(x_train, y_train):

    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    return clf


def predict(x_train, y_train, x_test):

    clf = fit_clf(x_train, y_train)
    results = clf.predict(x_test)

    return results


def assess(results, y_test):

    acc = accuracy_score(y_test, results)
    prec = precision_score(y_test, results)
    sens = recall_score(y_test, results)

    print("Accuracy :", acc)
    print("Precision :", prec)
    print("Sensitivity :", sens)

    confusion = confusion_matrix(y_test, results)
    sns.heatmap(confusion, annot=True, cbar=False, square=True, cmap='Blues', fmt='g', linewidths=0.5)
