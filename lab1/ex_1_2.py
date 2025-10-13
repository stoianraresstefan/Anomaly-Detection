import pyod
from pyod.models.knn import KNN

import matplotlib.pyplot as pyplot
import sklearn.metrics as sk


def main():
    ### EXERCISE 1
    dataset = pyod.utils.data.generate_data(
        n_train=400, n_test=100, n_features=2, contamination=0.1
    )

    X_train, X_test, y_train, y_test = dataset

    inliers = y_train == 0
    outliers = y_train == 1

    pyplot.scatter(
        X_train[inliers][:, 0], X_train[inliers][:, 1], c="b", label="Inlier"
    )
    pyplot.scatter(
        X_train[outliers][:, 0],
        X_train[outliers][:, 1],
        c="r",
        label="Outlier",
    )
    pyplot.legend()
    pyplot.show()

    ### EXERCISE 2
    model = KNN(contamination=0.1)

    model.fit(X_train)

    y_test_pred = model.predict(X_test)

    matrix = sk.confusion_matrix(y_test, y_test_pred)

    print("Confusion Matrix:\n", matrix)


if __name__ == "__main__":
    main()
