from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class knn_classifier:
    def __init__(self, neighbor_number):
        self.__knn_classifier = KNeighborsClassifier(n_neighbors=neighbor_number)

    def train(self, x_train, y_train):
        self.__knn_classifier.fit(x_train, y_train)
    
    def predict(self, test_data):
        return self.__knn_classifier.predict(test_data)

    def score(self, x_test, y_test):
        return self.__knn_classifier.score(x_test, y_test)


class knn_regresor:
    def __init__(self, neighbor_number):
        self.__knn_regresor = KNeighborsRegressor(n_neighbors= neighbor_number)

    def train(self, x_train, y_train):
        x_train = x_train.reshape(-1, 1)
        self.__knn_regresor.fit(x_train, y_train)

    def predict(self, test_data):
        test_data = test_data.reshape(-1, 1)
        return self.__knn_regresor.predict(test_data)

    def score(self, x_test, y_test):
        x_test = x_test.reshape(-1, 1)
        return self.__knn_regresor.score(x_test, y_test)

    