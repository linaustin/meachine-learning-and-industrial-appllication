from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

class knn_classifier:
    def __init__(self, neighbor_number):
        self.__knn_model = KNeighborsClassifier(n_neighbors=neighbor_number)

    def train(self, x_train, y_train):
        self.__knn_model.fit(x_train, y_train)
    
    def predict(self, test_data):
        temp = self.__knn_model.predict(test_data)
        return temp

    def score(self, x_test, y_test):
        return self.__knn_model.score(x_test, y_test)

class knn_regresor:
    def __init__(self):
        pass

    