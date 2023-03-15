import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class gradient_boost_reg:
    def __init__(self, estimators_val=100, learning_rate_val=0.1):
        self.__model = GradientBoostingClassifier(n_estimators=estimators_val, learning_rate=learning_rate_val)
        self.__estimator = estimators_val
        self.__learning_rate = learning_rate_val
        self.__cross_valid_train_score = None
        self.__cross_valid_valid_score = None

    def train(self, x_data, y_data):
        self.__model.fit(x_data, y_data)

    def predict(self, x_data):
        return self.__model.predict(x_data)

    def score(self, x_data, y_data):
        return self.__model.score(x_data, y_data)

    def cross_valid(self, x_train, y_train):

        split_len = int(len(x_train)/5)

        x_split = []
        y_split = []

        for i in range(4):
            x_split.append(x_train[i*split_len:(i+1)*split_len, :])
            y_split.append(y_train[i*split_len:(i+1)*split_len])

        x_split.append(x_train[4*split_len:, :])
        y_split.append(y_train[4*split_len:])

        valid_score = float(0)
        train_score = float(0)

        for i in range(5):
            
            is_array_exsist = False

            for j in range(5):
                if(not is_array_exsist):
                    is_array_exsist = True
                    x_valid_train = np.array(x_split[j])
                    y_valid_train = np.array(y_split[j])
                    continue

                if(j == i):
                    pass
                else:
                    x_valid_train = np.vstack((x_valid_train, x_split[j]))
                    y_valid_train = np.hstack((y_valid_train, y_split[j]))

            self.train(x_valid_train, y_valid_train)
            valid_score = valid_score  + self.__model.score(x_split[i], y_split[i])
            train_score = train_score + self.__model.score(x_valid_train, y_valid_train)

            # print(f'Ridge (alpha {self.ridge_alpha}) Boston, fold {i}, train/test score: ', end='')
            # print(f'{self.__model.score(x_valid_train, y_valid_train):.2f}/{self.__model.score(x_split[i], y_split[i]):.2f}')

        self.__model.fit(x_train, y_train)
        self.__cross_valid_train_score = float(train_score/5)
        self.__cross_valid_valid_score = float(valid_score/5)

        print(f'gradient_boost_regression (estimator: {self.__estimator} learning_rate: {self.__learning_rate}) 5-fold cross validation train/test: {self.__cross_valid_train_score:.3f}/{self.__cross_valid_valid_score:.3f}')

        return [self.__cross_valid_train_score, self.__cross_valid_valid_score]