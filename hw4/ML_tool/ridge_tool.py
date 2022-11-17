from sklearn.linear_model import Ridge
import numpy as np

class ridge_regression:
    def __init__(self, reg_alpha):
        self.__model = Ridge(alpha=reg_alpha)
        self.ridge_alpha = reg_alpha

    def train(self, x_train, y_train):
        self.__model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.__model.predict(x_test)

    def score(self, x_test, y_test):
        return self.__model.score(x_test, y_test)

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
            
            is_array_exits = False

            for j in range(5):
                if(not is_array_exits):
                    is_array_exits = True
                    x_valid_train = np.array(x_split[j])
                    y_valid_train = np.array(y_split[j])
                    continue

                if(j == i):
                    pass
                else:
                    x_valid_train = np.vstack((x_valid_train, x_split[j]))
                    y_valid_train = np.hstack((y_valid_train, y_split[j]))

            self.__model.fit(x_valid_train, y_valid_train)
            valid_score = valid_score  + self.__model.score(x_split[i], y_split[i])
            train_score = train_score + self.__model.score(x_valid_train, y_valid_train)

            print(f'Ridge (alpha {self.ridge_alpha}) Boston, fold {i}, train/test score: ', end='')
            print(f'{self.__model.score(x_valid_train, y_valid_train):.2f}/{self.__model.score(x_split[i], y_split[i]):.2f}')

        self.__model.fit(x_train, y_train)

        return [(train_score/5), (valid_score/5)]