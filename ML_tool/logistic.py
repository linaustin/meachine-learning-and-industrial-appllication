import numpy as np
from sklearn.linear_model import LogisticRegression

class logistic_reg:
    def __init__(self, penalty_val, max_iter_val, C_val=1, fit_intercept_val=True, random_state_val=None):
        self.__model = LogisticRegression(C=C_val, penalty=penalty_val, solver='liblinear', max_iter=max_iter_val, fit_intercept=fit_intercept_val, random_state=random_state_val)
        self.__max_iter = max_iter_val
        self.__solver = 'liblinear'
        self.__C = C_val
        self.__penalty=penalty_val
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

        print(f'logistic regression (C: {self.__C} max_iter: {self.__max_iter} penalty: {self.__penalty}) 5-fold cross validation train/test: {self.__cross_valid_train_score:.3f}/{self.__cross_valid_valid_score:.3f}')

        return [self.__cross_valid_train_score, self.__cross_valid_valid_score]