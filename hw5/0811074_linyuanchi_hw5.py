# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# %%
class input_file:
    def __init__(self, file_path):
        self.__path = str(file_path)
        self.__raw_data = '' 

    def read_from_file(self):
        file = open(self.__path, 'r')
        self.__raw_data = file.readlines()
        file.close()

        for i in range(len(self.__raw_data)):
            self.__raw_data[i] = self.__raw_data[i].rstrip('\n')

    def get_raw_data(self):
        return self.__raw_data

    def sort_iris_data(self):
        temp = []

        for data in self.__raw_data:
            data = data.rstrip(', ')
            data = data.split(',')

            for i in range(len(data)):
                try:
                    data[i] = float(data[i])
                except:
                    print(f'raw data cant cast to float : {data[i]}')
                    return None

            temp.append(data)

        return temp

    def sort_wave_dataset(self):
        x_data = []
        y_data = []
        
        for i in range(len(self.__raw_data)):
            self.__raw_data[i] = self.__raw_data[i].strip()
            self.__raw_data[i] = self.__raw_data[i].lstrip('[')
            self.__raw_data[i] = self.__raw_data[i].rstrip(']')
            self.__raw_data[i] = self.__raw_data[i].strip()

        x_head = self.__raw_data.index('X inputs:')
        y_head = self.__raw_data.index('y target:')

        for data in self.__raw_data[x_head+1:y_head]:
            x_data.append(float(data))
        
        for data in self.__raw_data[y_head+1:]:
            data = data.split()
            for number in data:
                y_data.append(float(number))

        return [x_data, y_data]
            
    def sort_boston_dataset(self):
        x_data = []
        y_data = []

        for data in self.__raw_data:
            data = data.strip(',')
            data = data.split(',')
            temp = []

            for i in range(len(data)):
                if i != (len(data) - 1):
                    temp.append(float(data[i]))
                else:
                    y_data.append(float(data[i]))

            x_data.append(temp)        

        return [x_data, y_data]
    
    def sort_forge_dataset(self):
        x_data = []
        y_data = []

        for i in range(len(self.__raw_data)):
            self.__raw_data[i] = self.__raw_data[i].strip()
            self.__raw_data[i] = self.__raw_data[i].lstrip('[')
            self.__raw_data[i] = self.__raw_data[i].rstrip(']')
            self.__raw_data[i] = self.__raw_data[i].strip()

        x_head = self.__raw_data.index('X inputs:')
        y_head = self.__raw_data.index('y target:')

        for line in self.__raw_data[x_head+1: y_head]:
            raw = line.split()
            temp = []

            for data in raw:
                temp.append(float(data))

            x_data.append(temp)

        for line in self.__raw_data[y_head+1:]:
            raw = line.split()
            
            for data in raw:
                y_data.append(float(data))
        

        return [x_data, y_data]

    def sort_cancer_dataset(self):
        x_data = []
        y_data = []

        for data in self.__raw_data[1:]:
            data = data.split(',')

            for i in range(len(data)):
                data[i] = float(data[i])

            temp = data[0:30]

            x_data.append(temp)
            y_data.append(data[30])

        return [x_data, y_data]

# %%
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

# %%
class random_forest:
    def __init__(self, estimators_val=100, max_feature_val='auto', max_depth_val=None):
        self.__model = RandomForestClassifier(n_estimators=estimators_val, max_features=max_feature_val, max_depth=max_depth_val)
        self.__estimator = estimators_val
        self.__max_feature = max_feature_val
        self.__max_depth = max_depth_val
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

        print(f'random forests (estimators: {self.__estimator} max_depth: {self.__max_depth} max_feature: {self.__max_feature}) 5-fold cross validation train/test: {self.__cross_valid_train_score:.3f}/{self.__cross_valid_valid_score:.3f}')

        return [self.__cross_valid_train_score, self.__cross_valid_valid_score]

# %%
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

# %%
if __name__ == '__main__':
    data_file = input_file('./hw5_cancer.csv')
    data_file.read_from_file()
    raw_xdata, raw_ydata = data_file.sort_cancer_dataset()
    
    raw_xdata = np.array(raw_xdata)
    raw_ydata = np.array(raw_ydata)
    
    x_train, x_test, y_train, y_test = train_test_split(raw_xdata, raw_ydata, test_size=94)

    logistic_reg1 = logistic_reg(penalty_val='l2', max_iter_val=1000)
    train_avg, valid_avg = logistic_reg1.cross_valid(x_train, y_train)
    score = logistic_reg1.score(x_test, y_test)
    print(f'logistic_regression test score:{score:.3f}')

    random_forest1 = random_forest(estimators_val=100)
    train_avg, valid_avg = random_forest1.cross_valid(x_train, y_train)
    score = random_forest1.score(x_test, y_test)
    print(f'random_forest test score:{score:.3f}')

    gradient_boost_reg1 = gradient_boost_reg(estimators_val=100, learning_rate_val=0.1)
    train_avg, valid_avg = gradient_boost_reg1.cross_valid(x_train, y_train)
    score = gradient_boost_reg1.score(x_test, y_test)
    print(f'gradient_boost_regression test score:{score:.3f}')