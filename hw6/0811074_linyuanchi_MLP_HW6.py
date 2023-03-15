# %%
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# %%
class mlp_classifier:
    def __init__(self, hidden_layer, solver_type='adam', activation_type='relu'):
        self.__model = MLPClassifier(hidden_layer_sizes=hidden_layer, solver=solver_type, activation=activation_type, max_iter=10000)
        self.__hidden_layer_sizes = hidden_layer
        self.__solver = solver_type
        self.__activation = activation_type

    def train(self, x_data, y_data):
        self.__model.fit(x_data, y_data)

    def score(self, x_data, y_data):
        test_score = self.__model.score(x_data, y_data)
        print(f'MLP (hidden layer sizes:{self.__hidden_layer_sizes} solver:{self.__solver} activation:{self.__activation}) training score: {test_score:.3f}')

# %%
class svc_classifier:
    def __init__(self, c_val, gamma_val, kernel_type='rbf'):
        self.__model = SVC(C=c_val, gamma=gamma_val, kernel=kernel_type)
        self.__C = c_val
        self.__gamma = gamma_val
        self.__kernel = kernel_type

    def train(self, x_data, y_data):
        self.__model.fit(x_data, y_data)

    def score(self, x_data, y_data):
        test_score = self.__model.score(x_data, y_data)
        print(f'SVC (C:{self.__C} gamma:{self.__gamma} kernel:{self.__kernel}) training score: {test_score:.3f}')


# %%
raw_data = pd.read_csv('./hw6_haberman.csv')
raw_data = raw_data.to_numpy()

raw_x = raw_data[:,0:3]
raw_y = raw_data[:,3]
del raw_data

# %%
mlp_1 = mlp_classifier(hidden_layer=(500,100), solver_type='lbfgs', activation_type='relu')
mlp_1.train(raw_x, raw_y)
mlp_1.score(raw_x, raw_y)

svc_1 = svc_classifier(c_val=5, gamma_val=0.1, kernel_type='rbf')
svc_1.train(raw_x, raw_y)
svc_1.score(raw_x, raw_y)


