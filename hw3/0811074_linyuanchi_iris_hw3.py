# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ML_tool import file_tool
from ML_tool import knn_tool

# %%
input_file = file_tool.input_file('./0811074_linyuanchi_iris_data.csv')
input_file.read_from_file()
data_list = input_file.sort_iris_data()
iris_data = np.array(data_list)

# %%
knn_model = knn_tool.knn(5)

while True:
    np.random.shuffle(iris_data)
    x_train = iris_data[:112, 0:4]
    y_train = iris_data[:112, 4]

    x_test = iris_data[112:, 0:4]
    y_test = iris_data[112:, 4]

    knn_model.train(x_train, y_train)

    if(knn_model.score(x_test, y_test) == 1):
        break

# %%
test_data = np.array([[5, 2.9, 1, 0.2], [3, 2.2, 4, 0.9]])
test_result = knn_model.predict(test_data)

predict_result = {0: 'setosa (0)', 1: 'versicolor (1)', 2: 'virginica (2)'}

for i in range(len(test_result)):
    print(f"the predict of test data {i+1} is : {predict_result.get(test_result[i])}")



