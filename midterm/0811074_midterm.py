# %%
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

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

# %%
class knn_regresor:
    def __init__(self, neighbor_number, weight_type):
        self.__knn_regresor = KNeighborsRegressor(n_neighbors= neighbor_number, weights = weight_type)
        self.neighbor_number = neighbor_number
        self.weight = weight_type

    def train(self, x_train, y_train):
        x_train = x_train.reshape(-1, 1)
        self.__knn_regresor.fit(x_train, y_train)

    def predict(self, test_data):
        test_data = test_data.reshape(-1, 1)
        return self.__knn_regresor.predict(test_data)

    def score(self, x_test, y_test):
        x_test = x_test.reshape(-1, 1)
        return self.__knn_regresor.score(x_test, y_test)

# %%
if __name__ == '__main__':
    wave_file = input_file('./wave60_dataset.txt')
    wave_file.read_from_file()
    raw_x_data, raw_y_data = wave_file.sort_wave_dataset()

    raw_x_data = np.array(raw_x_data)
    raw_y_data = np.array(raw_y_data)

    x_train, x_test, y_train, y_test = train_test_split(raw_x_data, raw_y_data, test_size=0.16)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    print(x_train.shape)
    print(x_test.shape)

    # %%
    knn_uniform = []
    knn_distance = []

    for i in range (1,10,2):
        knn_uniform.append(knn_regresor(i, 'uniform'))
        knn_distance.append(knn_regresor(i, 'distance'))

    for i in range(len(knn_uniform)):
        knn_uniform[i].train(x_train, y_train)
        knn_distance[i].train(x_train, y_train) 

    for i in range(len(knn_uniform)):
        print(f'{knn_uniform[i].weight} , KNN={knn_uniform[i].neighbor_number}, ', end='')
        print('x_test/x_train score = ', end= '')
        print(f'{knn_uniform[i].score(x_test, y_test):.2f}/{knn_uniform[i].score(x_train, y_train):.2f}')

    print()

    for i in range(len(knn_distance)):
        print(f'{knn_distance[i].weight} , KNN={knn_distance[i].neighbor_number}, ', end='')
        print('x_test/x_train score = ', end= '')
        print(f'{knn_distance[i].score(x_test, y_test):.2f}/{knn_distance[i].score(x_train, y_train):.2f}')


