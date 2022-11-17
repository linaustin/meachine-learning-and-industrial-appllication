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
            



        
        
        