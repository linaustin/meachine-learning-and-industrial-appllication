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
        
        