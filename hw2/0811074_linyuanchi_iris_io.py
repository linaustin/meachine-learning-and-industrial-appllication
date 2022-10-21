from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

input_file = open('iris_dataset.txt', 'r')

raw_data = input_file.readlines()

for i in range(len(raw_data)):
    raw_data[i] = raw_data[i].lstrip(' [')
    raw_data[i] = raw_data[i].rstrip(']\n')
    raw_data[i] = raw_data[i].strip()

inputs_tag = raw_data.index('all inputs:')
target_tag = raw_data.index('all targets:')

raw_inputs = raw_data[inputs_tag+1:target_tag]
raw_target = raw_data[target_tag+1:]

input_file.close()

count = 0

for data in raw_inputs:
    count+=1

    temp = data.split()
    
    for i in range(len(temp)):
        temp[i] = float(temp[i])

    if(count == 1):
        inputs = np.array(temp)
    else:
        np_temp = np.array(temp)
        inputs = np.vstack((inputs,np_temp))

count = 0

for data in raw_target:
    temp = data.split()

    for i in range(len(temp)):
        temp[i] = float(temp[i])
    
    for element in temp:
        count+=1

        if(count ==  1):
            targets = np.array([element])
        else:
            np_temp = np.array(element)
            targets = np.vstack((targets,np_temp))

input_data = np.hstack((inputs,targets))


input_median = np.median(input_data, axis=0)

target_count = [0,0,0]

for i in input_data:
    target_count[int(i[4])] += 1


print(f"species count : {target_count}")

for i in range(4):
    print(f"feature {i}, median = {input_median[i]:.2f}")

feature = np.array(range(4))
plt.bar(feature,input_median[0:4], width=0.5, color='blue', label="feature/median")
plt.legend()
plt.xlabel("feature")
plt.ylabel("median")
plt.show()

output_file = open("./0811074_linyuanchi_iris_data.csv", 'w')

np.random.shuffle(input_data)

for data in input_data:

    for i in range(len(data)):
        output_file.write(f"{data[i]},")
    
    output_file.write("\n")
    

output_file.close()


