# %%
from ML_tool import file_tool
from ML_tool import ridge
from sklearn.model_selection import train_test_split
import numpy as np

# %%
data_file = file_tool.input_file('./hw4_boston.csv')
data_file.read_from_file()
raw_x_data, raw_y_data = data_file.sort_boston_dataset()

raw_x_data = np.array(raw_x_data)
raw_y_data = np.array(raw_y_data)

print(raw_x_data.shape)

# %%
x_train, x_test, y_train, y_test = train_test_split(raw_x_data, raw_y_data, test_size=0.169)

print(x_train.shape)
print(x_test.shape, end='\n\n')

ridge_reg = ridge.ridge_regression(1)
train_score, valid_score = ridge_reg.cross_valid(x_train, y_train)
print('')

print(f'Ridge (alpha {ridge_reg.ridge_alpha}) Boston, 5-fold train/test average score: ', end='')
print(f'{train_score:.2f}/{valid_score:.2f}')
print()

print(f'Ridge (alpha {ridge_reg.ridge_alpha}) Boston, 5-fold/verify score: ', end='')
print(f'{train_score:.2f}/{ridge_reg.score(x_test, y_test):.2f}')


