from matplotlib import pyplot as plt
import numpy as np

x1 = np.array([5, 3, 1])
x2 = np.array([-6, -4, -2])
y = np.array([5, 10, 15])

fig , ax = plt.subplots()
ax.plot(x1,y)
