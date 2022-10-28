# pip install scipy
# python -m pip show scikit-learn  (my version 0.23.1)
# python -m pip freeze
# python -c "import sklearn; sklearn.show_versions()"
# pip install --user scikit-learn
# pip show scikit-learn    (new version 0.23.2)

# if import already in Python 3.8.3 Shell


import mglearn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import numpy as np


############################################################
def readForge(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    s = inFile.readline() # skip
    
    row = 0
    while (row < 26):
        s = inFile.readline()
        data1 = s.strip() # remove leading and ending blanks
        if (len(data1) <= 0):
            break
        
        # since we use append, value must be created in the loop
        value = [0., 0.]
        
        data1 = data1.replace('[', '') # remove [
        data1 = data1.replace(']', '') # remove ]
        strs2 = data1.split() # array of 2 str

        # convert to real
        value[0] = eval(strs2[0])
        value[1] = eval(strs2[1])
        
        #print("row = {}".format(row) + ", {}\n".format(value), end='')

        recArr.append(value) ; # add 1 record at ending
        
        row = row+1 # total read counter
    # end while

    # classification data
    s = inFile.readline() # skip
    s = inFile.readline()
    data1 = s.strip() # remove leading and ending blanks
        
    data1 = data1.replace('[', '') # remove [
    data1 = data1.replace(']', '') # remove ]
    strs26 = data1.split() # array of 26 str

    for t in strs26:
        clsArr.append(int(t))
    # end for
    
    # close input file
    inFile.close()

    npXY = np.array(recArr)
    npC  = np.array(clsArr)
    
    return npXY, npC

#end #######################################################


x, y = readForge("./forge_dataset.txt")
print(x)
print(y)
