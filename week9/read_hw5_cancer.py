import numpy as np

############################################################
def readHw5Cancer(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    s = inFile.readline() # skip
    
    row = 0
    while True:
        s = inFile.readline()
        data1 = s.strip() # remove leading and ending blanks
        if (len(data1) <= 0):
            break
        
        # since we use append, value must be created in the loop
        value = []
        
        strs31 = data1.split(',') # array of 31 str

        # convert to real
        for ix in range(30):
            value.append( eval(strs31[ix]) )
        # end for
        
        target = eval(strs31[30])

        recArr.append(value) ;  # add 1 record at end of array
        clsArr.append(target) ; # add 1 record at end of array
       
        row = row+1 # total read counter
    # end while
    
    # close input file
    inFile.close()

    # convert list to Numpy array
    npXY = np.array(recArr)
    npC  = np.array(clsArr)

    # pass out as Numpy array
    return npXY, npC

# end function