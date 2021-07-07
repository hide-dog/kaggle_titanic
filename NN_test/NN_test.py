# ------------------------------------------------
# import
# ------------------------------------------------
import numpy as np
import math

# ------------------------------------------------
# read file
# ------------------------------------------------
def read_train_file(inf):
    # read file
    with open(inf) as f:
        lines = f.readlines()
    #end
    s = len(lines)
    result = np.zeros(s-1)
    data = np.zeros((s-1,2))
    
    for i in range(s-1):
        l = lines[i+1].split(",")

        if l[1] == "":
            result[i] = 0
        else:
            result[i] = int(l[1])
        #end

        if l[2] == "":
            data[i][0] = 3
        else:
            data[i][0] = int(l[2])
        #end

        if l[6] == "":
            data[i][1] = 30
        else:
            data[i][1] = math.floor(float(l[6]))
        #end
    return  result, data   
#end

# ------------------------------------------------
# read file
# ------------------------------------------------
def read_test_file(inf):
    # read file
    with open(inf) as f:
        lines = f.readlines()
    #end
    s = len(lines)
    data = np.zeros((s-1,2))
    
    for i in range(s-1):
        l = lines[i+1].split(",")

        if l[1] == "":
            data[i][0] = 3
        else:
            data[i][0] = int(l[1])
        #end

        if l[5] == "":
            data[i][1] = 30
        else:
            data[i][1] = math.floor(float(l[5]))
        #end
    return  data
#end

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    f_train = "train.csv"
    f_test = "test.csv"
    
    # read file
    train_re, train_data = read_train_file(f_train)
    test_data = read_test_file(f_test)

    # machine learning by NN

    # get PassengerId
#end
if __name__ == "__main__":
    main()