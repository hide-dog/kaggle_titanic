# ------------------------------------------------
# import
# ------------------------------------------------
import numpy as np
import math
import copy
import random

# ------------------------------------------------
# read file
# ------------------------------------------------
def read_train_file(inf):
    # read file
    with open(inf) as f:
        lines = f.readlines()
    #end
    s = len(lines)
    result = np.zeros((s-1,2))
    data = np.zeros((s-1,2))
    
    for i in range(s-1):
        l = lines[i+1].split(",")

        if l[1] == "":
            result[i,0] = 0
            result[i,1] = 1
        else:
            result[i,int(l[1])] = 1
        #end

        if l[2] == "":
            data[i][0] = 1
        else:
            data[i][0] = int(l[2]) / 3.0
        #end

        if l[6] == "":
            data[i][1] = 1
        else:
            data[i][1] = math.floor(float(l[6])) / 100
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
            data[i][0] = 1.0
        else:
            data[i][0] = int(l[1]) / 3.0
        #end

        if l[5] == "":
            data[i][1] = 1.0
        else:
            data[i][1] = math.floor(float(l[5])) / 100
        #end
    return  data
#end

# ------------------------------------------------
# read file
# ------------------------------------------------
def read_corr_file(inf):
    # read file
    with open(inf) as f:
        lines = f.readlines()
    #end
    s = len(lines)
    data = np.zeros(s-1)
    
    for i in range(s-1):
        l = lines[i+1].split(",")

        if l[1] == "":
            data[i] = 1
        else:
            #print(l[1].replace("\n",""))
            data[i] = int(l[1].replace("\n",""))
        #end
    return data
#end

# ------------------------------------------------
# logistic sigmoid function
# ------------------------------------------------
def sigm_func(u):
    f = 1 / ( 1 + math.exp(-u) )
    return f
#end

# ------------------------------------------------
# softmax function 
# ------------------------------------------------
def softmax_func(u):
    size = len(u)
    y = np.zeros(size)

    s = 0
    for i in range(size):
        s += math.exp(u[i])
    #end
    for i in range(size):
        y[i] = math.exp(u[i]) / s
    #end
    return y
#end

# ------------------------------------------------
# test 
# ------------------------------------------------
def test_nn(test_data, corr_data, w):
    size = len(test_data)
    ans = np.zeros(size)
    u = np.zeros(2)
    
    point = 0
    for i in range(size):
        u[0] = 1.0 * w[0,0] + test_data[i][0] * w[0,1] + test_data[i][1] * w[0,2] 
        u[1] = 1.0 * w[1,0] + test_data[i][0] * w[1,1] + test_data[i][1] * w[1,2]

        y = softmax_func(u)

        if y[0] >= y[1]:
            ans[i] = 0
        else:
            ans[i] = 1
        #end

        if ans[i] == corr_data[i]:
            point += 1
        else:
            pass
        #end
    #end
    print(" point : " + str(point/size))
#end


# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    f_train = "train.csv"
    f_test = "test.csv"
    f_corr = "correct.csv"

    in_para    = 2
    num_layers = 1

    mu = 0.9
    ep = 0.5
    sigma = 5.0

    # setup
    w_old = np.ones((2, in_para+1)) # bを無くす常に1を出力する層を追加
    w_new = np.ones((2, in_para+1)) # bを無くす常に1を出力する層を追加
    w_temp = np.ones((2, in_para+1)) # bを無くす常に1を出力する層を追加
    
    for i in range(1, 2):
        for j in range(in_para+1):
            w_old[i,j] = random.gauss(0.5, sigma)
            w_new[i,j] = random.gauss(0.5, sigma)
        #end
    #end
    
    u = np.zeros(2)
    delta = np.zeros(2)
    dEdw  = np.zeros((2, in_para+1))

    # read file
    train_re, train_data = read_train_file(f_train)
    test_data = read_test_file(f_test)
    corr_data = read_corr_file(f_corr)
    s = len(train_re)

    # machine learning by NN
    for k in range(100):
        dEdw  = np.zeros((2, in_para+1))
        for l in range(10):
            i = random.randint(0,s-1)

            # Forward propagation
            u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
            u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
            
            # Back propagation
            # delta at output layer
            y = softmax_func(u)
            delta[0] = y[0] - train_re[i,0]
            delta[1] = y[1] - train_re[i,1]

            # slope
            dEdw[0,0] += delta[0] * 0.0
            dEdw[0,1] += delta[0] * train_data[i,0]
            dEdw[0,2] += delta[0] * train_data[i,1]
            dEdw[1,0] += delta[1] * 0.0
            dEdw[1,1] += delta[1] * train_data[i,0]
            dEdw[1,2] += delta[1] * train_data[i,1]
        #end

        # update w
        w_temp = w_old + mu*(w_new - w_old) - ep*dEdw / 10
        w_old = copy.deepcopy(w_new)
        w_new = copy.deepcopy(w_temp)

        test_nn(test_data, corr_data, w_new)
    #end
    print(w_new)
#end

if __name__ == "__main__":
    main()