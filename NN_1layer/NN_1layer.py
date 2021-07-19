# ------------------------------------------------
# import
# ------------------------------------------------
import numpy as np
import math
import copy
import random
import tqdm

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    # -------------------------------------
    # file
    # -------------------------------------
    f_train = "train.txt"
    f_test  = "test.txt"

    f_re_train = "corr_history_train"
    f_re_test  = "corr_history_test"
    
    # -------------------------------------
    # input & output
    # -------------------------------------
    in_n    = 10
    out_n   = 2 

    # -------------------------------------
    # nn para
    # -------------------------------------
    num_layers = 0
    mu = 0.9
    ep = 0.5
    sigma = 3.0

    # -------------------------------------
    # nn loop
    # -------------------------------------
    main_ln  = 300
    batch_ln = 10
    

    # ---------------------------------------------------------------------------------------
    # -------------------------------------
    # setup w
    # -------------------------------------
    w_old  = np.ones((out_n, in_n+1)) # bを無くす ,常に1を出力する層を追加
    w_new  = np.ones((out_n, in_n+1)) # bを無くす, 常に1を出力する層を追加
    w_temp = np.ones((out_n, in_n+1)) # bを無くす, 常に1を出力する層を追加

    # dis of gauss    
    for i in range(1, out_n):
        for j in range(in_n+1):
            w_old[i,j] = random.gauss(0.5, sigma)
            w_new[i,j] = random.gauss(0.5, sigma)
        #end
    #end

    # -------------------------------------
    # setup for nn
    # -------------------------------------    
    u     = np.zeros(out_n)
    delta = np.zeros(out_n)
    dEdw  = np.zeros((out_n, in_n+1))

    roca_test  = np.zeros(main_ln)
    roca_train = np.zeros(main_ln)

    # -------------------------------------
    # reset file
    # -------------------------------------
    with open(f_re_train, "w") as f:
        pass
    #end
    with open(f_re_test, "w") as f:
        pass
    #end


    # -------------------------------------
    # read file
    # -------------------------------------
    train_re, train_data, train_ans = read_train_file(f_train)
    test_re,  test_data,  test_ans  = read_train_file(f_test)
    s = len(train_re)
    
    # -------------------------------------
    # machine learning by NN
    # -------------------------------------
    for t in tqdm.tqdm(range(main_ln)):
        dEdw  = np.zeros((out_n, in_n+1))
        for l in range(batch_ln):
            i = random.randint(0, s-1)

            # Forward propagation
            for j in range(out_n):
                u[j] = 1.0
                for k in range(in_n):
                    u[j] += train_data[k,i] * w_new[j,k+1]
                #end
            #end
            """
            u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
            u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
            """
            
            # Back propagation
            # delta at output layer
            y = softmax_func(u)
            for j in range(out_n):
                delta[j] = y[j] - train_re[i,j]
            #end
            """
            delta[0] = y[0] - train_re[i,0]
            delta[1] = y[1] - train_re[i,1]
            """

            # slope
            for j in range(out_n):
                for k in range(1, in_n+1):
                    dEdw[j,k] += delta[j] * train_data[k-1,i]
                #end
            #end

            """
            dEdw[0,0] += delta[0] * 0.0
            dEdw[0,1] += delta[0] * train_data[i,0]
            dEdw[0,2] += delta[0] * train_data[i,1]
            dEdw[1,0] += delta[1] * 0.0
            dEdw[1,1] += delta[1] * train_data[i,0]
            dEdw[1,2] += delta[1] * train_data[i,1]
            """
        #end

        # update w
        w_temp = w_old + mu*(w_new - w_old) - ep*dEdw / batch_ln
        w_old  = copy.deepcopy(w_new)
        w_new  = copy.deepcopy(w_temp)

        # check rate of correct answer        
        roca_train[t] = test_nn(train_data, train_ans, w_new, in_n, out_n, f_re_train)
        roca_test[t]  = test_nn(test_data, test_ans, w_new, in_n, out_n, f_re_test)
    #end

    # output
    with open(f_re_train, "a") as f:
        for t in range(main_ln):
            f.write(str(roca_train[t]) + "\n")
        #end
    #end
    with open(f_re_test, "a") as f:
        for t in range(main_ln):
            f.write(str(roca_test[t]) + "\n")
        #end
    #end
#end


# ------------------------------------------------
# read file
# ------------------------------------------------
def read_train_file(inf):
    # read file
    PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked \
    = np.loadtxt(inf, delimiter = ",",unpack = True, skiprows = 1)

    s = len(PassengerId)
    result = np.zeros((s,2))
    
    train_ans = copy.deepcopy(Survived)
    data      = np.array([Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked])
    
    for i in range(s):
        # save ans
        result[i,int(train_ans[i])] = 1
    #end

    for i in range(len(data)):
        m = max(data[i])
        for j in range(s):
            data[i,j] = data[i,j] / m
        #end
    #edn

    return  result, data, train_ans 
#end

# ------------------------------------------------
# logistic sigmoid function
# ------------------------------------------------
def sigm_func(u):
    size = len(u)
    y = np.zeros(size)

    for i in range(size):
        y[i] = 1 / ( 1 + math.exp(-u[i]) )
    #end
    return y
#end

# ------------------------------------------------
# hyperbolic tangent function
# ------------------------------------------------
def hyperbolic_tangent_func(u):
    size = len(u)
    y = np.zeros(size)

    for i in range(size):
        y[i] = ( math.exp(u[i]) - math.exp(-u[i]) ) / ( math.exp(u[i]) + math.exp(-u[i]) )
    #end
    return y
#end

# ------------------------------------------------
# Rectified linear function
# ------------------------------------------------
def rectifier_func(u):
    size = len(u)
    y = np.zeros(size)

    for i in range(size):
        y[i] = max( u[i], 0 )
    #end
    return y
#end

# ------------------------------------------------
# linear mapping ( function )
# ------------------------------------------------
def linear_func(u):
    """
    size = len(u)
    y = np.zeros(size)

    for i in range(size):
        y[i] = u[i]
    #end
    """
    return u
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
def test_nn(test_data, corr_data, w, in_n, out_n, of):
    size = len(test_data[0])
    ans = np.zeros(size)
    u = np.zeros(out_n)
    
    point = 0
    for i in range(size):
        for j in range(out_n):
            u[j] = 1.0
            for k in range(in_n):
                u[j] += test_data[k,i] * w[j,k+1]
            #end
        #end
        
        y = softmax_func(u)

        max_index = np.argmax(y)
        ans[i] = max_index

        if ans[i] == corr_data[i]:
            point += 1
        else:
            pass
        #end
    #end

    # rate of correct answer    
    roca = float(point) / float(size)
    #end

    return roca
#end

# ---------------------------------------------
# ---------------------------------------------
if __name__ == "__main__":
    main()