# ------------------------------------------------
# import
# ------------------------------------------------
import numpy as np
import math
import copy
import random
import tqdm
import sys

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
    in_n       = 10           # +1 for b
    fout_n     = 2            # final output
    out_n_at_m = [in_n, 6, fout_n]          # number of cell at middle layers

    # -------------------------------------
    # nn para
    # -------------------------------------
    num_mlayers = 1   # middle layer
    mu = 0.9
    ep = 0.5

    # -------------------------------------
    # init weight
    # -------------------------------------
    mu_ave = 0.0    # average
    sigma  = 1.0    # variance value

    # -------------------------------------
    # nn loop
    # -------------------------------------
    main_ln  = 1000
    online_ln = 10


    # -------------------------------------
    # cal
    # -------------------------------------
    out_n_at_m = np.array(out_n_at_m) + 1
    out_n = max(out_n_at_m)
    num_layers = num_mlayers + 2

    # ---------------------------------------------------------------------------------------
    # -------------------------------------
    # setup w
    # -------------------------------------
    w_old  = np.ones((num_layers-1, out_n, out_n))
    w_new  = np.ones((num_layers-1, out_n, out_n))
    w_temp = np.ones((num_layers-1, out_n, out_n))

    # dis of gauss    
    for i in range(num_layers-1):
        for j in range(1, out_n):
            for k in range(out_n):
                w_old[i,j,k] = random.gauss(mu_ave, sigma)
                w_new[i,j,k] = random.gauss(mu_ave, sigma)
            #end
        #end
    #end

    # -------------------------------------
    # setup for nn
    # -------------------------------------    
    u     = np.zeros((num_layers  , out_n)) # sum result
    z     = np.zeros((num_layers  , out_n)) # output
    delta = np.zeros((num_layers  , out_n))
    dEdw  = np.zeros((num_layers-1, out_n, out_n))

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
    # whitening
    # -------------------------------------
    train_data = whitening(train_data)
    test_data  = whitening(test_data)

    # -------------------------------------
    # machine learning by NN
    # -------------------------------------
    for t in tqdm.tqdm(range(main_ln)):
        for l in range(online_ln):
            n = random.randint(0, s-1)
            
            # store init input para
            z[0,0] = 0.0
            for j in range(in_n):
                z[0,j+1] = train_data[n,j]
            #end
            
            # Forward propagation
            for i in range(1, num_layers):

                for k in range(out_n_at_m[i]):
                    u[i,k] = 1.0
                    for j in range(out_n_at_m[i-1]):
                        u[i,k] += w_new[i-1,j,k] * z[i-1,j]
                    #end
                #end
                """
                u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
                u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
                """
                
                # activation function
                if i != num_layers:
                    # middle layers
                    #z[i] = rectifier_func(u[i])
                    z[i] = sigm_func(u[i])
                else:
                    # output layer
                    z[i] = softmax_func(u[i])
                #end
                
                # set z[i,0]
                z[i,0] = 0.0
            #end
                
            # Back propagation
            # delta at output layer
            i = num_layers - 1
            for k in range(1, out_n_at_m[i]):
                delta[i,k] = z[i,k] - train_re[n,k-1]
            #end

            # delta
            for ii in range(num_layers-1):
                i = num_layers - (ii+2)
                for k in range(1, out_n_at_m[i]):
                    #df = rectifier_func_derivation(u[i,k])
                    df = sigm_func_derivation(u[i,k])
                    for j in range(out_n_at_m[i-1]):
                        delta[i,k] += delta[i+1,j] *(w_new[i,j,k] * df )
                    #end
                #end
            #end
            """
            delta[0] = 0.0
            delta[1] = z[1] - train_re[i,0]
            delta[2] = z[2] - train_re[i,1]
            """

            for ii in range(num_layers-1):
                i = num_layers - (ii+2)
                
                # slope
                for k in range(out_n_at_m[i]):
                    for j in range(out_n_at_m[i-1]):
                        dEdw[i,j,k] += delta[i+1,j] * z[i,k]
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
            w_temp = w_old + mu*(w_new - w_old) - ep*dEdw / online_ln
            w_old  = copy.deepcopy(w_new)
            w_new  = copy.deepcopy(w_temp)

            # reset 
            for i in range(num_layers-1):
                for j in range(out_n):
                    for k in range(out_n):        
                        dEdw[i,j,k] = 0.0
                    #end
                #end
            #end
            for i in range(num_layers):    
                for k in range(out_n):        
                    delta[i,k] = 0.0
                #end
            #end
        #end
        
        # check rate of correct answer        
        roca_test[t]  = test_nn( test_data,  test_ans, w_new, in_n, out_n, out_n_at_m, num_layers,  f_re_test)
        roca_train[t] = test_nn(train_data, train_ans, w_new, in_n, out_n, out_n_at_m, num_layers, f_re_train)
    #end
    print(w_new)

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
    #end
    
    # transposed matrix
    data = data.T 

    return  result, data, train_ans 
#end

# ------------------------------------------------
# whitening
# ------------------------------------------------
def whitening(data):
    nl = len(data)
    nd = len(data[0])

    # ave = 0    
    for i in range(nl):
        ave = 0.0
        for j in range(nd):
            ave += data[i,j]
        #end
        ave /= nd
        for j in range(nd):
            data[i,j] -= ave 
        #end
    #end
    
    # variance value = 0
    for i in range(nl):
        sigma = 0.0
        for j in range(nd):
            sigma += data[i,j]**2
        #end
        sigma /= nd
        sigma = (sigma)**0.5
        
        if sigma == 0.0:
            sigma = 1.0
        #end
        for j in range(nd):
            data[i,j] /= sigma
        #end
    #end

    return data
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
# Derivation of logistic sigmoid function
# ------------------------------------------------
def sigm_func_derivation(ui):
    y = 1 / ( 1 + math.exp(-ui) )
    y = (1-y)*y
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
# Derivation of Rectified linear function
# ------------------------------------------------
def rectifier_func_derivation(ui):
    if ui <= 0.0:
        fd = 0.0
    else:
        fd = 1.0
    #end
    return fd
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
def test_nn(test_data, corr_data, w, in_n, out_n, out_n_at_m, num_layers, of):
    size = len(test_data)
    ans = np.zeros(size)
    u     = np.zeros((num_layers, out_n)) # sum result
    z     = np.zeros((num_layers, out_n)) # output


    point = 0
    for n in range(size):
        # store init input para
        z[0,0] = 0.0
        for j in range(in_n):
            z[0,j+1] = test_data[n,j]
        #end
        
        # Forward propagation
        for i in range(1, num_layers):

            for k in range(out_n_at_m[i]):
                u[i,k] = 1.0
                for j in range(out_n_at_m[i-1]):
                    u[i,k] += w[i-1,j,k] * z[i-1,j]
                #end
            #end
            """
            u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
            u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
            """
            
            # activation function
            if i != num_layers:
                # middle layers
                z[i] = rectifier_func(u[i])
            else:
                # output layer
                z[i] = softmax_func(u[i])
            #end
            
            # set z[i,0]
            z[i,0] = 0.0
        #end
            
        # if corr or not
        max_index = np.argmax(z[num_layers-1])
        ans[n] = max_index - 1

        if ans[n] == corr_data[n]:
            point += 1
        else:
            pass
        #end
        """
        print(z[num_layers-1])
        print(ans)
        raise ValueError("error!")
        """
    #end
    
    # rate of correct answer    
    roca = point / size
    #end
    
    """
    print("------------------------------")
    print(ans[:50])
    print(corr_data[:50])
    print(point)
    print(size)
    print(roca)
    """

    return roca
#end

# ---------------------------------------------
# ---------------------------------------------
if __name__ == "__main__":
    main()