# ------------------------------------------------
# import
# ------------------------------------------------
using Random
using Printf
#using Distributions

# ------------------------------------------------
# main
# ------------------------------------------------
function main()
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
    mean  = 0.0    # average
    stdev = 1.0    # variance value

    # -------------------------------------
    # nn loop
    # -------------------------------------
    main_ln  = 1000
    online_ln = 10


    # -------------------------------------
    # cal
    # -------------------------------------
    for i in 1:length(out_n_at_m)
        out_n_at_m[i] += 1
    end
    out_n = maximum(out_n_at_m)
    num_layers = num_mlayers + 2

    # ---------------------------------------------------------------------------------------
    # -------------------------------------
    # setup w
    # -------------------------------------
    w_old  = ones(num_layers-1, out_n, out_n)
    w_new  = ones(num_layers-1, out_n, out_n)
    w_temp = ones(num_layers-1, out_n, out_n)

    # dis of gauss    
    for i in 1:num_layers-1
        for j in 2:out_n
            for k in 1:out_n
                #rng = MersenneTwister(1234)
                w_old[i,j,k] = rand_normal(mean, stdev)
                w_new[i,j,k] = rand_normal(mean, stdev)
            end
        end
    end

    # -------------------------------------
    # setup for nn
    # -------------------------------------    
    u     = zeros(num_layers  , out_n)   # sum result
    z     = zeros(num_layers  , out_n)   # output
    temp  = zeros(out_n)                 # output
    delta = zeros(num_layers  , out_n)
    dEdw  = zeros(num_layers-1, out_n, out_n)

    roca_test  = zeros(main_ln)
    roca_train = zeros(main_ln)

    # -------------------------------------
    # read file
    # -------------------------------------
    train_re, train_data, train_ans = read_file(f_train)
    test_re,  test_data,  test_ans  = read_file(f_test)
    s = size(train_re)[1]
    
    # -------------------------------------
    # machine learning by NN
    # -------------------------------------
    for t in 1:main_ln
        for l in 1:online_ln
            n = rand(1:s)
            
            # store init input para
            z[1,1] = 0.0
            for j in 1:in_n-1
                z[1,j+1] = train_data[n,j]
            end
            
            # Forward propagation
            for i in 2:num_layers

                for k in 1:out_n_at_m[i]
                    u[i,k] = 1.0
                    for j in 1:out_n_at_m[i-1]
                        u[i,k] += w_new[i-1,j,k] * z[i-1,j]
                    end
                end
                """
                u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
                u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
                """
                
                # activation function
                if i != num_layers
                    # middle layers
                    #rectifier_func(z, u, i)
                    z = sigm_func(z, u, i)
                else
                    # output layer
                    z = softmax_func(z, u, i)
                end
                
                # set z[i,0]
                z[i,1] = 0.0
            end
                
            # Back propagation
            # delta at output layer
            i = num_layers - 1
            for k in 2:out_n_at_m[num_layers]
                delta[i,k] = z[i,k] - train_re[n,k-1]
            end

            # delta
            for ii in 1:num_layers-1
                i = num_layers - ii+1
                for k in 1:out_n_at_m[i]
                    #df = rectifier_func_derivation(u[i,k])
                    df = sigm_func_derivation(u[i,k])
                    for j in 1:out_n_at_m[i-1]
                        delta[i-1,k] += delta[i,j] *(w_new[i-1,j,k] * df )
                    end
                end
            end
            """
            delta[0] = 0.0
            delta[1] = z[1] - train_re[i,0]
            delta[2] = z[2] - train_re[i,1]
            """

            for ii in 1:num_layers-1
                i = num_layers - ii+1
                
                # slope
                for k in 1:out_n_at_m[i]
                    for j in 1:out_n_at_m[i-1]
                        dEdw[i-1,j,k] += delta[i,j] * z[i,k]
                    end
                end
                """
                dEdw[0,0] += delta[0] * 0.0
                dEdw[0,1] += delta[0] * train_data[i,0]
                dEdw[0,2] += delta[0] * train_data[i,1]
                dEdw[1,0] += delta[1] * 0.0
                dEdw[1,1] += delta[1] * train_data[i,0]
                dEdw[1,2] += delta[1] * train_data[i,1]
                """
            end
            
            # update w
            for i in 1:num_layers-1
                for j in 2:out_n
                    for k in 1:out_n
                        w_temp[i,j,k] = w_old[i,j,k] + mu*(w_new[i,j,k] - w_old[i,j,k]) - ep*dEdw[i,j,k] / online_ln
                    end
                end
            end
            #w_temp = w_old + mu*(w_new - w_old) - ep*dEdw / online_ln
            w_old  = copy(w_new)
            w_new  = copy(w_temp)

            # reset 
            for i in 1:num_layers-1
                for j in 1:out_n
                    for k in 1:out_n        
                        dEdw[i,j,k] = 0.0
                    end
                end
            end
            for i in 1:num_layers   
                for k in 1:out_n        
                    delta[i,k] = 0.0
                end
            end
        end
        
        # check rate of correct answer
        roca_test[t]  = test_nn( test_data,  test_ans, w_new, in_n, out_n, out_n_at_m, num_layers,  f_re_test)
        roca_train[t] = test_nn(train_data, train_ans, w_new, in_n, out_n, out_n_at_m, num_layers, f_re_train)
    end
    
    # output
    output_result(f_re_train, main_ln, roca_train)
    output_result(f_re_test, main_ln, roca_test)
end

# ------------------------------------
# read file
# ------------------------------------
function read_file(infile)
    skipnum = 1

    fff = []
    open(infile, "r") do f
        fff = read(f,String)
    end 
    fff = split(fff, "\n", keepempty=false)
    num_data = length(fff) - skipnum
    
    temp   = split(fff[2],",")
    data   = zeros(num_data, length(temp)-2)
    result = zeros(num_data, 2)
    ans    = zeros(num_data)

    for i in 1+skipnum:length(fff)
        ii = i - 1
        fff[i] = replace(fff[i]," \r" => "")
        
        temp = split(fff[i],",")
        
        result[ii, (parse(Int,temp[2]) + 1)] = 1
        for j in 1:length(temp)-2
            data[ii,j] = parse(Float64, temp[j+2])
        end

        ans[ii] = parse(Int, temp[2])
    end

    #println(size(data))
    #println(size(result))
    #println(size(ans))

    return result, data, ans
end

# ------------------------------------
# output file
# ------------------------------------
function output_result(outf, main_ln, roca)
    open(outf,"w") do f
        for t in 1:main_ln
            a = @sprintf("%8.8e", roca[t])
            write(f, a)
            write(f, "\n")
        end
    end
end

# ------------------------------------------------
# whitening
# ------------------------------------------------
function whitening(data)
    nl = length(data)
    nd = length(data[1])

    # ave = 0    
    for i in 1:nl
        ave = 0.0
        for j in 1:nd
            ave += data[i,j]
        end
        ave /= nd
        for j in 1:nd
            data[i,j] -= ave 
        end
    end
    
    # variance value = 0
    for i in 1:nl
        sigma = 0.0
        for j in 1:nd
            sigma += data[i,j]^2
        end
        sigma /= nd
        sigma = (sigma)^0.5
        
        if sigma == 0.0
            sigma = 1.0
        end
        for j in 1:nd
            data[i,j] /= sigma
        end
    end

    return data
end

# ------------------------------------------------
# random on gauss distribution
# ------------------------------------------------
function rand_normal(mean, stdev)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r  = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    rg = mean + stdev*r*sin(theta)
    return rg
end
# ------------------------------------------------
# logistic sigmoid function
# ------------------------------------------------
function sigm_func(z, u, i)
    ld = length(u[i])
    for j in 1:ld
        z[i,j] = 1.0 / ( 1.0 + exp(-u[i,j]) )
    end
    return z
end

# ------------------------------------------------
# Derivation of logistic sigmoid function
# ------------------------------------------------
function sigm_func_derivation(ui)
    y = 1 / ( 1 + exp(-ui) )
    y = (1-y)*y
    return y
end

# ------------------------------------------------
# hyperbolic tangent function
# ------------------------------------------------
function hyperbolic_tangent_func(z, u, i)
    ld = length(u[i])
    for j in 1:ld
        z[i,j] = ( exp(u[i,j]) - exp(-u[i,j]) ) / ( exp(u[i,j]) + exp(-u[i,j]) )
    end
    return z
end

# ------------------------------------------------
# Rectified linear function
# ------------------------------------------------
function rectifier_func(z, u, i)
    ld = length(u[i])
    for j in 1:ld
        z[i,j] = max( u[i,j], 0.0 )
    end
    return z
end

# ------------------------------------------------
# Derivation of Rectified linear function
# ------------------------------------------------
function rectifier_func_derivation(ui)
    fd = 1.0

    if ui <= 0.0
        fd = 0.0
    end
    return fd
end

# ------------------------------------------------
# linear mapping ( function )
# ------------------------------------------------
function linear_func(z, u, i)
    ld = length(u[i])
    for j in 1:ld
        z[i,j] = u[i,j]
    end
    return z
end

# ------------------------------------------------
# softmax function 
# ------------------------------------------------
function softmax_func(z, u, i)
    ld = length(u[i])

    s = 0.0
    for j in 1:ld
        s += exp(u[i,j])
    end
    for j in 1:ld
        z[i,j] = exp(u[i,j]) / s
    end
    return z
end

# ------------------------------------------------
# test 
# ------------------------------------------------
function test_nn(test_data, corr_data, w, in_n, out_n, out_n_at_m, num_layers, of)
    size = Int(length(test_data) / in_n)
    ans  = zeros(size)
    u    = zeros(num_layers, out_n) # sum result
    z    = zeros(num_layers, out_n) # output

    point = 0
    for n in 1:size
        # store init input para
        z[1,1] = 0.0
        for j in 1:in_n
            z[1,j+1] = test_data[n,j]
        end

        # Forward propagation
        for i in 2:num_layers

            for k in 1:out_n_at_m[i]
                u[i,k] = 1.0
                for j in 1:out_n_at_m[i-1]
                    u[i,k] += w[i-1,j,k] * z[i-1,j]
                end
            end
            """
            u[0] = 1.0 + train_data[i][0] * w_new[0,1] + train_data[i][1] * w_new[0,2] 
            u[1] = 1.0 + train_data[i][0] * w_new[1,1] + train_data[i][1] * w_new[1,2]
            """
            
            # activation function
            if i != num_layers
                # middle layers
                z = rectifier_func(z, u, i)
            else
                # output layer
                z = softmax_func(z, u, i)
            end
            
            # set z[i,0]
            z[i,1] = 0.0
        end
            
        # if corr or not
        max_index = argmax(z[num_layers-1])
        ans[n] = max_index - 1

        if ans[n] == corr_data[n]
            point += 1
        end
        """
        print(z[num_layers-1])
        print(ans)
        raise ValueError("error!")
        """
    end
    
    # rate of correct answer    
    roca = point / size
    
    
    """
    print("------------------------------")
    print(ans[:50])
    print(corr_data[:50])
    print(point)
    print(size)
    print(roca)
    """

    return roca
end

# ---------------------------------------------
# ---------------------------------------------
main()