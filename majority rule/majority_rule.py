import pandas as pd
import numpy as np
import glob
import csv

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    ofile = "majority_rule.csv"
    
    fff = glob.glob("solution*")

    y = np.loadtxt(fff[0], delimiter=',', skiprows = 1, usecols = (1), unpack=True)
    y_majority = np.zeros(len(y), dtype=np.int16)

    for i in range(len(fff)):
        x, y = np.loadtxt(fff[i], delimiter=',', skiprows = 1, usecols = (0, 1), unpack=True)

        for j in range(len(y)):
            y_majority[j] += y[j]
    
    
    for i in range(len(y_majority)):
        if len(fff)/2 <= y_majority[i]:
            y_majority[i] = 1
        else:
            y_majority[i] = 0

    # read file 
    test  = pd.read_csv("test.csv")

    # get PassengerId
    pid        = np.array(test["PassengerId"]).astype(int)
    # combination of PassengerId and prediction
    solution   = pd.DataFrame(y_majority, pid, columns = ["Survived"])
    # output .csv
    solution.to_csv(ofile, index_label = ["PassengerId"])
    
    # 
    correct = pd.read_csv("correct.csv")
    correct_s = correct["Survived"].values
    
    score = 0.0
    for i in range(len(y_majority)):
        if correct_s[i] - y_majority[i] == 0.0:
            score += 1.0
    
    print( score / len(y_majority) )
    
    
    # output
    """
    with open(ofile, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Survived"])
        for i in range(len(y_majority)):
            writer.writerow([x[i], y_majority[i]])
    """
    


# ------------------------------------------------
# execution
# ------------------------------------------------
if __name__ == "__main__":
    main()