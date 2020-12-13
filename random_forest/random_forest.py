# ------------------------------------------------
# Macine Learnig     : random forest
# Target Competition : titanic
# https://www.kaggle.com/c/titanic/overview
# ------------------------------------------------

import pandas as pd
import numpy as np

# random forest
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------
# calculate the number of deficits for each data
# ------------------------------------------------
def check_deficit(f):
    # counting the number of nulls
    number_null  = f.isnull().sum()

    # calculate rate of deficit
    percent      = number_null/len(f) * 100.0

    # combine 
    deficit      = pd.concat([number_null, percent], axis=1)

    # rename
    data_deficit = deficit.rename( columns = {0 : 'deficits', 1 : '%'})
    return data_deficit

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    
    # read file 
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")

    # confirm size of array
    test_shape  = test.shape
    train_shape = train.shape
    print("\n size of test data")
    print(test_shape)
    print("\n size of train data")
    print(train_shape)

    # deficits
    print("\n deficits of test data")
    print(check_deficit(test))
    print("\n deficits of train data")
    print(check_deficit(train))
    
    # filling in blanks    
    train["Age"]      = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna("S")
    
    test["Age"]  = test["Age"].fillna(test["Age"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())
    
    # Convert strings to corresponding numbers
    train["Sex"]      = train["Sex"].map({"male" : 0, "female" : 1})
    train["Embarked"] = train["Embarked"].map({"S" : 0, "C" : 1, "Q" : 2})

    test["Sex"]      = test["Sex"].map({"male" : 0, "female" : 1})
    test["Embarked"] = test["Embarked"].map({"S" : 0, "C" : 1, "Q" : 2})

    # get the objective and explanatory variables for "train"
    objective   = train["Survived"].values
    explanatory = train[["Pclass", "Sex", "Age", "Fare"]].values

    # get the explanatory variables for "test"
    explanatory_test = test[["Pclass", "Sex", "Age", "Fare"]].values

    # machine learning by random forest
    clf = RandomForestClassifier(n_estimators=100, random_state=100)
    clf.fit(explanatory, objective)
    
    # prediction
    prediction = clf.predict(explanatory_test)

    # get PassengerId
    pid        = np.array(test["PassengerId"]).astype(int)
    # combination of PassengerId and prediction
    solution   = pd.DataFrame(prediction, pid, columns = ["Survived"])
    # output .csv
    solution.to_csv("solution_random_forest.csv", index_label = ["PassengerId"])

if __name__ == "__main__":
    main()