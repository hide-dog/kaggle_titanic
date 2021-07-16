# ------------------------------------------------
# import
# ------------------------------------------------
import re

# ------------------------------------------------
# read file
# ------------------------------------------------
def rewrite_train_file(inf, of):
    # read file
    with open(inf) as f:
        lines = f.readlines()
    #end

    s = len(lines)
    for i in range(s-1):

        # del name
        n = re.findall('".+"', lines[i+1])[0]
        stn = lines[i+1].find(n)
        enn = stn + len(n)
        lines[i+1] = lines[i+1][:stn] + lines[i+1][enn:]

        # split 
        l = lines[i+1].split(",")
        
        # refine data
        line = ""
        
        # Pclass
        if l[2] == "":
            l[2] = "3"
        else:
            pass
        #end

        # Name
        l[3] = "1"

        # Sex
        if l[4] == "male":
            l[4] = "1"
        else:
            l[4] = "0"
        #end

        # Age
        if l[5] == "":
            l[5] = "30"
        else:
            pass
        #end

        # SibSp
        if l[6] == "":
            l[6] = "1"
        else:
            pass
        #end

        # Parch
        if l[7] == "":
            l[7] = "0"
        else:
            pass
        #end

        # Ticket
        l[8] = "1"

        # Fare
        if l[9] == "":
            l[9] = "20.0"
        else:
            pass
        #end

        # Cabin
        l[10] = "1"

        # Embarked
        if l[11].find("S") > -1:
            l[11] = "0"
        elif l[11].find("C") > -1:
            l[11] = "1"
        elif l[11].find("Q") > -1:
            l[11] = "2"
        else:
            l[11] = "0"
        #end

        for j in range(len(l)): 
            if j == len(l) - 1:
                line = line + l[j] + "\n"
            else:
                line = line + l[j] + ","
            #end
        #end
        lines[i+1] = line
    #end

    # write file
    with open(of, "w") as f:
        for i in range(len(lines)):
            f.write(lines[i])
        #end
    #end
#end

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    # rewrite file
    rewrite_train_file("train.csv", "train.txt")
    rewrite_train_file("test.csv", "test.txt")
#end

if __name__ == "__main__":
    main()