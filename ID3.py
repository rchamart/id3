import numpy as np
import pandas as pd
import sys
import math

def Entropy(arr):
    """
            *  DESCRIPTION:
                * this dataset calculates the entropy of the value using the basic formula:
                    * -frac(att_pos) * log(frac(att_pos)) - frac(att_neg) * log(frac(att_neg))

            *  INPUT PARAMETERS:
                * array containing: count(att_pos), count(att_neg)

            *  OUTPUT PARAMETERS:
                * return calculated entropy
    """
    ret_ent = 0

    arr = np.asarray(arr)
    if np.sum(arr) == 0:
        return 0

    # CALCULATE E(S): ENTROPY OF WHOLE SET
    total_sum = arr[0] + arr[1]

    # CALCULATE POSITIVE
    pos_frac = arr[1]/total_sum
    if pos_frac == 0:
        return 0
    log_of_pos = math.log2(pos_frac)
    total_pos_comp = ((-1*pos_frac) * log_of_pos)

    # CALCULATE NEGATIVE SIDE
    neg_frac = arr[0] / total_sum
    if neg_frac == 0:
        return 0
    log_of_neg = math.log2(neg_frac)
    total_neg_comp = ((-1 * neg_frac) * log_of_neg)

    # SUM TO GET THE E(S)
    total_pos_comp+total_neg_comp
    return total_pos_comp + total_neg_comp


def InfoGain(data, e_s):
    """
                *  DESCRIPTION:
                    * this dataset calculates the information gain according the formula in the handout

                *  INPUT PARAMETERS:
                    * array containing: DataFrame for the attribute we wish to calculate the information gain for

                *  OUTPUT PARAMETERS:
                    * return calculated information gain for the attribute we wish to calculate for
    """

    # GET THE TOTAL NUMBER OF NEGS(F) AND POS(T) FOR ONLY THE CLASS LABEL (F, T)
    totalValues = numPosNumNeg(data, 0)

    # GET THE TOTAL NUMBER OF NEGS(F) AND POS(T) WITH RESPECT TO FEATURE VALUES (0, 1)
    valuesSplit = numPosNumNeg(data, 1)

    # CALCULATE THE ENTROPY OF THE TWO COMPOSITE FEATURES
    values_feature_false = [valuesSplit[0], valuesSplit[1]]
    e_feature_false = Entropy(values_feature_false)

    values_feature_true = [valuesSplit[2], valuesSplit[3]]
    e_feature_true = Entropy(values_feature_true)

    # CALCULATE AVERAGE INFORMATION GAIN
    frac_feature_false = ((valuesSplit[0]+valuesSplit[1])/(totalValues[0]+totalValues[1]))
    info_avg_feature_false = e_feature_false * frac_feature_false

    frac_feature_true = ((valuesSplit[2]+valuesSplit[3])/(totalValues[0]+totalValues[1]))
    info_avg_feature_true = frac_feature_true * e_feature_true

    # CALCULATE TOTAL INFO GAIN GIVEN THIS FEATURE
    total_gain = e_s - (info_avg_feature_false+info_avg_feature_true)

    return total_gain


def PreProcess(dataset, percent, type):
    """
        *  DESCRIPTION:
            * this dataset simply expands all of the features in the dataset to binary attributes, so ya, lets eshkitit

        *  INPUT PARAMETERS:
            * dataset: this is just the unaltered dataset
            * percent: percent of the dataset to sample and return to you

        *  OUTPUT PARAMETERS:
            * expand_df: this is a dataframe representing expanded dataset. expanded following the rules described in
              the description
    """

    # IMPORT THE DATASET AND CONVERT TO A DATAFRAME
    labels = ["Employment", "EducationLevel", "MarriageStatus", "JobType", "FamilyRole", "Race", "Gender",
              "Nationality", "salaryLevel"]
    expand_df = pd.read_csv(dataset, sep=",", names=labels)

    if type == "train":
        expand_df = expand_df.sample(frac=percent)

    # EXPAND THE ATTRIBUTES BB
    for col in expand_df:
        # GET THE SERIES ACCORDING TO THIS COL
        iter = pd.Series(expand_df[col])
        iter = iter.fillna("None")

        #print(col)

        # SKIP IF SALARY LEVEL BECAUSE WE JUST HAVE TO RUN ON A SINGLE ITERATION
        if col == "salaryLevel":
            # IMPORT THE SET OF VALUES WHICH THIS CAN TAKE
            # USED: >50K
            setValues = set(expand_df[col].values)
            attribute = setValues.pop()

            # CREATE A NEW COLUMN
            mod_tcol_name = "salaryLevel_B"

            # POPULATE NEW BINARY COLUMN
            modCol = iter.str.contains(attribute, regex="False")
            modCol.fillna(False)

            # ADD THE NEW ISH
            expand_df[mod_tcol_name] = modCol.values

            # DROP THE OLD BISH
            expand_df = expand_df.drop(columns=['salaryLevel'])

            continue

        # HANDLE ALL ATTRIBUTES EXCEPT FOR OUR CLASS LABEL
        for attribute in set(expand_df[col].values):
            # CREATE NEW COLUMN
            mod_tcol_name = attribute + "_" + col + "B"
            #print(mod_tcol_name)

            # POPULATE NEW COLUMN
            modCol = iter.str.contains(attribute, regex="False")
            modCol = modCol.fillna(False)

            # ADD THIS NEW COLUMN IN ITS PLACE
            expand_df[mod_tcol_name] = modCol.values

        # DROP THE OLD COLUMN FROM DATAFRAME
        expand_df = expand_df.drop(columns=[col], axis=1)

    return expand_df


class Node(object):
    def __iter__(self):
        self.left = None  # the left child of the node
        self.right = None  # the right child of the node
        self.attribute = None # the feature being represented by this path
        self.decision = None # the spontaneous decision if we stopped at this point
        self.type = None # the type that this node is, terminal, root, child, etc.
        self.dataset = None # the wittled dataset associated with this


class Tree(object):
    """
        *CONSTRUCTOR FOR MY TREE:
            * vanilla:
                * THIS IS JUST A vanilla TREE SO WE WILL CALL THE VANILLA TREE BUILDING METHOD
            * maxDepth:
                * THIS IS THE maxDepth CALL, WE WILL CALL OUR METHOD THAT BUILDS THE TREE WITH RESPECT TO MAX DEPTH
    """
    def __init__(self, dataset, tree_type, depth_val):
        if tree_type == "vanilla":
            # SET THE ROOT NODE ACCORDING TO VANILLA RULES
            self.root = Vanilla(dataset)
        elif tree_type == "maxDepth":
            # SET THE ROOT NODE ACCORDING TO MAX DEPTH RULES
            self.root = MaxDepth(dataset, 1, depth_val)


def doWeStop(data):
    # print("CALLED")
    """
        * DESCRIPTION:
            * calculate whether or not we stop at this point
            * HOW:
                * check if all the corresponding values in salaryLevel are the same

        * INPUT PARAMETERS  :
            * data: just the dataset associated with the iteration of the run

        * OUTPUT PARAMETERS :
            * retVal: an array reflecting whether the HOW of this statement holds
    """
    # EXTRACT THE SALARY LEVEL COLUMN TO POLL
    check = data[['salaryLevel_B']]

    # INSTEAD OF LOOPING JUST GET A SET OF ALL COMPOSITE VALUES
    unique_Vals = set(check['salaryLevel_B'].values)

    # IF THE SET ONLY CONTAINS ONE VALUE THIS IS A STOPPING POINT
    if len(unique_Vals) == 1:
        ret_val = [True, unique_Vals.pop()]
        return ret_val

    # THE SET STILL CONTAINS MULTIPLE VALUES SO CONTINUE FORWARD
    ret_val = [False, unique_Vals]
    return ret_val


def numPosNumNeg(data, flag):
    """
         * DESCRIPTION:
            * calculate the number of positives and negatives given the dataset

        * INPUT PARAMETERS  :
            * data: just the dataset associated with the iteration of the run
            * flag: the mode to run in
                * 0: just consider the end values
                * 1: take a look at the whole dataset columns wise and return whatever the bleep you need

        * OUTPUT PARAMETERS :
            * retVal: an array reflecting the number of positives and number of negatives given the flag condition
                * format: [numberOfTrue, numberOfFalse]
    """
    numPos = 0
    numNeg = 0
    ret_val = []

    if flag == 0:
        # JUST LOOK AT CLASS LABEL
        # CONVERT TO NUMPY TO AVOID ITERATION
        check = data[['salaryLevel_B']]
        numpy_check = check.to_numpy()

        # TAKE THE SUM TO REFLECT NUMBER OF TRUE
        pos = np.sum(numpy_check)

        # TAKE THE DIFF TO REFLECT NUMBER OF FALSE
        neg = len(numpy_check) - pos

        ret_val = [neg, pos]
    elif flag == 1:
        """
        IF WE ARE HERE WE ARE PASSED A SUBSETTED SET OF VALUES: RETURN THE FOLLOWING 
            * number of postive given 0 
            * number of negative given 0 
            * number of positive given 1 
            * number of negative given 1
        """
        # CONVERT TO NUMPY TO DRASTICALLY CUT DOWN RUNTIME
        numpy_data = data.to_numpy()
        variable_data = numpy_data[:, 0]  # COL
        class_label = numpy_data[:, 1]  # VARIABLE

        # GET CLASSIFICATION given that the VARIABLE IS TRUE
        numPos1 = np.logical_and(variable_data, class_label)
        numPos1 = np.sum(numPos1)
        numNeg1 = np.sum(variable_data) - numPos1

        # GET CLASSIFICATION given that the VARIABLE IS FALSE
        forNeg = np.invert(variable_data)
        numPos0 = np.logical_and(forNeg, class_label)
        numPos0 = np.sum(numPos0)
        numNeg0 = np.sum(forNeg) - numPos0

        ret_val = [numNeg0, numPos0, numNeg1, numPos1]

    # RETURN WHATEVER MODE WE RAN'S RESULTS
    return ret_val


def numNodes(root):
    """
                   * DESCRIPTION :
                       *  as the name suggests we utilize this helper function in order to get the total number of nodes
                            contained in the graph. we do this by recursing going through the tree and chaining from
                            the bottom up

                   * INPUT PARAMETERS :
                       * root = root node of the tree we wish to count
    """
    number_nodes = 0

    if root.type != "terminal":
        # NUMBER OF NODES FOR THE RIGHT
        number_nodes = number_nodes + numNodes(root.right)

        # NUMBER OF NODES FOR THE LEFT
        number_nodes = number_nodes + numNodes(root.left)

        # DON'T FORGET THE CURRENT NODE
        number_nodes = number_nodes + 1

        # RETURN AT THE END OF THE RECURSIVE CALL
        return number_nodes
    elif root.type == "terminal":
        # END OF THE ROAD BUCKAROO
        return 1


def treeTest(tree, data):
    """
               * DESCRIPTION :
                   *  iterate through row by row and follow the Decision Tree provided. we iterated down to a terminal
                        node and then check at the terminal node if we successfully predicted correctly.

               * INPUT PARAMETERS :
                   * data: just the dataset associated with the current iteration
                   * tree: the tree we will use to test row by row
    """
    length = len(data["salaryLevel_B"])
    correct = 0

    for r in range(length):
        # ITERATE THROUGH TREE FOR THIS ROW
        row = data.iloc[r]
        check_val = iterateTree(row, tree)

        if check_val == row["salaryLevel_B"]:
            correct = correct + 1

    return (correct/length)


def iterateTree(row, tree):
    """
                   * DESCRIPTION :
                       *  partner method with testTree. This is the part where we iterate down to the terminal node
                            and return the prediction contained at that node.

                   * INPUT PARAMETERS :
                       * row: the data input we are currently checking against the tree
                       * tree: the decision tree we will go through to get a label
    """
    root = tree.root

    while root.type != "terminal":
        row_value = row[root.attribute]

        if row_value:
            # GO RIGHT
            root = root.right

        elif not row_value:
            # GO LEFT
            root = root.left

    return root.decision


def prunesAreGood(data, root, tree):
    """
           * DESCRIPTION :
               *  nothing much, basically iterate down to the second before the terminal node and calculate if the
                    tree will be better off without this node (piazza)

           * INPUT PARAMETERS :
               * data: just the dataset associated with the current iteration
               * root: the root node of the tree
               * tree: the tree to be pruned
    """
    if root.type != "terminal":
        # GO DOWN ALL THE WAY FOR THE LEFT BRANCH
        prunesAreGood(data, root.left, tree)

        # GO DOWN ALL THE WAY ON THE RIGHT BRANCH
        prunesAreGood(data, root.right, tree)

        # CALCULATE CURRENT ACCURACY
        old_accuracy = treeTest(tree, data)

        # PRUNE THE NODE OFF
        backup_type = root.type
        backup_attribute = root.attribute

        # CONVERT TO A TERMINAL NODE
        root.type = "terminal"

        # GO WITH THE DECISION AVAILABLE AT THAT LEVEL
        decision = False
        arr_val = numPosNumNeg(data, 0)
        if arr_val[1] > arr_val[0]:
            decision = True

        root.decision = decision

        # CALCULATE NEW ACCURACY
        new_accuracy = treeTest(tree, data)

        # DECISION TIME
        if new_accuracy < old_accuracy:
            # WE GOT WORSE SO DON'T PRUNE BB
            root.type = backup_type


def Vanilla(data):
    """
        * DESCRIPTION :
            * This is where we will actually build the tree according to the basic ID3 algorithm, no modifications

        * INPUT PARAMETERS  :
            * data: just the dataset associated with the iteration of the run
    """
    # THIS WILL BE THE NODE FOR THIS RUN, THIS WILL BE SET
    iter_node = Node()

    # KEEP TRACK OF THE MAX INFORMATION GAIN
    max_gain = 0
    max_gain_col = None

    # SOME HELPER CALLS
    ret_val = doWeStop(data)

    # IG / ENTROPY STEP
    if ret_val[0]:
        # CHECK OUR HALT CONDITION AND SEE IF WE HAVE REACHED A CONCLUSION POINT ("LEAF")
        iter_node.type = "terminal"  # TERMINAL NODE (LEAF)
        iter_node.decision = ret_val[1]  # DECISION TO MAKE BASED ON SAME LABEL
    else:
        # WELCOME TO INFORMATION GAIN / ENTROPY BASED EXPANSION
        # GET THE E(S) --> TOTAL ENTROPY OF THE DATA SET
        totalValues = numPosNumNeg(data, 0)
        e_s = Entropy(totalValues)

        for col in data:
            # GET THE GAIN FOR THAT ATTRIBUTE
            if col == 'salaryLevel_B':
                continue

            temp_gain = InfoGain(data[[col, 'salaryLevel_B']], e_s)

            # KEEP TRACK CONSTANTLY OF THE MAXIMUM GAIN
            if temp_gain > max_gain:
                max_gain = temp_gain
                max_gain_col = col

        # TERMINATE IN THIS CASE
        if max_gain == 0:
            iter_node.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.decision = decision
            return iter_node

        # NOW THAT WE KNOW WHICH NODE SET CURRENT NODES TO ATTRIBUTES TO REFLECT PARENT STATUS
        iter_node.attribute = max_gain_col
        iter_node.type = "branch"
        iter_node.dataset = data

        # GET THE LEFT & RIGHT DATSETS
        left_conditional = data[max_gain_col] == 0
        right_conditional = data[max_gain_col] == 1

        left_data = data[left_conditional]
        right_data = data[right_conditional]

        left_data = left_data.drop(columns=[max_gain_col])
        right_data = right_data.drop(columns=[max_gain_col])

        # SET THE RIGHT CHILD
        if not right_data.empty:
            # THIS MEANS THE DATASET IS NOT EMPTY AND CAN BE SET
            iter_node.right = Vanilla(right_data)
        else:
            # EMPTY TERMINATE
            iter_node.right = Node()
            iter_node.right.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.right.decision = decision

        # SET THE LEFT CHILD
        if not left_data.empty:
            # THIS MEANS THE DATASET IS NOT EMPTY AND CAN BE SET
            iter_node.left = Vanilla(left_data)
        else:
            # EMPTY TERMINATE
            iter_node.left = Node()
            iter_node.left.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.left.decision = decision

    return iter_node


def MaxDepth(data, level, max_depth):
    """
        * DESCRIPTION :
            *  A modification of Vanilla but stop at the spec'd depth

        * INPUT PARAMETERS :
            * data: just the dataset associated with the current iteration
            * level: the level at which the current recursive call is at
            * max_depth: the maximum allowable depth for this tree
    """
    # THIS WILL BE THE NODE FOR THIS RUN, THIS WILL BE SET
    iter_node = Node()

    # KEEP TRACK OF THE MAX INFORMATION GAIN
    max_gain = 0
    max_gain_col = None

    # SOME HELPER CALLS
    ret_val = doWeStop(data)

    # CHECK IF THE MAXIMUM LEVEL WAS EXCEEDED
    if level >= max_depth:
        # IF WE HAVE REACHED THE LEVEL THRESHOLD: TERMINATE
        iter_node.type = 'terminal'

        decision = False
        arr_val = numPosNumNeg(data, 0)
        if arr_val[1] > arr_val[0]:
            decision = True

        iter_node.decision = decision

        return iter_node

    # IG / ENTROPY STEP
    elif ret_val[0]:
        # CHECK OUR HALT CONDITION AND SEE IF WE HAVE REACHED A CONCLUSION POINT ("LEAF")
        iter_node.type = "terminal"  # TERMINAL NODE (LEAF)
        iter_node.decision = ret_val[1]  # DECISION TO MAKE BASED ON SAME LABEL
        return iter_node

    else:
        # WELCOME TO INFORMATION GAIN / ENTROPY BASED EXPANSION
        # GET THE E(S) --> TOTAL ENTROPY OF THE DATA SET
        totalValues = numPosNumNeg(data, 0)
        e_s = Entropy(totalValues)

        for col in data:
            # GET THE GAIN FOR THAT ATTRIBUTE
            if col == 'salaryLevel_B':
                continue

            temp_gain = InfoGain(data[[col, 'salaryLevel_B']], e_s)

            # KEEP TRACK CONSTANTLY OF THE MAXIMUM GAIN
            if temp_gain > max_gain:
                max_gain = temp_gain
                max_gain_col = col

        # TERMINATE IN THIS CASE
        if max_gain == 0:
            iter_node.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.decision = decision
            return iter_node

        # NOW THAT WE KNOW WHICH NODE SET CURRENT NODES TO ATTRIBUTES TO REFLECT PARENT STATUS
        iter_node.attribute = max_gain_col
        iter_node.type = "branch"

        # GET THE LEFT & RIGHT DATSETS
        left_conditional = data[max_gain_col] == 0
        right_conditional = data[max_gain_col] == 1

        left_data = data[left_conditional]
        right_data = data[right_conditional]

        left_data = left_data.drop(columns=[max_gain_col])
        right_data = right_data.drop(columns=[max_gain_col])

        # SET THE RIGHT CHILD
        if not right_data.empty:
            # THIS MEANS THE DATASET IS NOT EMPTY AND CAN BE SET
            level = level + 1
            iter_node.right = MaxDepth(right_data, level, max_depth)
        else:
            # EMPTY TERMINATE
            iter_node.right = Node()
            iter_node.right.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.right.decision = decision

        # SET THE LEFT CHILD
        if not left_data.empty:
            # THIS MEANS THE DATASET IS NOT EMPTY AND CAN BE SET
            level = level + 1
            iter_node.left = MaxDepth(left_data, level, max_depth)
        else:
            # EMPTY TERMINATE
            iter_node.left = Node()
            iter_node.left.type = "terminal"

            decision = False
            arr_val = numPosNumNeg(data, 0)
            if arr_val[1] > arr_val[0]:
                decision = True

            iter_node.left.decision = decision

    return iter_node


if __name__ == '__main__':
    """
            * DESCRIPTION :
                *  Take in the input of the user and calculate the right function to call 
                *  Use the argv[] to make the right decision to go forward

            * INPUT PARAMETERS :
                *argv[1] = train-set.data
                *argv[2] = test-set.data 
                *argv[3] = type of model we are running one of the following: {prune, vanilla, maxDepth} 
                *argv[4] = train percentage for both maxDepth and prune 
                *argv[5] = test percentage for both maxDepth and prune 
                *argv[6] = maximum level allowed for maxDepth  
    """
    # THIS RUNS THE PROGRAM

    # FIRST DEFINE THE INPUT ARGUMENTS
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    type_run = sys.argv[3]
    train_percent = sys.argv[4]

    # BEFORE YOU GO TO THE TYPE OF EXECUTION PROCESS THE TEST AND TRAINING DATA
    # print(float(train_percent)/100)
    train_p = PreProcess(train_data, float(train_percent)/100, "train")
    test_p = PreProcess(test_data, 1, "test")

    # DEPENDING ON THE INPUT SELECT THE PROPER PATH FOR EXECUTION
    if type_run == "prune":
        # DETERMINE HOW MUCH TO TAKE FOR OUR VALIDATION SET
        validation_percent = sys.argv[5]

        # DO OUR OWN PREPROCESS HERE
        train_pure = PreProcess(train_data, 1, "test")

        # GET THE LENGTH OF ALL THE ROWS
        length = len(train_pure["salaryLevel_B"])

        # TAKE THE FIRST PERCENT
        cut_one = float(train_percent)/100
        cut_one = round(length * cut_one)
        first_cut = train_pure.head(cut_one)

        # TAKE THE SECOND PERCENT
        cut_two = float(validation_percent)/100
        cut_two = round(length * cut_two)
        second_cut = train_pure.tail(cut_two)

        # BUILD THE TREE BASED ON THE CUT
        p_tree = Tree(first_cut, "vanilla", 1000000000000)
        # print("TREE BUILT")

        # ACCURACY PRIOR TO PRUNE
        # print("ACCURACY PRE PRUNE: ")
        # pre_train_accuracy = treeTest(p_tree, first_cut)
        # pre_test_accuracy = treeTest(p_tree, second_cut)
        # print("Train set accuracy: %f" % pre_train_accuracy)
        # print("Test set accuracy: %f" % pre_test_accuracy)
        # print("Number of Nodes: ")
        # print(numNodes(p_tree.root))

        # PRUNE THE TREE BB GURL
        # print("RUNNING PRUNING")
        prunesAreGood(second_cut, p_tree.root, p_tree)

        # EVALUATE THE TREE ACCORDING TO TRAINING SET
        # print("CHECKING ACCURACY: ")
        p_train_accuracy = treeTest(p_tree, first_cut)

        # SECOND FOR THE TEST
        p_test_accuracy = treeTest(p_tree, second_cut)

        # PRINT THE OUTCOME
        print("Train set accuracy: %f" % p_train_accuracy)
        print("Test set accuracy: %f" % p_test_accuracy)
        # print("Number of Nodes: ")
        # print(numNodes(p_tree.root))

    elif type_run == "maxDepth":
        validation_percent = sys.argv[5]
        maxLevel = sys.argv[6]

        # DO OUR OWN PREPROCESS HERE
        train_pure = PreProcess(train_data, 1, "test")

        # GET THE LENGTH OF ALL THE ROWS
        length = len(train_pure["salaryLevel_B"])

        # TAKE THE FIRST PERCENT
        cut_one = float(train_percent) / 100
        cut_one = round(length * cut_one)
        first_cut = train_pure.head(cut_one)

        # TAKE THE SECOND PERCENT
        cut_two = float(validation_percent) / 100
        cut_two = round(length * cut_two)
        second_cut = train_pure.tail(cut_two)

        # BUILD TREE BASED ON FIRST CUT
        m_tree = Tree(first_cut, type_run, int(maxLevel))

        # TESTING ACCURACY
        train_accuracy = treeTest(m_tree, first_cut)
        test_accuracy = treeTest(m_tree, second_cut)

        # PRINT
        print("Train set accuracy: %f" % train_accuracy)
        print("Test set accuracy: %f" % test_accuracy)

    elif type_run == "vanilla":
        # print("MAJ LABEL")
        # print(numPosNumNeg(train_p, 0))
        v_tree = Tree(train_p, type_run, 1000000000000)
        train_accuracy = treeTest(v_tree, train_p)
        test_accuracy = treeTest(v_tree, test_p)
        print("Train set accuracy: %f" % train_accuracy)
        print("Test set accuracy: %f" % test_accuracy)
        # print("Total number of nodes: ")
        # print(numNodes(v_tree.root))


