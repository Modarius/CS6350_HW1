# Written by Alan Felt for CS6350 Machine Learning

import numpy as np
import pandas as pd


# for leaf nodes, name is the same as the branch it is connected to
# for regular nodes, name is the name of the attribute it represents. Its children will be named the values of the attributes values
class Node:
    def __init__(self, name_in, type_in, parent_node_in, children_node_in, label_in, depth_in):
        self.name = name_in  # name of the node
        self.type = type_in  # 'root', 'node', 'leaf', 'unknown'
        self.parent = parent_node_in  # will include a node (if not the root)
        self.children = children_node_in  # will include node(s) instance
        self.label = label_in
        self.depth = depth_in
        return

    def setChild(self, child_name_in, child_node_in):
        if self.children is None: # if there are no children, add one in dictionary form
            self.children = {child_name_in: child_node_in}
        elif child_name_in not in self.children: # otherwise check if the child is already there
            self.children[child_name_in] = child_node_in
        else: # this statement should never run but is there just in case something goes wrong
            print("could not add" + child_name_in)
        return

    def getDepth(self):
        return self.depth

    def getName(self):
        return self.name
    
    def getLabel(self):
        return self.label

    def getType(self):
        return self.type

    def getChildren(self):
        return self.children

    def getChild(self, child_name_in):
        if child_name_in not in self.children:
            return None # this should never be called
        else:
            return self.children[child_name_in] # return the child node with the attribute provided in child_name_in


def validData(terms, attrib, data_labels):
    for A in attrib.keys(): # for all the attributes possible
        data_attrib_values = set(terms.get(A).unique()) # get the set of values for attribute A
        if (not attrib[A].issubset(data_attrib_values)): # check if there is a value from the data not included in the possible values for A
            # print the offending invalid attribute value
            print("Attribute " + A + " cannot take value " +
                  str(data_attrib_values.difference(attrib[A])))
            return False
    if (not set(terms.index.unique().to_numpy()).issubset(data_labels)): # also check that all the labels are valid
        print("Data Label cannot take value " +
              str(set(terms.index.unique()).difference(data_labels))) # return values that are not valid
        return False
    return True


def importData(filename, attrib, attrib_labels, data_labels, index_col=0, numeric_data=None):
    terms = pd.read_csv(filename, sep=',', names=attrib_labels, index_col=index_col) # read in the csv file into a DataFrame object
    if (numeric_data != None): # if there is information on which columns are numeric
        for label in numeric_data.keys(): #the for all the labels in numeric data
            column = terms.get(label) # get the column pertaining to that label
            new_column = column.copy(deep=True) # make a second copy, but not a linked copy
            split_value = np.median(column.to_numpy()) # find the median numeric value to split on
            # find all the values in the column that are less than and equal to 
            # and greater than and replace them with a label from numeric_data
            new_column.where(column <= split_value, numeric_data[label][0], inplace=True) 
            new_column.where(column > split_value, numeric_data[label][1], inplace=True)
            terms[label] = new_column # replace the column with the updated one

    for A in terms.columns.to_numpy():
        if(terms[A].unique().__contains__('unknown')): # if the column contains unknown values
            column2 = terms.get(A) # get that column
            new_column2 = column2.copy(deep=True)  # make a duplicate of the column
            l, c = np.unique(column2.to_numpy(), return_counts=True) # find the most comon value in attribute A
            # this is a hacky way of getting the index of 'unknown' in l and c (unique labels, and their counts)
            idx = np.squeeze(np.where(l == 'unknown'))[()] # https://thispointer.com/find-the-index-of-a-value-in-numpy-array/, https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
            l = np.delete(l, idx) # remove unknown from the running for most common value
            c = np.delete(c, idx) # remove unknown from the running for most common value
            best_value = l[c.argmax()] # find the most common value (index into L with the index of the largest # in c)
            new_column2.where(column2 != 'unknown', best_value, inplace=True) # replace "unknown" with most common value
            terms[A] = new_column2 # replace the column with the updated values

    if (not validData(terms, attrib, data_labels)): # check for incorrect attribute values
        return
    return terms


def entropy(S):
    labels = S.index.to_numpy() # get the labels in S
    num_S = len(labels) # number of labels is size of S
    l, c = np.unique(labels, return_counts=True) # find unique labels in S
    p = c / num_S # calculate array of probabilities of each label
    H_S = -np.sum(p * np.log2(p)) # sum of the probabilites multiplied by log2 of probabilities
    return H_S


def majorityError(S): 
    labels = S.index.to_numpy()  # get all the labels in the current set S
    num_S = len(labels)  # count the labels

    # find all the unique labels and how many of each unique label there are
    l, c = np.unique(labels, return_counts=True)
    best_choice = c.argmax()  # choose the label with the greatest representation

    # delete the count of the label with the greatest representation
    # sum up the number of remaining labels
    neg_choice = np.sum(np.delete(c, best_choice, axis=0))

    # calculate ratio of # of not best labels over the total number of labels
    m_error = neg_choice / num_S

    # return this number
    return m_error


def giniIndex(S):
    labels = S.index.to_numpy() # get all labels in S
    num_S = len(labels) # count the labels in S
    l, c = np.unique(labels, return_counts=True) # find the unique labels and their respective counts
    p_l = c / num_S # calculate the probability of each label
    gi = 1 - np.sum(np.square(p_l)) # square and sum the probabilities
    return gi


def infoGain(S, method='entropy'):
    if (method == 'majority_error'): # choose method to use
        Purity_S = majorityError(S)
    elif (method == 'gini'):
        Purity_S = giniIndex(S)
    else:
        Purity_S = entropy(S)
    num_S = np.size(S, 0) # number of entries in S
    ig = dict() # initialize a dictionary
    best_ig = 0 # track the best information gain
    best_attribute = "" # track the Attribute of the best information gain
    for A in S.columns:  # for each attribute in S
        total = 0
        # get the unique values that attribute A has in S
        values_A = S.get(A).unique()
        for v in values_A:  # for each of those values
            # select a subset of S where S[A] equals that value of A
            Sv = S[S[A] == v]
            # get the size of the subset (number of entries)
            num_Sv = np.size(Sv, 0)
            if (method == 'majority_error'):  # choose the method for getting the purity value
                Purity_Sv = majorityError(Sv)
            elif (method == 'gini'):  # this seems to work
                Purity_Sv = giniIndex(Sv)
            else:
                Purity_Sv = entropy(Sv)
            # sum the weighted values of each purity for v in A
            total = total + num_Sv/num_S * Purity_Sv
        # subtract the sum from the purity of S to get the information gain
        ig[A] = Purity_S - total
        if (ig[A] >= best_ig):  # if that information gain is better than the others, select that attribute as best
            best_attribute = A
            best_ig = ig[A]
    if (best_attribute == ""): # handle edge case where no value is best, in that case just return what is in A (should be one value)
        best_attribute = A
    # once we have checked all attributes A in S, return the best attribute to split on
    return best_attribute


def bestLabel(S):
    l, c = np.unique(S.index.to_numpy(), return_counts=True) # find the best label in S (most common)
    best_label = l[c.argmax()]
    return best_label


def ID3(S, attribs, root, method, max_depth):
    # Check if all examples have one label
    # Check whether there are no more attributes to split on
    # if so make a leaf node
    if (S.index.unique().size == 1 or S.columns.size == 0):
        label = bestLabel(S)
        return Node(label, "leaf", None, None, label, root.getDepth() + 1)

    A = infoGain(S, method) # get the best attribute to split on

    if (root == None): # if there is no root, make one
        new_root = Node(A, "root", None, None, None, 0)
    else:
        new_root = Node(A, "node", None, None, None, root.getDepth() + 1)

    # v is the branch, not a node, unless v splits the dataset into one with no subsets
    for v in attribs[A]:
        Sv = S[S[A] == v].drop(A, axis=1) # find the subset where S[A] == v and drop the column A
        
        if (Sv.index.size == 0):  # if the subset is empty, make a child with the best label in S
            v_child = Node(v, "leaf", None, None, bestLabel(S), root.getDepth() + 1)
        elif (new_root.getDepth() == (max_depth - 1)): # if we are almost at depth, truncate and make a child with the best label in the subset
            #print("At depth, truncating")
            v_child = Node(v, "leaf", None, None, bestLabel(Sv), new_root.getDepth() + 1)
        else:  # if the subset is not empty make a child with the branch v but not the node name v, node name will be best attribute found for splitting Sv
            v_child = ID3(Sv, attribs, new_root, method, max_depth) # recursive call down the tree
        new_root.setChild(v, v_child) # set a new child of new_root
    return new_root


def follower(data, tree):
    if (tree.getType() != 'leaf'): # if we're not at a leaf
        v = data.pop(tree.getName()) # pop off the value from data which corresponds to the name of the current node
        return follower(data, tree.getChild(v)) # recursive call down the tree
    else:
        return tree.getLabel() # if we're at a leaf, return the label of the leaf


# this is a kludgy way of printing the tree
# referenced https://simonhessner.de/python-3-recursively-print-structured-tree-including-hierarchy-markers-using-depth-first-search/
def printTree(tree):
    ttype = tree.getType() # get all the data for the current node
    tname = tree.getName()
    tlabel = tree.getLabel()
    tchildren = tree.getChildren()
    print('\t' * tree.getDepth(), end='') # pad the name by depth # of tab characters

    if(ttype == "leaf"): # if we're a leaf, print the label with || around it
        print('|' + tlabel + '|')
        return
    elif(ttype == "root" or ttype == "node"): # if we're a root or node, just print the name
        print(tname)
    for c in tchildren: # for each of the children of the current node
        print('\t' * tree.getDepth() + '-> ' + c) # pad the child name by depth tabs and a ->
        printTree(tchildren[c]) # recursively call this function
    return


def treeError(tree, S):
    c_right = 0
    c_wrong = 0
    for data in S.itertuples(index=True): # for each datapoint in S
        if (data.Index != follower(data._asdict(), tree)): # check if the label returned by the tree is the same as the provided label
            c_wrong += 1
            #print("not a match")
        else:
            c_right += 1
            #print("matched!")
    error = c_wrong / (c_right + c_wrong) # find the ratio of wrong answers to all answers (error)
    return error


def DecisionTree():
    # initialization data for car dataset
    # attrib_labels = ['buying', 'maint', 'doors',
    #                  'persons', 'lug_boot', 'safety']
    # attribs = {
    #     'buying': {'vhigh', 'high', 'med', 'low'},
    #     'maint': {'vhigh', 'high', 'med', 'low'},
    #     'doors': {'2', '3', '4', '5more'},
    #     'persons': {'2', '4', 'more'},
    #     'lug_boot': {'small', 'med', 'big'},
    #     'safety': {'low', 'med', 'high'}
    # }
    # data_labels = {'unacc', 'acc', 'good', 'vgood'}
    # train_filepath = "car/train.csv"
    # test_filepath = "car/test.csv"
    # numeric_data = None
    # index_col = 6

    # initilalization data for bank dataset
    attrib_labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
            'blue-collar','self-employed','retired','technician','services'}, # 'unknown'
        'marital':{"married","divorced","single"}, 
        'education':{"secondary","primary","tertiary"},  # "unknown"
        'default':{'yes', 'no'}, 
        'balance':{'low', 'high'}, # this is a numeric value which will be converted to categorical
        'housing':{'yes', 'no'}, 
        'loan':{'yes', 'no'}, 
        'contact':{'telephone', 'cellular'}, #'unknown'
        'day':{'early', 'late'}, # this is a numeric value which will be converted to categorical
        'month':{"jan", "feb", "mar", 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', "nov", "dec"}, 
        'duration':{'short', 'long'}, # this is a numeric value which will be converted to categorical
        'campaign':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'pdays':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'previous':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'poutcome':{'other', 'failure', 'success'} # 'unknown'
        }
    data_labels = {'yes', 'no'}
    train_filepath = 'bank/train.csv'
    test_filepath = 'bank/test.csv'
    numeric_data = {
        'age':['young', 'old'], # this is a numeric value which will be converted to categorical
        'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
        'day':['early', 'late'], # this is a numeric value which will be converted to categorical
        'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
        'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
        'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
        'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
        }
    index_col = 16 
    
    training_data = importData(train_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)
    test_data = importData(test_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)
    train_error = np.zeros([16,1])
    test_error = np.zeros([16,1])
    for max_depth in np.arange(start=1, stop=17):
        tree = ID3(training_data, attribs, None, 'entropy', max_depth) # build the tree
        print("Depth: " + str(max_depth)) # print out current max_depth value
        # printTree(tree) # crude print of tree
        train_error[max_depth - 1 ] = treeError(tree, training_data) # test the error for the given dataset
        test_error[max_depth - 1] = treeError(tree, test_data) # test the error for the given dataset
    print('Avg Error Training Dataset = ' + str(np.average(train_error)))
    print('Avg Error Test Dataset = ' + str(np.average(test_error)))

    return

if __name__ == "__main__":
    DecisionTree()
