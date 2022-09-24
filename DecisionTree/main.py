# Written by Alan Felt for CS6350 Machine Learning

import numpy as np
import pandas as pd
from enum import Enum


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
        if self.children is None:
            self.children = {child_name_in: child_node_in}
        elif child_name_in not in self.children:
            self.children[child_name_in] = child_node_in
        else:
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
    for A in attrib.keys():
        data_attrib_values = set(terms.get(A).unique())
        if (not attrib[A].issubset(data_attrib_values)):
            # print the offending invalid attribute value
            print("Attribute " + A + " cannot take value " +
                  str(data_attrib_values.difference(attrib[A])))
            return False
    if (not set(terms.index.unique().to_numpy()).issubset(data_labels)):
        print("Data Label cannot take value " +
              str(set(terms.index.unique()).difference(data_labels)))
        return False
    return True


def importData(filename, attrib, attrib_labels, data_labels):
    terms = pd.read_csv(filename, sep=',', names=attrib_labels, index_col=6)
    if (not validData(terms, attrib, data_labels)):
        return
    return terms


def entropy(S):
    labels = S.index.to_numpy()
    num_S = len(labels)
    l, c = np.unique(labels, return_counts=True)
    p = c / num_S
    H_S = -np.sum(p * np.log2(p))
    return H_S


def majorityError(S):  # this doesn't seem to be working
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
    labels = S.index.to_numpy()
    num_S = len(labels)
    l, c = np.unique(labels, return_counts=True)
    p_l = c / num_S
    gi = 1 - np.sum(np.square(p_l))
    return gi


def infoGain(S, method='entropy'):
    if (method == 'majority_error'):
        Purity_S = majorityError(S)
    elif (method == 'gini'):
        Purity_S = giniIndex(S)
    else:
        Purity_S = entropy(S)
    num_S = np.size(S, 0)
    ig = dict()
    best_ig = 0
    best_attribute = ""
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
                Purity_Sv = majorityError(Sv)  # this doesn't work
            elif (method == 'gini'):  # this seems to work fine
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
    # once we have checked all attributes A in S, return the best attribute to split on
    return best_attribute


def bestLabel(S):
    l, c = np.unique(S.index.to_numpy(), return_counts=True)
    best_label = l[c.argmax()]
    return best_label


def ID3(S, attribs, root, method, max_depth):
    # Check if all examples have one label
    # Check whether there are no more attributes to split on
    # if so make a leaf node
    if (S.index.unique().size == 1 or S.columns.size == 0):
        label = bestLabel(S)
        return Node(label, "leaf", None, None, label, root.getDepth() + 1)

    A = infoGain(S, method)

    if (root == None):
        new_root = Node(A, "root", None, None, None, 0)
    else:
        # need to change parent to root
        new_root = Node(A, "node", None, None, None, root.getDepth() + 1)

    # v is the branch, not a node, unless v splits the dataset into one with no subsets
    for v in attribs[A]:
        Sv = S[S[A] == v].drop(A, axis=1)
        
        if (Sv.index.size == 0):  # if the subset is empty, make a child with the best label in S
            v_child = Node(v, "leaf", None, None, bestLabel(S), root.getDepth() + 1)
        elif (new_root.getDepth() == (max_depth - 1)): # if we are almost at depth, truncate and make a child with the best label in the subset
            #print("At depth, truncating")
            v_child = Node(v, "leaf", None, None, bestLabel(Sv), new_root.getDepth() + 1)
        else:  # if the subset is not empty make a child with the branch v but not the node name v, node name will be best attribute found for splitting Sv
            v_child = ID3(Sv, attribs, new_root, method, max_depth)
        new_root.setChild(v, v_child)
    return new_root


def follower(data, tree):
    if (tree.getType() != 'leaf'):
        v = data.pop(tree.getName())
        return follower(data, tree.getChild(v))
    else:
        return tree.getLabel()

def printTree(tree):
    ttype = tree.getType()
    tname = tree.getName()
    tlabel = tree.getLabel()
    tchildren = tree.getChildren()
    print('\t' * tree.getDepth(), end='')

    if(ttype == "leaf"):
        print('|' + tlabel + '|')
        return
    elif(ttype == "root" or ttype == "node"):
        print(tname)
    for c in tchildren:
        print('\t' * tree.getDepth() + '-> ' + c)
        printTree(tchildren[c])
    return

def treeError(tree, S):
    c_right = 0
    c_wrong = 0
    for data in S.itertuples(index=True):
        if (data.Index != follower(data._asdict(), tree)):
            c_wrong += 1
            #print("not a match")
        else:
            c_right += 1
            #print("matched!")
    error = c_wrong / (c_right + c_wrong)
    return error
def DecisionTree():
    attrib_labels = ['buying', 'maint', 'doors',
                     'persons', 'lug_boot', 'safety']
    attribs = {
        'buying': {'vhigh', 'high', 'med', 'low'},
        'maint': {'vhigh', 'high', 'med', 'low'},
        'doors': {'2', '3', '4', '5more'},
        'persons': {'2', '4', 'more'},
        'lug_boot': {'small', 'med', 'big'},
        'safety': {'low', 'med', 'high'}
    }
    data_labels = {'unacc', 'acc', 'good', 'vgood'}

    training_data = importData("car/train.csv", attribs, attrib_labels, data_labels)
    test_data = importData("car/test.csv", attribs, attrib_labels, data_labels)
    train_error = np.zeros([6,1])
    test_error = np.zeros([6,1])
    for max_depth in np.arange(start=1, stop=7):
        tree = ID3(training_data, attribs, None, 'entropy', max_depth)
        # printTree(tree)
        train_error[max_depth - 1 ] = treeError(tree, training_data)
        test_error[max_depth - 1] = treeError(tree, test_data)
    print('Avg Error Training Dataset = ' + str(np.average(train_error)))
    print('Avg Error Test Dataset = ' + str(np.average(test_error)))
    print(train_error)
    print(test_error)

    return

if __name__ == "__main__":
    DecisionTree()
