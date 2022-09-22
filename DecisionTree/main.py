# Written by Alan Felt for CS6350 Machine Learning


import numpy as np
import os
import pandas as pd
import anytree as at
from anytree.exporter import UniqueDotExporter


def validData(terms, attrib, data_labels):
    for A in attrib.keys():
        data_attrib_values = set(terms.get(A).unique())
        if (data_attrib_values != attrib[A]):
            # print the offending invalid attribute value
            print("Attribute " + A + " cannot take value " +
                  str(data_attrib_values.difference(attrib[A])))
            return False
    if (data_labels != set(terms.index.unique().to_numpy())):
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
    H_S = -np.sum(p * np.log(p))
    return H_S

def majorityError(S): # this doesn't seem to be working
    labels = S.index.to_numpy()
    num_S = len(labels)
    l, c = np.unique(labels, return_counts=True)
    best_choice = c.argmax()
    neg_choice = float(np.sum(np.delete(c, best_choice, axis=0)))
    m_error = neg_choice / num_S
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
    for A in S.columns:
        total = 0
        values_A = S.get(A).unique()
        for v in values_A:
            Sv = S[S[A] == v]
            num_Sv = np.size(Sv, 0)
            if (method == 'majority_error'):
                Purity_Sv = majorityError(Sv)
            elif (method == 'gini'):
                Purity_Sv = giniIndex(Sv)
            else: 
                Purity_Sv = entropy(Sv)
            total = total + num_Sv/num_S * Purity_Sv
        ig[A] = Purity_S - total
        if (ig[A] >= best_ig):
            best_attribute = A
            best_ig = ig[A]
    return best_attribute

def leafNode(S, root):
    l, c = np.unique(S.index.to_numpy(), return_counts=True)
    best_label = l[c.argmax()]
    if (root == None):
        return at.Node(best_label)
    return at.Node(best_label, root)

def ID3(S, attribs, root=None, method='entropy', max_depth=np.inf):
    if (root != None):
        if (root.depth == (max_depth - 2)):
            return leafNode(S, root)
    # check if all examples have one label and whether there are no more attributes to split on
    if (S.index.unique().size == 1 or S.columns.size == 0):
        return leafNode(S, root)

    A = infoGain(S, method)

    if (root == None):
        new_root = at.Node(A)
    else:
        new_root = at.Node(A, root)

    for v in attribs[A]:
        new_branch = at.Node(name=v, parent=new_root)
        Sv = S[S[A] == v].drop(A, axis=1)
        if (Sv.index.size == 0): # how does this account for when there is only one attribute and 
            leafNode(S, root=new_root)
        else:
            ID3(Sv, attribs, root=new_root, max_depth=max_depth)
    return new_root

def findLabel(data, tree):
    next_node = tree.children == data[tree.name] 

# def labelData(S, tree):
#     for i in S:
#         findLabel(data, tree)

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

    S = importData("car/train.csv", attribs, attrib_labels, data_labels)
    tree = ID3(S, attribs)
    # print(at.RenderTree(tree, style=at.AsciiStyle()))
    for pre, _, node in at.RenderTree(tree):
        print("%s%s" % (pre, node.name))
    UniqueDotExporter(tree).to_picture("tree.png")

    # labelData(S, tree)

if __name__ == "__main__":
    DecisionTree()
