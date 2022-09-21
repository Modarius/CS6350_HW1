# Written by Alan Felt for CS6350 Machine Learning
from cProfile import label
from fileinput import filename
import numpy as np
import os
import pandas as pd
import anytree as at

def validData(terms, attrib, data_labels):
    for A in attrib.keys():
        data_attrib_values = set(terms.get(A).unique())
        if (data_attrib_values != attrib[A]):
            print("Attribute " + A + " cannot take value " + str(data_attrib_values.difference(attrib[A]))) # print the offending invalid attribute value
            return False
    if (data_labels != set(terms.index.unique().to_numpy())):
        print("Data Label cannot take value " + str(set(terms.index.unique()).difference(data_labels)))
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

def infoGain(S):
    H_S = entropy(S)
    num_S = np.size(S,0)
    ig = dict()
    best_ig = 0
    best_attribute = ""
    for A in S.columns:
        total = 0
        values_A = S.get(A).unique()
        for v in values_A:
            Sv = S[S[A] == v]
            num_Sv = np.size(Sv,0)
            H_Sv = entropy(Sv)
            total += num_Sv/num_S * H_Sv
        ig[A] = H_S - total
        if (ig[A] > best_ig):
            best_attribute = A
            best_ig = ig[A]
    return best_attribute

        

def ID3(S, attribs, root=None):
    if (S.index.unique().shape[0] == 1):
        if (root == None):
            return at.Node(S.index.unique())
        else:
            return at.Node(S.index.unique(), root)

    best_attribute = infoGain(S)
    if (root == None):
        new_node = at.Node(best_attribute)
    else:
        new_node = at.Node(best_attribute,root)
    for a in attribs[best_attribute]:
        new_child = at.Node(name=a, parent=new_node)
        ID3(S[S[best_attribute] == a].drop(best_attribute,axis=1), attribs, root=new_child)

def DecisionTree():
    attrib_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    attribs = {
        'buying':{'vhigh', 'high', 'med', 'low'},
        'maint':{'vhigh', 'high', 'med', 'low'},
        'doors':{'2', '3', '4', '5more'},
        'persons':{'2', '4', 'more'},
        'lug_boot':{'small', 'med', 'big'},
        'safety':{'low', 'med', 'high'}
    }
    data_labels = {'unacc', 'acc', 'good', 'vgood'}

    terms = importData("car/train.csv", attribs, attrib_labels, data_labels)
    ID3(terms, attribs)

if __name__ == "__main__":
    DecisionTree()