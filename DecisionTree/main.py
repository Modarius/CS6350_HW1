# Written by Alan Felt for CS6350 Machine Learning
from fileinput import filename
import numpy as np
import os

def validateData(terms, attrib, attrib_labels, data_labels):
    for i in np.arange(np.size(terms, 1)-1): # for all of the potential attributes' values
        if (set(np.unique(terms[:,i])) != attrib[attrib_labels[i]]): # if the set of the attributes values differs from the set of unique values in the data in that attributes column
            print("Attribute " + attrib_labels[i] + " cannot take value " + str(set(np.unique(terms[:,i])).difference(attrib[attrib_labels[i]]))) # print the offending invalid attribute value
            terms = list() # clear the data
    if (set(np.unique(terms[:,6])) != data_labels): # check if the data labels in the data are within the set of potential data labels
            print("Data Label cannot take value " + str(set(np.unique(terms[:,i])).difference(attrib[attrib_labels[i]])))
            terms = list() # clear the data
    return terms

def importData(filename, attrib, attrib_labels, data_labels):
    terms = list()
    with open(filename, 'r') as f:
        for line in f :
            terms.append(line.strip().split(','))
    terms = np.array(terms)
    terms = validateData(terms, attrib, attrib_labels, data_labels)
    return terms

def DecisionTree():
    attrib_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    attrib = {
        'buying':{'vhigh', 'high', 'med', 'low'},
        'maint':{'vhigh', 'high', 'med', 'low'},
        'doors':{'2', '3', '4', '5more'},
        'persons':{'2', '4', 'more'},
        'lug_boot':{'small', 'med', 'big'},
        'safety':{'low', 'med', 'high'}
    }
    data_labels = {'unacc', 'acc', 'good', 'vgood'}

    terms = importData("car/train.csv", attrib, attrib_labels, data_labels)

if __name__ == "__main__":
    DecisionTree()