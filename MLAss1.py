import sys
import csv
import math
import copy
import time
import numpy as np
import random
from collections import Counter
from numpy import *
feature = []

#Multiclass classifier decision tree using ID3 algorithm

#normalize the entire dataset prior to learning using min-max normalization
def normalize(matrix):
    a= np.array(matrix)
    a = a.astype(np.float)
    b = np.apply_along_axis(lambda x: (x-np.min(x))/float(np.max(x)-np.min(x)), 0, a)
    return b

# reading from the file using numpy genfromtxt
def load_csv(file):
    X = genfromtxt(file, delimiter=",", dtype=str)
    return(X)

#method to randomly shuffle the array
def random_np_array(ar):
    np.random.shuffle(ar)
    arr = ar
    return arr

#Normalize the data and generate the training labels,training features, test labels and test training
def generate_set(X):
    Y = X[:, -1]
    j = Y.reshape(len(Y),1)
    new_X = X[:, :-1]
    normalized_X = normalize(new_X)
    normalized_final_X = np.concatenate((normalized_X, j), axis=1)
    X = normalized_final_X
    size_of_rows = X.shape[0]
    num_test = round(.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end, :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        y_test = X_test[:, -1]
        y_test = y_test.flatten()
        y_training = X_training[:, -1]
        y_training = y_training.flatten()
        X_test = X_test[:, :-1]
        X_training = X_training[:, :-1]
        X_test = X_test.astype(np.float)
        X_training = X_training.astype(np.float)
        test_attri_list.append(X_test)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training)
        training_class_names_list.append(y_training)
        start = end
        end = end+num_test
    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list #(X_test,y_test,X_training,y_train)

# ID3 entropy calculation
def entropy(y):
    class_freq = {}
    attribute_entropy = 0.0
    for i in y:
        if i in class_freq:
            class_freq[i] += 1
        else:
            class_freq[i] = 1
    #print(class_freq)
    for freq in class_freq.values():
        attribute_entropy += (-freq/float(len(y))) * math.log(freq/float(len(y)),2)
    #print(attribute_entropy)
    return attribute_entropy

#calculating the predicited accuracy
def accuracy_of_predicted_values(test_class_names1,l):
    true_count = 0
    false_count = 0
    for i in range(len(test_class_names1)):
        if(test_class_names1[i] == l[i]):
            true_count += 1
        else:
            false_count += 1
    return true_count, false_count, float(true_count) / len(l)

#build a dictionary where the key is the class label and values are the features which belong to that class.
def dict_for_attributes_with_class_values(X,y):
    dict_of_attri_class_values = {}
    fea_list =[]
    for i in range(X.shape[1]):
        fea = i
        l = X[:,i]
        #print(l)
        attribute_list =[]
        count = 0
        for j in l:
            attribute_value = []
            attribute_value.append(j)
            attribute_value.append(y[count])
            attribute_list.append(attribute_value)
            count += 1
        dict_of_attri_class_values[fea]= attribute_list
        fea_list.append(fea)
    return dict_of_attri_class_values,fea_list

def return_features(Y):
    return feature

#Class node and explanation is self explaination
class Node(object):
    def __init__(self, val, lchild, rchild,the,leaf):
    #def __init__(self,val,):
        self.root_value = val
        self.root_left = lchild
        self.root_right = rchild
        self.threshold = the
        self.leaf = leaf

    #method to identify if the node is leaf
    def is_leaf(self):
        return self.leaf

    #method to return threshold value
    def ret_threshold(self):
        return self.threshold

    def ret_root_value(self):
        return self.root_value

    def ret_llist(self):
        return self.root_left

    def ret_rlist(self):
        return self.root_right

    def __repr__(self):
        return "(%r, %r, %r, %r)" %(self.root_value,self.root_left,self.root_right,self.threshold)

#Decision tree object
class DecisionTree(object):

    fea_list = []
    def __init__(self):
        self.root_node = None

    #fit the decisin tree
    def fit(self, dict_of_everything, cl_val, features, eta_min_val):
        global fea_list
        fea_list = features
        root_node = self.recursively_create_decision_tree(dict_of_everything,cl_val,eta_min_val) #,fea_list)
        return root_node

    #method to return the major class value
    def cal_major_class_values(self, class_values):
        data = Counter(class_values)
        data = data.most_common(1)[0][0]
        return data

    #method to calculate best threshold value for each feature
    def cal_best_threshold_value(self,ke,attri_list):
        data = []
        class_values = []
        for i in attri_list:
            data.append(i[0])
            class_values.append(i[1])
        entropy_of_par_attr = entropy(class_values)
        max_info_gain = 0
        threshold = 0
        best_index_left_list = []
        best_index_right_list = []
        class_labels_list_after_split = []
        data.sort()
        for i in range(len(data) - 1):
            cur_threshold = float(data[i]+data[i+1])/ 2
            index_less_than_threshold_list = []
            values_less_than_threshold_list = []
            index_greater_than_threshold_list = []
            values_greater_than_threshold_list = []
            count = 0
            for c,j in enumerate(attri_list):
                if j[0] <= cur_threshold:
                    values_less_than_threshold_list.append(j[1])
                    index_less_than_threshold_list.append(c)
                else:
                    values_greater_than_threshold_list.append(j[1])
                    index_greater_than_threshold_list.append(c)
            entropy_of_less_attribute = entropy(values_less_than_threshold_list)
            entropy_of_greater_attribute = entropy(values_greater_than_threshold_list)

            cur_info_gain = entropy_of_par_attr - (entropy_of_less_attribute*(len(index_less_than_threshold_list)/float(len(attri_list)))) \
                            - (entropy_of_greater_attribute*(len(index_greater_than_threshold_list)/float(len(attri_list))))

            if cur_info_gain > max_info_gain:
                max_info_gain = cur_info_gain
                threshold = cur_threshold
                best_index_left_list = index_less_than_threshold_list
                best_index_right_list = index_greater_than_threshold_list
                class_labels_list_after_split = values_less_than_threshold_list + values_greater_than_threshold_list
        return max_info_gain, threshold, best_index_left_list, best_index_right_list, class_labels_list_after_split


    #method to select the best feature out of all the features.
    def best_feature(self,dict_rep):
        key_value = None
        best_info_gain = -1
        best_threshold = 0
        best_index_left_list = []
        best_index_right_list = []
        best_class_labels_after_split = []
        tmp_list = []
        for ke in dict_rep.keys():
            info_gain, threshold, index_left_list, index_right_list, class_labels_after_split = self.cal_best_threshold_value(ke,dict_rep[ke])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold
                key_value = ke
                best_index_left_list = index_left_list
                best_index_right_list = index_right_list
                best_class_labels_after_split = class_labels_after_split
        tmp_list.append(key_value)
        #tmp_list.append(best_info_gain)
        tmp_list.append(best_threshold)
        tmp_list.append(best_index_left_list)
        tmp_list.append(best_index_right_list)
        tmp_list.append(best_class_labels_after_split)
        return tmp_list

    def get_remainder_dict(self,dict_of_everything,index_split):
        global fea_list
        splited_dict = {}

        for ke in dict_of_everything.keys():
            val_list = []
            modified_list = []
            l = dict_of_everything[ke]
            for i,v in enumerate(l):
                #print(i,v)
                if i not in index_split:
                    #print(ke,i,v)
                    modified_list.append(v)
                    val_list.append(v[1])
            #print(modified_list)
            splited_dict[ke] = modified_list
            key_name = splited_dict.keys()
        return splited_dict,val_list

    #method to create decision tree
    def recursively_create_decision_tree(self, dict_of_everything,class_val,eta_min_val):
        global fea_list
        #if all the class labels are same, then we are set
        if len(set(class_val)) == 1:
            root_node = Node(class_val[0], None, None, 0, True)
            return root_node

        #if the no class vales are less than threshold, we assign the class with max values as the class label
        elif len(class_val) < eta_min_val:
            majority_val = self.cal_major_class_values(class_val)
            root_node = Node(majority_val, None, None, 0, True)
            return root_node

        else:
            best_features_list = self.best_feature(dict_of_everything)
            node_name = best_features_list[0]
            threshold = best_features_list[1]
            index_left_split = best_features_list[2]
            index_right_split = best_features_list[3]
            class_values = best_features_list[4]
            left_dict,class_val1 = self.get_remainder_dict(dict_of_everything,index_left_split)
            right_dict,class_val2 = self.get_remainder_dict(dict_of_everything,index_right_split)
            leftchild = self.recursively_create_decision_tree(left_dict,class_val1,eta_min_val)
            #leftchild = None
            rightchild = self.recursively_create_decision_tree(right_dict,class_val2,eta_min_val)
            root_node = Node(node_name,rightchild,leftchild,threshold,False)
            return root_node

    #method to the labels for the test data
    def predict(self, X, root):
        predicted_list = []
        for row in X:
            y_pred = self.classify(row,root)
            predicted_list.append(y_pred)
        return predicted_list

    def classify(self,row,root):
        dict_test ={}
        for k,j in enumerate(row):
            dict_test[k] = j
        current_node = root
        while not current_node.leaf:
            if dict_test[current_node.root_value] <= current_node.threshold:
                current_node = current_node.root_left
            else:
                current_node = current_node.root_right
        return current_node.root_value

def printTree(tree):
    printNode(tree)

def printNode(node, indent=""):
    # import ipdb; ipdb.set_trace()
    if not node.is_leaf():
        if node.threshold is None:
            #discrete
            for index,child in enumerate(node.children):
                if child.leaf:
                    # print(indent + node.root_value + " = " + self.attributes[index] + " : " + child.root_value)
                    print ("%s %s = %s : %s"%(indent, node.root_value, self.attributes[index], child.root_value))
                else:
                    # print(indent + node.root_value + " = " + self.attributes[index] + " : ")
                    print ("%s %s = %s : "%(indent, node.root_value, self.attributes[index]))
                    printNode(child, indent + "	")
        else:
            #numerical
            leftChild = node.root_left
            rightChild = node.root_right
            if leftChild.is_leaf():
                # import ipdb; ipdb.set_trace()
                # print(indent + node.root_value + " <= " + str(node.threshold) + " : " + leftChild.root_value)
                print ("%s %s <= %s : %s"%(indent, node.root_value , str(node.threshold), leftChild.root_value))

            else:
                # print(indent + node.root_value + " <= " + str(node.threshold)+" : ")
                print ("%s %s <= %s : "%(indent, node.root_value, str(node.threshold)))

                printNode(leftChild, indent + "	")

            if rightChild.is_leaf():
                # print(indent + node.root_value + " > " + str(node.threshold) + " : " + rightChild.root_value)
                print ("%s %s > %s : %s"%(indent, node.root_value, str(node.threshold), leftChild.root_value))

            else:
                # print(indent + node.root_value + " > " + str(node.threshold) + " : ")
                print ("%s %s > %s : "%(indent, node.root_value, str(node.threshold)))

                printNode(rightChild, indent + "	")


def main(num_arr, eta_min):

    eta_min_val = round(eta_min*num_arr.shape[0])

    #randomly shuffle the array so that we can divide the data into test/training
    random_arr1 = random_np_array(num_arr)

    #divide data into test labels,test features,training labels, training features
    test_attri_list,test_class_names_list,training_attri_list,training_class_names_list = generate_set(random_arr1)
    accu_count = 0

    #ten fold iteration
    for i in range(10):

        #build a dictionary with class labels and respective features values belonging to that class
        dict_of_input,fea = dict_for_attributes_with_class_values(training_attri_list[i],training_class_names_list[i])

        #instantiate decision tree instance
        build_dict = DecisionTree()

        # build the decision tree model.
        dec = build_dict.fit(dict_of_input,training_class_names_list[i],fea,eta_min_val)

        #predict the class labels for test features
        l = build_dict.predict(test_attri_list[i], dec)

        #calculate the accuracy for the predicted values
        right,wrong,accu = accuracy_of_predicted_values(test_class_names_list[i], l)
        accu_count += accu
        print("\n Accuracy is:", accu, "\n")
    printTree(dec)

    # import ipdb; ipdb.set_trace()

    print("\n Accuracy across 10-cross validation for:", eta_min, "is", float(accu_count)/10)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        newfile = sys.argv[1]

        #load the data file and do the preprocessing
        num_arr = load_csv(newfile)
        eta_min_list = [0.15, 0.10, 0.15, 0.20]

        #for each threshold value run the classifier for 10 cross-validation
        for i in eta_min_list:
            main(num_arr, i)
