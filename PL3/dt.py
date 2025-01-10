import pandas as pd 
import numpy as np


def load_data(nome_ficheiro):
    df  = pd.read_csv(nome_ficheiro)
    return df

def calc_total_entropy(df,label,class_list):
    total_rows = df.shape[0]
    total_entropy = 0
    for c in class_list:
        total_class_count = df[df[label] == c].shape[0]
        total_class_entr = -total_class_count/total_rows * np.log2(total_class_count/total_rows)
        total_entropy += total_class_entr
        
    return total_entropy

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info
    

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unique feature values
    tree = {} #sub-tree or node
    
    for feature_value, count in feature_value_count_dict.items():  # usando items() em vez de iteritems()
        feature_value_data = train_data[train_data[feature_name] == feature_value] # dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking if feature_value is pure class or not
        for c in class_list: # for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] # count of class c

            if class_count == count: # if feature_value is a pure class
                tree[feature_value] = c # add node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] # remove rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: # not a pure class
            tree[feature_value] = "?" # branch marked with "?" to expand further
            
    return tree, train_data



def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset

def id3(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree(tree, None, train_data, label, class_list) #start calling recursion
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None
        
def evaluate(tree, test_data, label):
    correct = 0
    total = len(test_data)

    if total == 0:
        print("O conjunto de dados de teste está vazio.")
        return 0

    for index in range(total):
        try:
            result = predict(tree, test_data.iloc[index])  # prediz a linha
            if result == test_data.iloc[index][label]:  # compara com o rótulo real
                correct += 1
        except IndexError:
            print(f"Índice {index} fora dos limites.")
            continue

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    # Carregar os dados
    df = load_data("apartment.csv")
    
    # Definir o rótulo (label) e classes (as possíveis classes de previsão)
    label = 'Acceptable'  # Substitua pelo nome da coluna que contém o rótulo
    class_list = df[label].unique()  # Lista de classes exclusivas do rótulo

    # Dividir os dados em treino e teste
    train_data = df.sample(frac=0.8, random_state=1)  # 80% para treino
    test_data = df.drop(train_data.index)  # 20% para teste

    # Treinar o modelo ID3
    tree = id3(train_data, label)
    print("Árvore de decisão gerada:", tree)

    # Avaliar a precisão do modelo
    accuracy = evaluate(tree, test_data, label)
    print(f"Precisão do modelo: {accuracy * 100:.2f}%")