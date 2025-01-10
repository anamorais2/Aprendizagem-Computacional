from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree




# Função para carregar os dados a partir de um arquivo CSV
def load_data(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro)
    return df

# Função para calcular a entropia
def entropy(data, label):
    label_counts = data[label].value_counts()
    total = len(data)
    ent = 0
    for count in label_counts:
        probability = count / total
        ent -= probability * math.log2(probability)
    return ent

# Função para calcular o ganho de informação
def information_gain(data, split_attribute, label):
    total_entropy = entropy(data, label)
    values = data[split_attribute].unique()
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[split_attribute] == value]
        prob = len(subset) / len(data)
        weighted_entropy += prob * entropy(subset, label)
    
    gain = total_entropy - weighted_entropy
    return gain

# Função para selecionar o melhor atributo
def best_split(data, label):
    attributes = data.columns.drop(label)
    best_gain = 0
    best_attribute = None
    
    for attribute in attributes:
        gain = information_gain(data, attribute, label)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    return best_attribute

#Se houver dois atributos com o mesmo ganho ou /gini, devemos escolher o atributo que nos parece fazer mais sentido, ou seja, o que nos parece mais relevante para a classificação dos dados.


def gini_index(data, label):
    label_counts = data[label].value_counts()
    total = len(data)
    gini = 1.0
    
    for count in label_counts:
        probability = count / total
        gini -= probability ** 2
    
    return gini

def gini_gain(data, split_attribute, label):
    total_gini = gini_index(data, label)
    values = data[split_attribute].unique()
    weighted_gini = 0
    
    for value in values:
        subset = data[data[split_attribute] == value]
        prob = len(subset) / len(data)
        weighted_gini += prob * gini_index(subset, label)
    
    gain = total_gini - weighted_gini
    return gain


def best_split_gini(data, label):
    attributes = data.columns.drop(label)
    best_gain = 0
    best_attribute = None
    
    for attribute in attributes:
        gain = gini_gain(data, attribute, label)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    return best_attribute

# Função para construir a árvore de decisão
def id3(data, label):
    if len(data[label].unique()) == 1:
        return data[label].iloc[0]
    
    if len(data.columns) == 1:
        return data[label].mode()[0]
    
    best_attribute = best_split_gini(data, label)
    tree = {best_attribute: {}}
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value].drop(columns=[best_attribute])
        subtree = id3(subset, label)
        tree[best_attribute][value] = subtree
    
    return tree



if __name__ == "__main__":
    
    data = load_data("apartment.csv")

    
    label = "Acceptable"  
    
    tree = id3(data, label)
    print("Árvore de decisão gerada:", tree)

    # Codificar as variáveis categóricas para valores numéricos
    label_encoder = LabelEncoder()
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Criando um classificador DecisionTreeClassifier para visualização
    clf = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        
    )
    X = data.drop(columns=[label])
    y = data[label]  
    clf.fit(X, y)

    # Plotando a árvore
    plt.figure(figsize=(8, 6))
    plot_tree(clf, feature_names=X.columns, filled=True, rounded=True)
    plt.show()


    y_pred = clf.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculando métricas
    performance = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y, y_pred, pos_label=1)  # 1 representa a classe positiva

    # Exibindo as métricas
    print(f"Performance: {performance:.2f}")
    print(f"Sensibilidade (SE): {sensitivity:.2f}")
    print(f"Especificidade (SP): {specificity:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    print("\n----------------------------------------- 2.1------------------------------------")
    
    data = load_data("P1_cardiacRisk.csv")  

    T = data['Event']  
    X = data.drop(columns=['Event'])  

    Xtrain, Xtest, Ttrain, Ttest = train_test_split(X, T, test_size=0.3, random_state=42) # Dividir o conjunto de dados em treino e teste, porque é importante avaliar o modelo com dados que não foram usados para treinar o modelo. Melhores resultados no teste do que no treino.
    # O objetivo da divisão de dados é conseguir avaliar o modelo com dados que não foram usados para treinar o modelo. Se não for feita a divisão, o modelo pode ter um desempenho muito bom no treino, mas não ter um bom desempenho no teste.

    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=3, #Não ter muita profundidade, por causa do overfitting, para o evitar.
        min_samples_split=4, # O minimo de amostras para dividir um nó, 
        min_samples_leaf=2,
        max_features=None,
        random_state=42,
        max_leaf_nodes=4
    )

    # Treinar o modelo
    model.fit(Xtrain, Ttrain)

    # Fazer previsões no conjunto de treino
    Ytrain = model.predict(Xtrain)

    # Visualizar a árvore de decisão
    plt.figure(figsize=(10, 8))
    plot_tree(model, filled=True, rounded=True, feature_names=X.columns)
    plt.show()

    # Avaliar o modelo
    Ytest = model.predict(Xtest)
    print("Matriz de Confusão:\n", confusion_matrix(Ttest, Ytest))
    print("\nRelatório de Classificação:\n", classification_report(Ttest, Ytest))
    
    #A arvore de decisão, consegue indicar o motivo da decisão. Devido ás regras que são criadas, é possivel perceber o motivo da decisão, o que não é possivel com outros algoritmos de classificação.