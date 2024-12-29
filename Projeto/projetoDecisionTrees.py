import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns

# Importing the dataset
def load_data():
    colunas = ["GENDER", "AGE", "MARITAL STATUS", "VACINATION", "RESPIRATION CLASS", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE", "TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img

def add_rule(X):
    # Adicionar uma regra
    X['RULE'] = ((X["RESPIRATION CLASS"] >= 2) & (X["TEMPERATURE"] > 37.8)).astype(int)
    return X

def process_img_data(df_img):
    # Achatar as imagens (21x21 -> 441)
    X_img = df_img.values.reshape(df_img.shape[0], -1)
    # Normalizar valores binários (0 e 1)
    X_img = X_img / 1.0
    
    return X_img

# Preprocessing the data
def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Função de seleção de features
def feature_selection(X, y):
    
    # Treinar um modelo de Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y) 
    
    # Importância das features
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    # Plotar a importância das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Importância das Features')
    plt.show()
    
    # Selecionar features importantes
    selector = SelectFromModel(rf, threshold=-np.inf, prefit=True)
    X_selected = selector.transform(X)
    
    # Obter os nomes das features selecionadas
    selected_features = X.columns[selector.get_support()]
    
    return selected_features

# Função de seleção de features usando ANOVA
def feature_selection_anova(X, y):
    # Seleção de features usando ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Obter os scores das features
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    
    # Plotar os scores das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Scores das Features usando ANOVA')
    plt.show()
    
    return feature_scores.index

# Função de seleção de features usando Informação Mútua
def feature_selection_mutual_info(X, y):
    # Seleção de features usando Informação Mútua
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Obter os scores das features
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    # Plotar os scores das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Scores das Features usando Informação Mútua')
    plt.show()
    
    return feature_scores.index
    
def save_to_csv(X, filename):
    X.to_csv(filename, index=False)
    print(f"Dados salvos em {filename}")
    
# 4. Concatenar dados tabulares e imagens
def combine_features(X_tabular, X_img):
    # Concatenar os dados tabulares com os dados das imagens
    X_combined = np.hstack((X_tabular, X_img))
    return X_combined

#Escolha de features com base nos 3 modelos de seleção de features
#Selecionar as features mais importantes
def select_features(X, target, num_features=7):
    # Seleção de features usando Random Forest
    selected_features_rf = feature_selection(X, target)

    # Seleção de features usando ANOVA
    selected_features_anova = feature_selection_anova(X, target)

    # Seleção de features usando Informação Mútua
    selected_features_mutual_info = feature_selection_mutual_info(X, target)

    # Criar um ranking global ponderado
    feature_scores = {}

    # Atribuir pontuação baseada na posição em cada lista
    for rank, feature in enumerate(selected_features_rf):
        feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_rf) - rank)

    for rank, feature in enumerate(selected_features_anova):
        feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_anova) - rank)

    for rank, feature in enumerate(selected_features_mutual_info):
        feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_mutual_info) - rank)

    # Ordenar features pelo score acumulado
    sorted_features = sorted(feature_scores.items(), key=lambda x: -x[1])
    top_features = [feature for feature, score in sorted_features[:num_features]]

    # Selecionar o DataFrame com as features escolhidas
    X_selected = X[top_features]

    return X_selected, top_features

def preprocess_decision_tree(df_numerics):
    # Adicionar uma regra (presumindo que a função 'add_rule' já foi definida)
    df_numerics = add_rule(df_numerics)
    
    # Remover duplicados
    df_numerics.drop_duplicates(inplace=True)
    
    # Substituir valores ausentes por média (para colunas numéricas)
    df_numerics.fillna(df_numerics.mean(), inplace=True)
    
    # Análise da correlação
    print("Correlation between variables") 
    corr = df_numerics.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    
    # One-hot encoding para variáveis categóricas
    df_numerics = pd.get_dummies(df_numerics, columns=["GENDER", "MARITAL STATUS", "VACINATION", "RESPIRATION CLASS"], drop_first=True)
    
    # Separar a variável alvo (target) e as variáveis independentes (X)
    target = df_numerics["TARGET"]
    X = df_numerics.drop(columns=["TARGET"])
    
    return X, target


# Training Decision Trees with different parameters
def train_decision_trees(X_train, y_train):
    models = {}
    for criterion in ['gini']:
        for max_depth in [3, 5, 10, None]:
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            models[f'{criterion}_depth_{max_depth}'] = model
    
    # ID3 (similar to entropy)
    id3_model = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
    id3_model.fit(X_train, y_train)
    models['ID3'] = id3_model
    
    # CART (similar to gini)
    cart_model = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
    cart_model.fit(X_train, y_train)
    models['CART'] = cart_model
    
    # Gain Ratio (approximated using entropy and balanced splits)
    gain_ratio_model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42)
    gain_ratio_model.fit(X_train, y_train)
    models['Gain_Ratio'] = gain_ratio_model

    return models

# Evaluating each model
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f'\nEvaluating Model: {name}')
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing each of the trained trees
def visualize_all_trees(models, feature_names):
    for name, model in models.items():
        plt.figure(figsize=(15, 10))
        tree.plot_tree(model, feature_names=feature_names, class_names=['Return Home', 'Stay at Hospital'], filled=True)
        plt.title(f"Visualization of {name}")
        plt.show()

# Main function to run the pipeline
def main():
    df_numerics, df_img = load_data()
    print(df_numerics.head())
    print(df_img.shape)
    
    X, target = preprocess_decision_tree(df_numerics)
    
    # Com base nestes 3 métodos de seleção de features, podemos escolher as features mais importantes, iremos juntar uma combinação destas features
    X_selected, selected_features = select_features(X, target)
    
    X_img = process_img_data(df_img)
    
    X_combined = combine_features(X_selected, X_img)
    
    X_train, X_test, y_train, y_test = preprocess_data(df_numerics)
    models = train_decision_trees(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    
    # Visualize all the trees
    visualize_all_trees(models, df_numerics.columns[:-1])

if __name__ == "__main__":
    main()
