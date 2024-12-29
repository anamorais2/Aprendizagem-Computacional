from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# Load the data
def load_data():
    # Load the COVID_numerics.csv file
    colunas = ["GENDER","AGE","MARITAL STATUS","VACINATION","RESPIRATION CLASS","HEART RATE","SYSTOLIC BLOOD PRESSURE","TEMPERATURE","TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    # Load the COVID_IMG.csv file without header
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img

def add_rule(X):
    # Adicionar uma regra
    X['RULE'] = ((X["RESPIRATION CLASS"] >= 2) & (X["TEMPERATURE"] > 37.8)).astype(int)
    return X

    

def preprocess_neural_network(df_numerics):
    
    # Adicionar uma regra
    df_numerics = add_rule(df_numerics)
    
    # Remover duplicados
    df_numerics.drop_duplicates(inplace=True)
    # Substituir valores ausentes por média
    df_numerics.fillna(df_numerics.mean(), inplace=True)
    # Tratamento de valores ausentes
    """
    for col in df_numerics.columns:
        if df_numerics[col].dtype == 'object':
            df_numerics[col].fillna(df_numerics[col].mode()[0], inplace=True)
        else:
            df_numerics[col].fillna(df_numerics[col].mean(), inplace=True)
            
    """
    continuous_columns = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE"]
    for column in continuous_columns:
        Q1 = df_numerics[column].quantile(0.25)
        Q3 = df_numerics[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_numerics = df_numerics[(df_numerics[column] >= lower_bound) & (df_numerics[column] <= upper_bound)]
        
    # Normalização
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])
    
    
    # One-hot encoding
    df_numerics = pd.get_dummies(df_numerics, columns=["GENDER", "MARITAL STATUS", "VACINATION", "RESPIRATION CLASS"], drop_first=True)
    
    #Separar a variável alvo
    target = df_numerics["TARGET"]
    X = df_numerics.drop(columns=["TARGET"])
    
    return X, target

def process_img_data(df_img):
    # Achatar as imagens (21x21 -> 441)
    X_img = df_img.values.reshape(df_img.shape[0], -1)
    # Normalizar valores binários (0 e 1)
    X_img = X_img / 1.0
    return X_img

# Função de seleção de features
def feature_selection(X, y):
    
    #Correlação
    # Verificar a correlação entre as variáveis
    print("Correlation between variables") 
    corr = X.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    
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
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    X_selected = selector.transform(X)
    
    # Obter os nomes das features selecionadas
    selected_features = X.columns[selector.get_support()]
    
    return X_selected, selected_features

# Função de seleção de features usando ANOVA
def feature_selection_anova(X, y):
    # Seleção de features usando ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Obter os scores das features
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    print("Scores das features usando ANOVA:")
    print(feature_scores)
    
    # Plotar os scores das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Scores das Features usando ANOVA')
    plt.show()
    
    return X_selected, feature_scores.index

# Função de seleção de features usando Informação Mútua
def feature_selection_mutual_info(X, y):
    # Seleção de features usando Informação Mútua
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Obter os scores das features
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    print("Scores das features usando Informação Mútua:")
    print(feature_scores)
    
    # Plotar os scores das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Scores das Features usando Informação Mútua')
    plt.show()
    
    return X_selected, feature_scores.index
    
def save_to_csv(X, filename):
    X.to_csv(filename, index=False)
    print(f"Dados salvos em {filename}")
    
# 4. Concatenar dados tabulares e imagens
def combine_features(X_tabular, X_img):
    # Concatenar os dados tabulares com os dados das imagens
    X_combined = np.hstack((X_tabular, X_img))
    return X_combined

# 5. Treinar a rede neural
def train_neural_network(X_train, y_train):
    # Configurar MLPClassifier
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam')
    model.fit(X_train, y_train)
    return model

def main():
    
    df_numerics, df_img = load_data()
    print("Shape of df_numerics: ", df_numerics.shape)
    print("Shape of df_img: ", df_img.shape)
    
    X, target = preprocess_neural_network(df_numerics)
    
    X_selected, selected_features = feature_selection(X, target)
    
    print("Selected features: ", selected_features)
    print("Shape of X: ", X.shape)
    print("Shape of X_selected: ", X_selected.shape)
    
    #X_selected_anova, selected_features_anova = feature_selection_anova(X, target)
    #print("Features selecionadas usando ANOVA:")
    #print(selected_features_anova)
    
    #X_selected_mutual_info, selected_features_mutual_info = feature_selection_mutual_info(X, target)
    #print("Features selecionadas usando Informação Mútua:")
    #print(selected_features_mutual_info)
    
    # Não iremos usar nem o ANOVA nem a Informação Mútua, uma vez que é bem vísivel pela matriz de Correlação que, por exemplo, a variável "AGE" tem uma correlação de 1 para todas as outras variáveis, o que significa que é uma variável redundante.
    
    
if __name__ == "__main__":
    main()
