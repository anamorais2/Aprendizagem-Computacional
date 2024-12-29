from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_curve
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
        
    # Normalização
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])
    
    #Correlação
    # Verificar a correlação entre as variáveis
    print("Correlation between variables") 
    corr = df_numerics.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    
    
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

# 5. Treinar a rede neural
def train_neural_network(X,T):
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, T, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam')
    model.fit(Xtrain, ytrain)
    
    return model, Xtest, ytest

def evaluate_model(model, Xtest, ytest):
    #SE,SP,F1score, AUC, ROC, Confusion Matrix, Accuracy
    ypred = model.predict(Xtest)
    cm = confusion_matrix(ytest, ypred)
    TN, FP, FN, TP = cm.ravel()
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    print("Sensitivity: ", SE)
    print("Specificity: ", SP)
    print("F1 Score: ", F1)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    print(cm)
    # Plotar a matriz de confusão
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # Plotar a curva ROC
    yprob = model.predict_proba(Xtest)[:, 1]
    fpr, tpr, _ = roc_curve(ytest, yprob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.show()
    
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


def main():
    
    df_numerics, df_img = load_data()

    X, target = preprocess_neural_network(df_numerics)
    
    # Com base nestes 3 métodos de seleção de features, podemos escolher as features mais importantes, iremos juntar uma combinação destas features
    X_selected, selected_features = select_features(X, target)
    
    X_img = process_img_data(df_img)
    
    X_combined = combine_features(X_selected, X_img)
    
    print(X_combined.shape)
    
    model, Xtest, ytest = train_neural_network(X_combined, target)
    
    evaluate_model(model, Xtest, ytest)
    
    
    
if __name__ == "__main__":
    main()
