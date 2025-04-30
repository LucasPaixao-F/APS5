import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Relatório de Desempenho
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from time import time

# Classificador de exemplo
from sklearn.naive_bayes import MultinomialNB

# Carregando o dataset
url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, sep=",", encoding="utf-8")

print("\nMostrando os 5 primeiros registros:")
pd.options.display.max_columns = None
print(df.head())

print("\nInformações do DataFrame:")
df.info()

print("\nDistribuição das classes:")
print(df.iloc[:, -1].value_counts())  # Última coluna como target

# Renomeando colunas para facilitar o uso
df.columns = ["Noticia", "Classe"]

# Vetorização do texto
vectorizer = TfidfVectorizer(stop_words="portuguese")
X = vectorizer.fit_transform(df["Noticia"])
y = df["Classe"]

# Divisão em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Função para exibir desempenho
def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000

    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000

    print(f"\nRelatório Utilizando Algoritmo: {name}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_predicted))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_predicted))
    accuracy = accuracy_score(y_test, y_predicted)
    relatorio = classification_report(y_test, y_predicted, output_dict=True)
    print("Accuracy:", accuracy)
    print("Precision:", relatorio["macro avg"]["precision"])
    print("Tempo de treinamento (ms):", tempo_treinamento)
    print("Tempo de previsão (ms):", tempo_previsao)
    return accuracy, tempo_treinamento, tempo_previsao


# Chamando com Naive Bayes como exemplo
modelo = MultinomialNB()
mostrar_desempenho(x_train, y_train, x_test, y_test, modelo, "Naive Bayes")
