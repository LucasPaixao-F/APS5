import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Relatório de Desempenho
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time

url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, sep=",", encoding="utf-8")
print("\nMostrandoos5primeirosregistros:")
pd.options.display.max_columns = None
print(df.head(5))
print("\nMostrandoasinformaçõesdoDataFrame:")
df.info()
print("\nMostrandoLabels:")
print(df.Outcome.value_counts())
print("\nFiltrandodadosdoDataFrame:")
print(df.query("Outcome == 1 and Glucose > 150 and Age > 33 and Insulin > 79").head())

# Mostrando correlação entre variáveis
df_correlacao = df.corr()
sns.heatmap(df_correlacao, cmap="RdBu", fmt=".2f", square=True, annot=True)

# Separando os dados de teste (30%) e de treino (70%
# Retirando as 4 colunas menos importantes

x_data = df.drop(
    ["Outcome", "Insulin", "Pregnancies", "SkinThickness", "BloodPressure"],
    axis=1,
    inplace=False,
)
y_data = df["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42
)
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    # Treinando modelo
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000
    # Prevendo dados
    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000
    print("Relatório Utilizando Algoritmo", name)
    print("\nMostrando Matriz de Confusão:")
    print(confusion_matrix(y_test, y_predicted))
    print("\nMostrando Relatório de Classificação:")
    print(metrics.classification_report(y_test, y_predicted))
    accuracy = accuracy_score(y_test, y_predicted)
    print("Accuracy:", accuracy)
    relatorio = metrics.classification_report(y_test, y_predicted, output_dict=True)
    print("Precision:", relatorio["macro avg"]["precision"])
    print("Tempo de treinamento (ms):", tempo_treinamento)
    print("Tempo de previsão (ms):", tempo_previsao)
    return accuracy, tempo_treinamento, tempo_previsao
