import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from time import time
import matplotlib.pyplot as plt
import numpy as np

nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, encoding="latin1")
df.columns = ["Noticia", "Classe"]

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(df["Noticia"])
y = df["Classe"]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

x_train_dense = x_train.toarray()
x_test_dense = x_test.toarray()

modelos = [
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("GaussianNB", GaussianNB()),
    ("SVM", SVC(kernel="linear", random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("AdaBoost", AdaBoostClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)),
]

resultados = []


def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000

    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000

    acuracia = accuracy_score(y_test, y_predicted)

    print(f"\nModelo: {name}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_predicted))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_predicted, zero_division=0))
    print("Acurácia:", acuracia)
    print("Tempo de Treinamento (ms):", tempo_treinamento)
    print("Tempo de Previsão (ms):", tempo_previsao)

    # Armazenar os resultados
    resultados.append(
        {
            "Modelo": name,
            "Acurácia": acuracia,
            "Tempo de Treinamento (ms)": tempo_treinamento,
            "Tempo de Previsão (ms)": tempo_previsao,
        }
    )


# Avaliação de todos os modelos
for nome, modelo in modelos:
    if nome in ["QDA", "LDA", "GaussianNB"]:
        mostrar_desempenho(x_train_dense, y_train, x_test_dense, y_test, modelo, nome)
    else:
        mostrar_desempenho(x_train, y_train, x_test, y_test, modelo, nome)

resultados_ordenados = sorted(resultados, key=lambda x: x["Acurácia"], reverse=True)

modelos_nome = [r["Modelo"] for r in resultados_ordenados]
acuracias = [r["Acurácia"] for r in resultados_ordenados]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(modelos_nome, acuracias, color="skyblue")

ax.set_title("Comparação de Desempenho - Acurácia", fontsize=14)
ax.set_xlabel("Modelos")
ax.set_ylabel("Acurácia")

plt.xticks(rotation=45, ha="right")

for bar, acc in zip(bars, acuracias):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{acc:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

import numpy as np

modelos_nome = [r["Modelo"] for r in resultados_ordenados]
tempos_treinamento = [r["Tempo de Treinamento (ms)"] for r in resultados_ordenados]
tempos_previsao = [r["Tempo de Previsão (ms)"] for r in resultados_ordenados]

x = np.arange(len(modelos_nome))  # posições no eixo x
largura_barra = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

b2 = ax.bar(
    x,
    tempos_treinamento,
    width=largura_barra,
    label="Treinamento (ms)",
    color="lightgreen",
)
b3 = ax.bar(
    x + largura_barra,
    tempos_previsao,
    width=largura_barra,
    label="Previsão (ms)",
    color="salmon",
)

ax.set_title("Comparação de Modelos - Tempo de Treinamento e Previsão", fontsize=14)
ax.set_xlabel("Modelos", fontsize=12)
ax.set_ylabel("Valores", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(modelos_nome, rotation=45, ha="right")
ax.legend()


def adicionar_valores(barras, formato=".2f"):
    for bar in barras:
        height = bar.get_height()
        ax.annotate(
            f"{height:{formato}}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # deslocamento
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


adicionar_valores(b2, formato=".0f")
adicionar_valores(b3, formato=".0f")

plt.tight_layout()
plt.show()

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_modelos.csv", index=False)
