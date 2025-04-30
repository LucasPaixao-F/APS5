# ESTE MODELO UTILIZA O MLPClassifier DO SKLEARN PARA CLASSIFICAR NOTÍCIAS EM TRÊS CATEGORIAS: RUIM, BOA E IRRELEVANTE.

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from time import time
import matplotlib.pyplot as plt

# Baixando stopwords
nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

# Lendo o dataset
url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, encoding="latin1")
df.columns = ["Noticia", "Classe"]

# Visualizando a distribuição das classes
contagem_classes = df["Classe"].value_counts().sort_index()
labels = ["Ruim (0)", "Boa (1)", "Irrelevante (2)"]

plt.figure(figsize=(6, 4))
plt.bar(labels, contagem_classes, color=["red", "green", "gray"])
plt.title("Distribuição das notícias por classe")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(df["Noticia"])
y = df["Classe"]

# Balanceamento com oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Divisão em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)


# Função de avaliação
def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000

    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000

    print(f"\nModelo: {name}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_predicted))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_predicted, zero_division=0))
    print("Acurácia:", accuracy_score(y_test, y_predicted))
    print("Tempo de Treinamento (ms):", tempo_treinamento)
    print("Tempo de Previsão (ms):", tempo_previsao)


# Classificação com MLP
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(100,), max_iter=500, learning_rate="adaptive", random_state=42
)

mostrar_desempenho(x_train, y_train, x_test, y_test, modelo_mlp, "MLPClassifier")
