import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from time import time
import pickle

# Baixando stopwords
nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

# Lendo o dataset
url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, encoding="latin1")
df.columns = ["Noticia", "Classe"]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(df["Noticia"])
y = df["Classe"]

# Balanceamento com oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Definição do modelo SVM
modelo_svm = SVC(kernel="linear", random_state=42)
inicio_treino = time()
modelo_svm.fit(x_train, y_train)
fim_treino = time()
inicio_pred = time()
y_predicted = modelo_svm.predict(x_test)
fim_pred = time()
acuracia = accuracy_score(y_test, y_predicted)

print("\nModelo: SVM")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_predicted))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_predicted, zero_division=0))
print(f"Acurácia: {acuracia}")
print(f"Tempo de Treinamento (ms): {(fim_treino - inicio_treino) * 1000:.2f}")
print(f"Tempo de Previsão (ms): {(fim_pred - inicio_pred) * 1000:.2f}")

# Salvando o modelo e o vetorizer para futuras predições
with open("modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Modelo e vetorizer salvos!")


# Função para classificar novas notícias
def classificar_nova_noticia(texto):
    with open("modelo_svm.pkl", "rb") as f:
        modelo = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    texto_transformado = vectorizer.transform([texto])
    predicao = modelo.predict(texto_transformado)

    classes = {0: "Ruim", 1: "Boa", 2: "Irrelevante"}
    return classes.get(predicao[0], "Classe desconhecida")


novas_noticias_url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/teste.csv"
novas_noticias = pd.read_csv(novas_noticias_url, header=None)

resultados = []

for noticia in novas_noticias[0]:
    resultado = classificar_nova_noticia(noticia)
    resultados.append(resultado)
    print(f"Notícia: {noticia}\nClassificação: {resultado}\n")

contagem_classes = pd.Series(resultados).value_counts()

plt.figure(figsize=(8, 6))
contagem_classes.plot(kind="bar", color=["red", "green", "gray"])
plt.title("Classificação das Novas Notícias")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
