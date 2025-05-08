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
import matplotlib.pyplot as plt


# Baixando stopwords
nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

# Lendo o dataset de treinamento
url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv"
df = pd.read_csv(url, encoding="latin1")
df.columns = ["Noticia", "Classe"]

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(df["Noticia"])
y = df["Classe"]

# Balanceamento com oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Treinando o modelo SVM
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

# Salvando modelo e vetorizer
with open("modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Modelo e vetorizer salvos!")

novas_noticias_url = "C:/Users/Lucas Paixao/Documents/periodo 5/APS/teste.csv"
df_novas = pd.read_csv(
    novas_noticias_url, parse_dates=["Data"]
)  # <-- precisa ter coluna Data

# Faz previsão
X_novas = vectorizer.transform(df_novas["Noticia"])
y_pred = modelo_svm.predict(X_novas)

# Mapeamento para nomes legíveis
mapeamento = {0: "Ruim", 1: "Boa", 2: "Irrelevante"}
df_novas["Classificacao"] = y_pred
df_novas["Descricao"] = df_novas["Classificacao"].map(mapeamento)

# Exibe as classificações
for _, row in df_novas.iterrows():
    print(
        f"{row['Data'].date()} - {row['Noticia']}\nClassificação: {row['Descricao']}\n"
    )

tabela = df_novas.groupby(["Data", "Descricao"]).size().unstack(fill_value=0)

tabela.plot(kind="bar", stacked=False, figsize=(10, 6), color=["green", "gray", "red"])
plt.title("Classificações de Notícias por Data")
plt.xlabel("Data")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.legend(title="Tipo de Notícia")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

df_novas.to_csv("noticias_classificadas.csv", index=False)

df = pd.read_csv("C:/Users/Lucas Paixao/Documents/periodo 5/APS/teste.csv")

# Confirma que a coluna existe
if "Noticia" not in df.columns:
    raise ValueError("Coluna 'Noticia' não encontrada no arquivo CSV.")

df["Noticia_lower"] = df["Noticia"].str.lower()


def contar_queimadas(texto):
    return texto.count("queimada")


def contar_incendio(texto):
    return texto.count("incêndio")


def contar_fogo(texto):
    return texto.count("fogo")


df["queimadas"] = df["Noticia_lower"].apply(contar_queimadas)
df["incendio"] = df["Noticia_lower"].apply(contar_incendio)
df["fogo"] = df["Noticia_lower"].apply(contar_fogo)

ocorrencias_por_dia = df.groupby("Data")[["queimadas", "incendio", "fogo"]].sum()

ocorrencias_por_dia.plot(kind="bar", figsize=(10, 6), color=["orange", "red", "blue"])
plt.title("Ocorrências de Palavras por Dia")
plt.xlabel("Data")
plt.ylabel("Número de Ocorrências")
plt.xticks(rotation=45)
plt.legend(title="Palavra")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
