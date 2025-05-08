import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

# Baixar stopwords
nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

# === Treinamento do modelo com 3 classes ===
df_treino = pd.read_csv(
    "C:/Users/Lucas Paixao/Documents/periodo 5/APS/noticias_queimadas.csv",
    encoding="latin1",
)
df_treino.columns = ["Noticia", "Classe"]

# Verifica se há as 3 classes
print("Distribuição das classes no dataset de treino:")
print(df_treino["Classe"].value_counts())

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(df_treino["Noticia"])
y = df_treino["Classe"]

# Balanceamento
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Treinamento do modelo
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_resampled, y_resampled)

# === Previsão com novas notícias ===
df_novas = pd.read_csv(
    "CAMINHO/DO/SEU/ARQUIVO.csv", encoding="utf-8"
)  # <-- ajustar caminho
X_novas = vectorizer.transform(df_novas["Noticia"])
y_pred = svm_model.predict(X_novas)

# Mapeamento de classe para texto (opcional)
mapeamento = {0: "Ruim", 1: "Boa", 2: "Irrelevante"}
df_novas["Classificacao"] = y_pred
df_novas["Descricao"] = df_novas["Classificacao"].map(mapeamento)

# Exibir resultado
print(df_novas[["Data", "Noticia", "Classificacao", "Descricao"]])

# Salvar em CSV
df_novas.to_csv("noticias_classificadas.csv", index=False)
print("\n✅ Classificações salvas em 'noticias_classificadas.csv'")
