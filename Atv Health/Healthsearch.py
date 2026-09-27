# Aula 4: BM25
# Prática: Laboratório de Parâmetros BM25 (k1  e o)
# Objetivo: Permitir que o aluno altere os parâmetros   

#Novas atividades:
# Envie Multiplos Arquivos
# Vai Ler esses arquivos
# Query Input
# Emitir Resultado

# Busca Hibrida
# Implementar Embendings(Semantico) 
# Reciprocal Rank Fusion (RRF)
#Desenvolver um dashboard analítico interativo em Streamlit contendo sliders de calibração, diagnóstico em abas
#comparativas e métricas de desempenho da busca.


#Junto com copilot

import streamlit as st
import math
import pandas as pd
from pypdf import PdfReader
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Normalização
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# Stopwords
stopwords = {
    "os", "o", "a", "para", "de", "em", "e", "uns", "na", "da",
    "que", "do", "das", "dos", "uma", "por", "pode", "mais",
    "as", "como", "um", "mas", "com", "aos", "ou", "nos"
}

# Stemmer simples
def stemmer_simples(palavra):
    if palavra.endswith("entos"):
        return palavra[:-4]
    if palavra.endswith("amente"):
        return palavra[:-5]
    if palavra.endswith("ados"):
        return palavra[:-3]
    return palavra

# Pipeline de pré-processamento
def preprocess(text):

    # Normalização
    text = normalize(text)

    # Tokenização simples
    tokens = re.findall(r'\w+', text)

    # Remoção de stopwords
    tokens = [t for t in tokens if t not in stopwords]

    # Stemming
    tokens = [stemmer_simples(t) for t in tokens]

    return " ".join(tokens)

st.title("⚙️ Laboratório de Rank BM25")

# Modelos de Embending
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
        )

modelo = carregar_modelo()

#nova base
arquivos = st.file_uploader(
    "Envie um Material",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Consulta (Query):")

st.write(f"**Query:** `{query}`")

# Sliders para os parâmetros
col1, col2, col3 = st.columns(3)
k1 = col1.slider("Parâmetro k1 (Saturação)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
b = col2.slider("Parâmetro b (Tamanho do doc)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
alpha = col3.slider("Peso α (BM25)", min_value=0.0, max_value=1.0, value=0.5,step=0.05)

#Condições de controle de parada
if not arquivos:
    st.info("Envie um ou mais arquivos.")
    st.stop()

if not query:
    st.info("Digite uma consulta.")
    st.stop()

# Geração de Docs:
# Construção da coleção de documentos

docs = {}

for arquivo in arquivos:
    texto = ""
    if arquivo.name.endswith(".txt"):
        texto = arquivo.read().decode("utf-8", errors="ignore")

    elif arquivo.name.endswith(".pdf"):
        reader = PdfReader(arquivo)

        for page in reader.pages:

            page_text = page.extract_text()
            if page_text:
                texto += page_text + " "

    docs[arquivo.name] = preprocess(texto)

query_processada = preprocess(query)

# Cálculos auxiliares
dl = {nome: len(texto.split()) for nome, texto in docs.items()}
avgdl = sum(dl.values()) / len(dl)
N = len(docs)
query = preprocess(query)
df_t = sum(1 for texto in docs.values() if query in texto.split())
idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1) # Fórmula IDF com suavização comum

resultados_bm25 = []
for nome, texto in docs.items():
    f = texto.split().count(query_processada)
    numerador = f * (k1 + 1)
    denominador = f + k1 * (1 - b + b * (dl[nome] / avgdl))
    score = 0

    if denominador > 0:
        score = idf * (numerador / denominador)

    resultados_bm25.append({"Documento": nome, "Tamanho (|D|)": dl[nome], "Score BM25": round(score, 4)})

df_bm25 = pd.DataFrame(resultados_bm25).sort_values(by="Score BM25", ascending=False)

# Busca Semantica
nomes_docs = list(docs.keys())
textos_docs = list(docs.values())

embeddings_docs = modelo.encode(
    textos_docs,
    convert_to_numpy=True
    )

embedding_query = modelo.encode(
    [query_processada],
    convert_to_numpy=True
    )

similaridades = cosine_similarity(
    embedding_query,
    embeddings_docs
    )[0]

resultados_semanticos = []

for nome, score in zip(
    nomes_docs,similaridades
):
    resultados_semanticos.append(
        {
            "Documento": nome,
            "Similaridade Semântica": round(
                float(score), 4
                )
            }
    )

df_semantico = pd.DataFrame(
    resultados_semanticos
    ).sort_values(
        by="Similaridade Semântica",
        ascending=False)


# Ranking dos Documentos
df_bm25["Rank_BM25"] = range(1, len(df_bm25) + 1)
df_semantico["Rank_Semantico"] = range(1, len(df_semantico) + 1)

df_rrf = pd.merge(
    df_bm25[["Documento", "Rank_BM25"]],
    df_semantico[["Documento", "Rank_Semantico"]],
    on="Documento"
)

k_rrf = 60

df_rrf["Score RRF"] = (
    alpha * (1 / (k_rrf + df_rrf["Rank_BM25"])) +
    (1 - alpha) * (1 / (k_rrf + df_rrf["Rank_Semantico"]))
)

df_rrf = df_rrf.sort_values(
    by="Score RRF",
    ascending=False
)

# Arredonda o score
df_rrf["Score RRF"] = df_rrf["Score RRF"].round(6)


#Exibição dos Resultados
st.subheader("Ranking Resultante:")
st.dataframe(df_bm25, use_container_width=True)
st.caption("Mexa nos sliders para ver como documentos longos são penalizados (aumentando b) ou como a repetição de palavras deixa de ter efeito (diminuindo k1).")

st.subheader("Ranking Semântico")
st.dataframe(df_semantico,use_container_width=True)
st.caption(
"BM25 prioriza correspondências exatas de termos. "
"A busca semântica utiliza embeddings e similaridade "
"de cosseno para recuperar documentos relacionados "
"por significado e contexto."
)

st.subheader("Ranking Unificado — RRF")
st.dataframe(
    df_rrf,
    use_container_width=True
)
st.caption(
    f"RRF combina os rankings BM25 e Semântico. "
    f"α = {alpha:.2f} define o peso do BM25, enquanto "
    f"1 - α = {1 - alpha:.2f} define o peso da busca semântica. "
    f"A constante k_RRF = {k_rrf} suaviza a influência da posição."
)

