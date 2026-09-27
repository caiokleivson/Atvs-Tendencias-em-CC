# -*- coding: utf-8 -*-
"""
app_ouvidoria.py
Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
Entrega 4 do desafio prático de NLP Aplicado.

Equipe: Caio Kleivson

Execução:
    streamlit run app_ouvidoria.py

Requisitos (ver requirements.txt):
    streamlit, pandas, numpy, matplotlib, scikit-learn,
    sentence-transformers, langchain-text-splitters
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------------------
# Configuração geral da página
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

DATA_PATH = Path(__file__).parent / "manifestacoes.json"

STOPWORDS_PT = [
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no",
    "na", "nos", "nas", "por", "para", "com", "sem", "e", "ou", "que", "se", "ao", "aos", "à",
    "às", "é", "foi", "ser", "está", "estão", "já", "não", "mais", "muito", "até", "como",
    "também", "seu", "sua", "seus", "suas", "este", "esta", "isso", "essa", "esse", "há", "tem",
    "têm", "desde", "entre", "sobre", "após", "cerca", "bem", "assim", "pelo", "pela", "pelos",
    "pelas",
]

MODELOS_EMBEDDING = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-pt-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]

CORES_CATEGORIA = {
    "infraestrutura": "#2563eb",
    "saúde": "#dc2626",
    "segurança": "#7c3aed",
    "educação": "#16a34a",
    "meio ambiente": "#059669",
}


# --------------------------------------------------------------------------------------
# Carregamento de dados e modelos (cacheados conforme sugerido no enunciado)
# --------------------------------------------------------------------------------------
@st.cache_data
def carregar_manifestacoes():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)


@st.cache_resource(show_spinner=False)
def carregar_modelo_embedding(nome_modelo):
    """Carrega um modelo Sentence-Transformers. Retorna None se indisponível
    (ex.: sem acesso à internet para baixar o modelo pela primeira vez)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(nome_modelo)
    except Exception as e:  # noqa: BLE001
        st.session_state["_erro_modelo"] = f"{type(e).__name__}: {e}"
        return None


@st.cache_data(show_spinner=False)
def gerar_embeddings(nome_modelo, textos):
    modelo = carregar_modelo_embedding(nome_modelo)
    if modelo is None:
        return None
    return modelo.encode(list(textos), show_progress_bar=False)


@st.cache_data(show_spinner=False)
def gerar_tfidf(textos):
    vect = TfidfVectorizer(stop_words=STOPWORDS_PT)
    X = vect.fit_transform(textos)
    return X.toarray(), vect


def obter_representacao(df, nome_modelo, usar_embeddings=True):
    """Retorna (matriz, rótulo, embeddings_ok)."""
    if usar_embeddings:
        emb = gerar_embeddings(nome_modelo, tuple(df["texto"].tolist()))
        if emb is not None:
            return emb, f"Embeddings ({nome_modelo})", True
    X_tfidf, _ = gerar_tfidf(df["texto"].tolist())
    return X_tfidf, "TF-IDF (fallback — embeddings indisponíveis)", False


def cor_score(score):
    if score > 0.7:
        return "🟢"
    elif score > 0.5:
        return "🟡"
    return "🔴"


# --------------------------------------------------------------------------------------
# Sidebar — configurações globais
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações")

modelo_escolhido = st.sidebar.selectbox("Modelo de embedding", MODELOS_EMBEDDING, index=0)
usar_embeddings = st.sidebar.checkbox(
    "Usar embeddings (desmarque para forçar TF-IDF)", value=True,
    help="Se o download do modelo falhar (ex.: sem internet), o app usa TF-IDF automaticamente.",
)
top_k = st.sidebar.slider("Top-K resultados na busca semântica", min_value=1, max_value=10, value=5)
st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Modelos multilíngues (ex.: paraphrase-multilingual-MiniLM-L12-v2, BAAI/bge-small-pt-v1.5) "
    "tendem a capturar melhor o português coloquial das manifestações."
)

df = carregar_manifestacoes()

st.title("🏛️ Ouvidoria Inteligente — Triagem Semântica de Manifestações")
st.caption(
    "Protótipo de sistema de triagem semântica para manifestações cidadãs, "
    "substituindo a busca por palavra-chave por representações vetoriais."
)

X_repr, rotulo_repr, embeddings_ok = obter_representacao(df, modelo_escolhido, usar_embeddings)
if not embeddings_ok and usar_embeddings:
    st.warning(
        "⚠️ Não foi possível carregar o modelo de embeddings selecionado "
        f"(provável falta de acesso à internet para baixar '{modelo_escolhido}'). "
        "O app está operando com **TF-IDF** como representação de fallback. "
        "Ao rodar este app em um ambiente com internet liberada, os embeddings serão usados normalmente."
    )
st.sidebar.info(f"Representação ativa: **{rotulo_repr}**")

tab_busca, tab_base, tab_espaco, tab_chunking = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)

# --------------------------------------------------------------------------------------
# Aba 1 — Busca Semântica
# --------------------------------------------------------------------------------------
with tab_busca:
    st.subheader("Busca Semântica de Manifestações")
    st.write(
        "Digite uma descrição livre do problema (como um cidadão faria) para encontrar "
        "as manifestações mais similares já registradas."
    )

    query = st.text_input(
        "Descreva o problema:",
        value="buraco grande na rua causando acidentes",
        placeholder="Ex.: falta de médico no posto de saúde do bairro",
    )

    if st.button("🚀 Buscar", type="primary"):
        if not query.strip():
            st.warning("Digite uma descrição para buscar.")
        else:
            if embeddings_ok:
                modelo = carregar_modelo_embedding(modelo_escolhido)
                q_vec = modelo.encode([query], show_progress_bar=False)
            else:
                _, vect = gerar_tfidf(df["texto"].tolist())
                q_vec = vect.transform([query]).toarray()

            sims = cosine_similarity(q_vec, X_repr)[0]
            ranking = np.argsort(sims)[::-1][:top_k]

            st.markdown(f"### 🏆 Top {top_k} manifestações mais similares")
            for pos, idx in enumerate(ranking, 1):
                score = float(sims[idx])
                row = df.iloc[idx]
                emoji = cor_score(score)
                with st.expander(
                    f"{emoji} #{pos} — {row['id']} | score: {score:.3f} | "
                    f"categoria: {row['categoria_oficial']}"
                ):
                    st.write(f"**Data:** {row['data']}")
                    st.write(f"**Texto:** {row['texto']}")

    st.markdown("---")
    st.caption(
        "Legenda de score: 🟢 acima de 0.7 (alta similaridade) · "
        "🟡 acima de 0.5 (similaridade moderada) · 🔴 demais (baixa similaridade)."
    )

# --------------------------------------------------------------------------------------
# Aba 2 — Base Completa
# --------------------------------------------------------------------------------------
with tab_base:
    st.subheader("Base Completa de Manifestações")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filtro_categoria = st.multiselect(
            "Filtrar por categoria", sorted(df["categoria_oficial"].unique()),
            default=sorted(df["categoria_oficial"].unique()),
        )
    with col_f2:
        busca_texto = st.text_input("Filtrar por palavra no texto", "")

    df_filtrado = df[df["categoria_oficial"].isin(filtro_categoria)]
    if busca_texto.strip():
        df_filtrado = df_filtrado[df_filtrado["texto"].str.contains(busca_texto, case=False, na=False)]

    st.dataframe(df_filtrado, use_container_width=True, height=400)
    st.caption(f"Exibindo {len(df_filtrado)} de {len(df)} manifestações.")

    if st.button("🔢 Gerar matriz de similaridade completa"):
        with st.spinner("Calculando similaridades..."):
            sim_matrix = cosine_similarity(X_repr)
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(df)))
        ax.set_yticks(range(len(df)))
        ax.set_xticklabels(df["id"], rotation=90, fontsize=6)
        ax.set_yticklabels(df["id"], fontsize=6)
        plt.colorbar(im, ax=ax, label="similaridade de cosseno")
        plt.title(f"Matriz de Similaridade — {rotulo_repr}")
        plt.tight_layout()
        st.pyplot(fig)

        limiar = st.slider(
            "Limiar para destacar pares duplicados candidatos", 0.5, 0.99, 0.85, 0.01,
            key="limiar_base",
        )
        pares = []
        n = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= limiar:
                    pares.append(
                        {
                            "manifestacao_1": df.iloc[i]["id"],
                            "manifestacao_2": df.iloc[j]["id"],
                            "similaridade": round(float(sim_matrix[i, j]), 3),
                        }
                    )
        st.markdown(f"**Pares candidatos a duplicata (similaridade ≥ {limiar:.2f}):** {len(pares)}")
        if pares:
            st.dataframe(pd.DataFrame(pares).sort_values("similaridade", ascending=False),
                         use_container_width=True)

# --------------------------------------------------------------------------------------
# Aba 3 — Espaço Vetorial
# --------------------------------------------------------------------------------------
with tab_espaco:
    st.subheader("Visualização do Espaço Vetorial")
    metodo = st.radio("Método de redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)

    if st.button("🌐 Gerar visualização"):
        with st.spinner(f"Reduzindo dimensionalidade com {metodo}..."):
            if metodo == "PCA":
                coords = PCA(n_components=2, random_state=42).fit_transform(X_repr)
            else:
                perp = min(30, max(2, len(df) - 1))
                coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_repr)

        fig, ax = plt.subplots(figsize=(10, 7))
        for categoria, cor in CORES_CATEGORIA.items():
            mask = df["categoria_oficial"] == categoria
            ax.scatter(
                coords[mask.values, 0], coords[mask.values, 1],
                label=categoria, color=cor, s=90, edgecolor="black", alpha=0.85,
            )
        for k, (x, y) in enumerate(coords):
            ax.annotate(df.iloc[k]["id"], (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        ax.set_title(f"Espaço Semântico das Manifestações — {metodo} ({rotulo_repr})")
        ax.legend(title="Categoria oficial")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown(
            """
            ##### 🔎 Os clusters semânticos coincidem com as categorias oficiais?

            *Observação a preencher pelo aluno após executar com o ambiente completo (embeddings
            reais):* em geral, espera-se que manifestações de **infraestrutura** (buracos,
            iluminação, calçadas) e **meio ambiente** (lixo, poluição, esgoto) formem clusters
            razoavelmente distintos, já que usam vocabulário bem característico. Já **saúde** e
            **educação** podem apresentar alguma sobreposição pontual quando os textos mencionam
            estruturas físicas (ex.: "telhado da escola com goteira" vs. "telhado do posto de
            saúde"), pois a semântica de "problema estrutural em prédio público" se aproxima,
            mesmo pertencendo a categorias oficiais diferentes. Isso ilustra uma limitação
            importante: a categoria oficial reflete o *setor responsável* pela solução, enquanto o
            embedding captura *similaridade de conteúdo textual* — os dois nem sempre coincidem
            perfeitamente, o que é justamente um dos motivos para a Ouvidoria usar triagem
            semântica como apoio (não substituto) à classificação manual.
            """
        )

# --------------------------------------------------------------------------------------
# Aba 4 — Chunking
# --------------------------------------------------------------------------------------
with tab_chunking:
    st.subheader("Chunking de Manifestações Longas")
    st.write(
        "Cole uma manifestação longa (ou selecione uma já existente na base) para visualizar "
        "como ela seria dividida em chunks antes de ser indexada em um banco vetorial."
    )

    origem = st.radio("Origem do texto", ["Selecionar da base", "Colar texto livre"], horizontal=True)

    if origem == "Selecionar da base":
        df_longas = df[df["texto"].str.len() > 400].sort_values(
            "texto", key=lambda s: s.str.len(), ascending=False
        )
        opcao = st.selectbox(
            "Manifestação", df_longas["id"].tolist(),
            format_func=lambda i: f"{i} ({df.loc[df['id'] == i, 'texto'].values[0][:40]}...)",
        )
        texto_alvo = df.loc[df["id"] == opcao, "texto"].values[0]
    else:
        texto_alvo = st.text_area("Cole o texto da manifestação:", height=200)

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("chunk_size", 50, 800, 300, 10)
    with col2:
        chunk_overlap = st.slider("chunk_overlap", 0, 300, 60, 10)

    estrategia = st.selectbox("Estratégia de chunking", ["RecursiveCharacterTextSplitter"])

    if st.button("🧩 Gerar chunks"):
        if not texto_alvo or not texto_alvo.strip():
            st.warning("Selecione ou cole um texto para gerar os chunks.")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_text(texto_alvo)
            st.success(f"{len(chunks)} chunks gerados.")

            for i, c in enumerate(chunks):
                with st.expander(f"Chunk {i + 1} ({len(c)} caracteres)"):
                    st.write(c)

            if len(chunks) > 1:
                if embeddings_ok:
                    modelo = carregar_modelo_embedding(modelo_escolhido)
                    emb_chunks = modelo.encode(chunks, show_progress_bar=False)
                    rotulo_chunk_repr = f"Embeddings ({modelo_escolhido})"
                else:
                    vect_local = TfidfVectorizer()
                    emb_chunks = vect_local.fit_transform(chunks).toarray()
                    rotulo_chunk_repr = "TF-IDF (fallback)"

                st.markdown(f"**Dimensão dos embeddings dos chunks ({rotulo_chunk_repr}):** {emb_chunks.shape}")

                sim_chunks = cosine_similarity(emb_chunks)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(sim_chunks, cmap="Purples", vmin=0, vmax=1)
                ax.set_xticks(range(len(chunks)))
                ax.set_yticks(range(len(chunks)))
                ax.set_xticklabels([f"C{i+1}" for i in range(len(chunks))])
                ax.set_yticklabels([f"C{i+1}" for i in range(len(chunks))])
                plt.colorbar(im, ax=ax)
                plt.title("Similaridade entre chunks gerados")
                st.pyplot(fig)
            else:
                st.info("Apenas um chunk foi gerado — aumente o texto ou reduza o chunk_size para comparar.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para o desafio prático de NLP Aplicado — Ouvidoria Inteligente.")
