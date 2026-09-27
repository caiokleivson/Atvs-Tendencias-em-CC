# =============================================================================
# AgroSearch — Motor de Busca Inteligente (Desafio Integrador - Laboratório 04)
# Disciplina: Tópicos Avançados - Recuperação de Informação / PLN
# UNIPÊ — Prof. Me. Ricardo Roberto de Lima
#
# Este arquivo único integra os 3 pilares exigidos:
#   Fase 1 — Pipeline de pré-processamento (Tokenização, Normalização,
#            Remoção de Stopwords, Stemming)
#   Fase 2 — Índice Invertido (Termo -> [IDs de Documentos])
#   Fase 3 — Busca e Ranqueamento por TF-IDF
#   Bônus  — Similaridade de Cosseno entre a Query e os documentos
#
# RESTRIÇÃO TÉCNICA ATENDIDA: nenhuma biblioteca de alto nível de NLP/ML
# (scikit-learn, TfidfVectorizer, nltk, spaCy etc.) é utilizada. Tokenização,
# stopwords, stemming, índice invertido, TF, IDF, TF-IDF e similaridade de
# cosseno são implementados do zero com re / unicodedata / math / collections.
# O pandas é usado apenas para RENDERIZAR tabelas (st.dataframe), nunca para
# calcular TF-IDF.
# =============================================================================

import re
import math
import unicodedata
from collections import defaultdict

import streamlit as st
import pandas as pd

st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")

# -----------------------------------------------------------------------------
# 0. BASE DE DOCUMENTOS (item 5 do Desafio — hardcoded, conforme sugerido)
# -----------------------------------------------------------------------------
DOCUMENTOS = {
    "Doc 1": "A soja requer irrigação constante durante o período de floração "
             "para garantir a produtividade.",
    "Doc 2": "O controle biológico de lagartas na soja pode ser feito com a "
             "vespa Trichogramma.",
    "Doc 3": "A adubação verde com leguminosas melhora o nitrogênio no solo "
             "para o milho.",
    "Doc 4": "Lagartas desfolhadoras causam grande prejuízo na cultura da "
             "soja e do algodão.",
    "Doc 5": "A irrigação por gotejamento economiza água e é ideal para o "
             "cultivo orgânico.",
}
N_DOCS = len(DOCUMENTOS)

# -----------------------------------------------------------------------------
# 1. FASE 1 — PIPELINE DE PRÉ-PROCESSAMENTO (implementado do zero)
# -----------------------------------------------------------------------------

# Lista própria de stopwords em PT-BR (artigos, preposições, conjunções e
# pronomes de altíssima frequência — ver Aula 2, slide "Stopwords").
STOPWORDS_PT = frozenset({
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sob", "sobre", "entre", "até", "após",
    "ante", "desde", "durante", "perante",
    "e", "ou", "mas", "nem", "que", "se", "como", "quando", "onde",
    "porque", "pois", "porém", "contudo", "todavia",
    "é", "são", "foi", "ser", "estar", "era", "seu", "sua", "seus", "suas",
    "ao", "aos", "à", "às", "este", "esta", "isso", "isto", "aquele",
    "aquela", "lhe", "lhes", "me", "te", "vos", "eu", "tu", "ele", "ela",
    "nós", "vós", "eles", "elas", "não", "sim", "mais", "menos",
    "também", "já", "ainda", "muito", "pode",
})

# Sufixos heurísticos para o stemmer "rudimentar" (do zero), ordenados dos
# mais específicos/longos para os mais genéricos, evitando cortes agressivos
# demais em palavras curtas.
_SUFIXOS_STEM = [
    "izacao", "acao", "ecao",          # nominalização: irrigação -> irrig
    "amente", "mente",                 # advérbios: rapidamente -> rapida
    "idades", "idade",
    "adores", "adoras", "ador", "adora",
    "icos", "icas", "ico", "ica",      # biológico -> biolog
    "osos", "osas", "oso", "osa",
    "ados", "adas", "ado", "ada",
]


def tokenizar(texto: str) -> list[str]:
    """Etapa 1: divide o texto em tokens (sequências alfanuméricas)."""
    return re.findall(r"\b\w+\b", texto)


def normalizar_token(token: str) -> str:
    """Etapa 2: minúsculas + remoção de acentuação (NFD -> ASCII)."""
    t = token.lower()
    t = unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode("utf-8")
    return t


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Etapa 3: remove termos de baixo poder discriminativo."""
    return [t for t in tokens if t not in STOPWORDS_PT]


def stem(token: str) -> str:
    """Etapa 4: stemming heurístico (corte de sufixos), 'do zero'."""
    for suf in _SUFIXOS_STEM:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    if token.endswith("res") and len(token) > 5:
        return token[:-3]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ns"):
        return token[:-1]
    return token


def processar_pipeline(texto: str, usar_stopwords: bool, usar_stemming: bool) -> dict:
    """Executa as 4 etapas e devolve o resultado de CADA etapa (para exibição
    na Fase 1) além da lista final de tokens (usada nas Fases 2 e 3)."""
    tokens = tokenizar(texto)
    normalizados = [normalizar_token(t) for t in tokens]

    sem_stopwords = remover_stopwords(normalizados) if usar_stopwords else list(normalizados)
    finais = [stem(t) for t in sem_stopwords] if usar_stemming else list(sem_stopwords)

    return {
        "tokens": tokens,
        "normalizados": normalizados,
        "sem_stopwords": sem_stopwords,
        "finais": finais,
    }


# -----------------------------------------------------------------------------
# 2. FASE 2 — ÍNDICE INVERTIDO (do zero)
# -----------------------------------------------------------------------------

def construir_indice_invertido(tokens_por_doc: dict[str, list[str]]) -> dict[str, list[str]]:
    """Termo -> lista ordenada de IDs de documentos onde o termo ocorre."""
    indice = defaultdict(set)
    for doc_id, termos in tokens_por_doc.items():
        for termo in termos:
            indice[termo].add(doc_id)
    return {termo: sorted(doc_ids) for termo, doc_ids in sorted(indice.items())}


# -----------------------------------------------------------------------------
# 3. FASE 3 — TF, IDF, TF-IDF e (BÔNUS) SIMILARIDADE DE COSSENO (do zero)
# -----------------------------------------------------------------------------

def calcular_tf(termo: str, tokens_doc: list[str]) -> float:
    """TF(t, d) = ocorrências de t em d / total de termos em d."""
    if not tokens_doc:
        return 0.0
    return tokens_doc.count(termo) / len(tokens_doc)


def calcular_idf(termo: str, tokens_por_doc: dict[str, list[str]], n_docs: int) -> float:
    """IDF(t) = log10(N / df(t)). Usa log base 10 para reproduzir
    exatamente os valores do exemplo da Aula 3 (slides 19-20: 'sistema'
    df=80/N=100 -> IDF ~0.10; 'gato' df=10 -> IDF ~1.00; 'felinofilia'
    df=1 -> IDF ~2.00)."""
    df_t = sum(1 for termos in tokens_por_doc.values() if termo in termos)
    if df_t == 0:
        return 0.0
    return math.log10(n_docs / df_t)


def produto_interno(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def norma(v: list[float]) -> float:
    return math.sqrt(sum(a * a for a in v))


def similaridade_cosseno(v1: list[float], v2: list[float]) -> float:
    n1, n2 = norma(v1), norma(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return produto_interno(v1, v2) / (n1 * n2)


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.title("🌱 AgroSearch — Motor de Busca Inteligente")
st.caption(
    "Protótipo desenvolvido para a AgroTech Solutions — busca textual sobre "
    "manuais técnicos de agricultura sustentável, controle de pragas e irrigação."
)

# --- Sidebar: controles GLOBAIS do pipeline (propagam para as 3 fases) -----
with st.sidebar:
    st.header("⚙️ Configurações do Pipeline")
    st.caption(
        "Os toggles abaixo alteram o vocabulário em **todas as fases** "
        "(Pré-processamento → Índice Invertido → TF-IDF), pois o app usa um "
        "pipeline único e integrado."
    )
    usar_stopwords = st.checkbox("Remover Stopwords", value=True)
    usar_stemming = st.checkbox("Aplicar Stemming", value=True)
    st.divider()
    st.metric("Documentos na base", N_DOCS)
    st.caption("Base de documentos fixa (item 5 do Desafio), não editável.")

# --- Processa TODOS os documentos uma única vez, com as flags atuais -------
resultados_pipeline = {
    doc_id: processar_pipeline(texto, usar_stopwords, usar_stemming)
    for doc_id, texto in DOCUMENTOS.items()
}
tokens_por_doc = {doc_id: r["finais"] for doc_id, r in resultados_pipeline.items()}

tab_docs, tab_fase1, tab_fase2, tab_fase3 = st.tabs(
    ["📄 Base de Documentos", "🧹 Fase 1 — Pré-processamento",
     "🔗 Fase 2 — Índice Invertido", "📊 Fase 3 — Busca & Ranking"]
)

# ------------------------------- TAB: Documentos ----------------------------
with tab_docs:
    st.subheader("Base de Documentos (Estudo de Caso AgroTech)")
    df_docs = pd.DataFrame(
        [{"ID": doc_id, "Texto": texto} for doc_id, texto in DOCUMENTOS.items()]
    )
    st.dataframe(df_docs, use_container_width=True, hide_index=True)

# ------------------------------- TAB: Fase 1 --------------------------------
with tab_fase1:
    st.subheader("Pipeline de Pré-processamento — passo a passo")
    st.markdown(
        f"Configuração atual: **Stopwords {'ativas' if usar_stopwords else 'desativadas'}** · "
        f"**Stemming {'ativo' if usar_stemming else 'desativado'}**"
    )

    total_brutos = sum(len(r["tokens"]) for r in resultados_pipeline.values())
    vocab_bruto = len(set(t for r in resultados_pipeline.values() for t in r["normalizados"]))
    vocab_final = len(set(t for r in resultados_pipeline.values() for t in r["finais"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens brutos (total)", total_brutos)
    c2.metric("Vocabulário após normalização", vocab_bruto)
    c3.metric("Vocabulário final (pós toggles)", vocab_final,
              delta=vocab_final - vocab_bruto)

    st.divider()
    for doc_id, r in resultados_pipeline.items():
        with st.expander(f"{doc_id}: \"{DOCUMENTOS[doc_id]}\""):
            st.write("**1. Tokenização**")
            st.write(r["tokens"])
            st.write("**2. Normalização** (minúsculas + sem acentos)")
            st.write(r["normalizados"])
            st.write(f"**3. Remoção de Stopwords** ({'aplicada' if usar_stopwords else 'ignorada'})")
            st.write(r["sem_stopwords"])
            st.write(f"**4. Stemming** ({'aplicado' if usar_stemming else 'ignorado'})")
            st.write(r["finais"])

# ------------------------------- TAB: Fase 2 --------------------------------
with tab_fase2:
    st.subheader("Índice Invertido (Termo → Documentos)")
    st.caption("Construído a partir dos tokens finais da Fase 1 (respeitando os toggles ativos).")

    indice_invertido = construir_indice_invertido(tokens_por_doc)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        df_indice = pd.DataFrame(
            [{"Termo": termo, "Documentos": ", ".join(docs), "DF (nº docs)": len(docs)}
             for termo, docs in indice_invertido.items()]
        )
        st.dataframe(df_indice, use_container_width=True, hide_index=True, height=420)
    with col_b:
        with st.expander("Ver estrutura bruta (st.json)", expanded=False):
            st.json(indice_invertido)

    st.info(f"📚 Vocabulário indexado: **{len(indice_invertido)}** termos únicos "
            f"distribuídos em **{N_DOCS}** documentos.")

# ------------------------------- TAB: Fase 3 --------------------------------
with tab_fase3:
    st.subheader("Busca e Ranqueamento (TF-IDF) + Bônus: Similaridade de Cosseno")

    query = st.text_input("Digite a consulta (Query):", value="irrigação soja")

    if st.button("🔍 Buscar", type="primary") and query.strip():
        # --- Pré-processa a query com o MESMO pipeline dos documentos -----
        q_result = processar_pipeline(query, usar_stopwords, usar_stemming)
        termos_query = q_result["finais"]
        termos_query_unicos = list(dict.fromkeys(termos_query))  # preserva ordem, remove dup.

        vocabulario = sorted(set(t for tokens in tokens_por_doc.values() for t in tokens))
        termos_fora_vocab = [t for t in termos_query_unicos if t not in vocabulario]

        st.markdown(f"**Termos da query após pipeline:** `{termos_query_unicos}`")
        if termos_fora_vocab:
            st.warning(f"⚠️ Termo(s) fora do vocabulário da coleção (IDF=0, não contribuem "
                       f"para o score): `{termos_fora_vocab}`")

        # --- IDF de cada termo do vocabulário (calculado uma vez) ---------
        idf_vocab = {t: calcular_idf(t, tokens_por_doc, N_DOCS) for t in vocabulario}

        # --- TF-IDF acumulado por documento (soma sobre os termos da query, Fase 3 "oficial") ---
        linhas_detalhe = []
        score_tfidf = {doc_id: 0.0 for doc_id in DOCUMENTOS}
        for termo in termos_query_unicos:
            idf_t = idf_vocab.get(termo, 0.0)
            for doc_id, tokens_doc in tokens_por_doc.items():
                tf_t = calcular_tf(termo, tokens_doc)
                tfidf_t = tf_t * idf_t
                score_tfidf[doc_id] += tfidf_t
                linhas_detalhe.append({
                    "Termo": termo, "Documento": doc_id,
                    "TF": round(tf_t, 4), "IDF": round(idf_t, 4),
                    "TF-IDF": round(tfidf_t, 4),
                })

        # --- BÔNUS: vetores TF-IDF completos (todo o vocabulário) p/ cosseno ---
        vetores_doc = {
            doc_id: [calcular_tf(t, tokens_por_doc[doc_id]) * idf_vocab[t] for t in vocabulario]
            for doc_id in DOCUMENTOS
        }
        vetor_query = [
            calcular_tf(t, termos_query) * idf_vocab[t] if t in idf_vocab else 0.0
            for t in vocabulario
        ]
        score_cosseno = {
            doc_id: similaridade_cosseno(vetor_query, vetores_doc[doc_id])
            for doc_id in DOCUMENTOS
        }

        # --- Tabela final (lado a lado, ordenada por TF-IDF acumulado) ----
        df_resultado = pd.DataFrame([
            {
                "Documento": doc_id,
                "Texto": DOCUMENTOS[doc_id],
                "TF-IDF Acumulado": round(score_tfidf[doc_id], 4),
                "Similaridade de Cosseno": round(score_cosseno[doc_id], 4),
            }
            for doc_id in DOCUMENTOS
        ]).sort_values("TF-IDF Acumulado", ascending=False).reset_index(drop=True)

        st.markdown("### 🏁 Resultado do Ranqueamento")
        styled = df_resultado.style.highlight_max(
            subset=["TF-IDF Acumulado", "Similaridade de Cosseno"], color="#c6f6c6"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        vencedor_tfidf = df_resultado.iloc[0]
        vencedor_cosseno = df_resultado.loc[df_resultado["Similaridade de Cosseno"].idxmax()]

        if vencedor_tfidf["TF-IDF Acumulado"] > 0:
            st.success(
                f"🏆 Documento mais relevante por **TF-IDF acumulado**: "
                f"**{vencedor_tfidf['Documento']}** (score = {vencedor_tfidf['TF-IDF Acumulado']})"
            )
        else:
            st.error("Nenhum termo da query foi encontrado na coleção — score zerado em todos os documentos.")

        if vencedor_cosseno["Similaridade de Cosseno"] > 0:
            st.success(
                f"🎯 Documento mais relevante por **Similaridade de Cosseno**: "
                f"**{vencedor_cosseno['Documento']}** (score = {round(vencedor_cosseno['Similaridade de Cosseno'], 4)})"
            )
            if vencedor_cosseno["Documento"] != vencedor_tfidf["Documento"]:
                st.info(
                    "ℹ️ Os dois rankings **divergiram**: a Similaridade de Cosseno normaliza pelo "
                    "tamanho do vetor do documento inteiro (todo o vocabulário), enquanto o TF-IDF "
                    "acumulado soma apenas os termos da query — por isso ela tende a favorecer "
                    "melhor consultas com múltiplas palavras, como pede o Desafio Bônus."
                )

        with st.expander("🔬 Ver cálculo detalhado (TF, IDF, TF-IDF por termo × documento)"):
            df_detalhe = pd.DataFrame(linhas_detalhe)
            st.dataframe(df_detalhe, use_container_width=True, hide_index=True)

        with st.expander("🧮 Ver vetores completos usados na Similaridade de Cosseno (bônus)"):
            st.caption(f"Vocabulário completo ({len(vocabulario)} termos) usado como dimensões dos vetores.")
            df_vetores = pd.DataFrame(vetores_doc, index=vocabulario).T
            df_vetores.loc["Query"] = vetor_query
            st.dataframe(df_vetores.round(4), use_container_width=True)
    else:
        st.caption("Digite uma consulta e clique em **Buscar** para ver o ranqueamento.")