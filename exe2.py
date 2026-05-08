import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="AHP Gaussiano - Escolha de Produto",
    layout="wide"
)

st.title("🧠 Método AHP Gaussiano - Decisão de Compra")
st.markdown(
    "Matriz aberta para editar, incluir novos produtos e recalcular automaticamente a melhor opção."
)


# =====================================================
# 1. BASE PADRÃO
# =====================================================
def carregar_dados_padrao():
    df = pd.DataFrame({
        "Produto": ["1", "2", "3", "4"],
        "Marca": [
            "KIT NUTRI",
            "KIT WHEY",
            "WHEY PROTEIN ISOLADO",
            "WHEY PROTEIN ALPEX"
        ],
        "Custo (R$)": ["0.170", "0.224", "0.360", "0.246"],
        "Quantidade (g)": ["0.188", "0.417", "0.208", "0.188"],
        "Benefício Extra": ["1", "0", "0", "0"]
    })

    return df.astype(str)

if "dados_produtos" not in st.session_state:
    st.session_state["dados_produtos"] = carregar_dados_padrao()


# =====================================================
# 2. BOTÕES DE CONTROLE
# =====================================================
col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("🔄 Restaurar matriz padrão"):
        st.session_state["dados_produtos"] = carregar_dados_padrao()
        st.rerun()

with col_b:
    st.info(
        "Você pode editar os dados e adicionar novas linhas diretamente na tabela."
    )


# =====================================================
# 3. MATRIZ DE DECISÃO ABERTA
# =====================================================
st.subheader("📋 Matriz de Decisão")

st.session_state["dados_produtos"] = st.session_state["dados_produtos"].astype(str)

df_editado = st.data_editor(
    st.session_state["dados_produtos"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Produto": st.column_config.TextColumn("Produto"),
        "Marca": st.column_config.TextColumn("Marca", required=True),
        "Custo (R$)": st.column_config.TextColumn("Custo (R$)"),
        "Quantidade (g)": st.column_config.TextColumn("Quantidade (g)"),
        "Benefício Extra": st.column_config.TextColumn("Benefício Extra")
    },
    key="editor_produtos"
)

st.session_state["dados_produtos"] = df_editado.copy()


# =====================================================
# 4. TRATAMENTO DOS DADOS
# =====================================================
def converter_numero(valor):
    """
    Aceita:
    0.170
    0,170
    1.250
    1,250
    """

    if pd.isna(valor):
        return 0

    valor = str(valor).strip()

    if valor == "":
        return 0

    # troca vírgula por ponto
    valor = valor.replace(",", ".")

    try:
        return float(valor)
    except:
        return 0


df = df_editado.copy()

df = df.dropna(subset=["Marca"])

df["Custo (R$)"] = df["Custo (R$)"].apply(converter_numero)
df["Quantidade (g)"] = df["Quantidade (g)"].apply(converter_numero)
df["Benefício Extra"] = df["Benefício Extra"].apply(converter_numero)

if df.empty:
    st.warning("Inclua pelo menos um produto válido na matriz.")
    st.stop()


# =====================================================
# 5. CONFIGURAÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")

criterios = [
    "Custo (R$)",
    "Quantidade (g)",
    "Benefício Extra"
]

col1, col2, col3 = st.columns(3)

with col1:
    tipo_custo = st.selectbox(
        "Custo (R$)",
        ["Menor é melhor", "Maior é melhor"],
        index=0
    )

with col2:
    tipo_quantidade = st.selectbox(
        "Quantidade (g)",
        ["Maior é melhor", "Menor é melhor"],
        index=0
    )

with col3:
    tipo_beneficio = st.selectbox(
        "Benefício Extra",
        ["Maior é melhor", "Menor é melhor"],
        index=0
    )

tipo_criterio = {
    "Custo (R$)": tipo_custo,
    "Quantidade (g)": tipo_quantidade,
    "Benefício Extra": tipo_beneficio
}


# =====================================================
# 6. FUNÇÕES DO AHP GAUSSIANO
# =====================================================
def normalizar_coluna(serie, tipo):
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(
            [1] * len(serie),
            index=serie.index
        )

    if tipo == "Maior é melhor":
        return (serie - minimo) / (maximo - minimo)

    return (maximo - serie) / (maximo - minimo)


def calcular_ahp_gaussiano(df, criterios, tipo_criterio):
    matriz_normalizada = pd.DataFrame(index=df.index)

    for criterio in criterios:
        matriz_normalizada[criterio] = normalizar_coluna(
            df[criterio].astype(float),
            tipo_criterio[criterio]
        )

    desvios_padrao = matriz_normalizada.std(ddof=0)

    if desvios_padrao.sum() == 0:
        pesos = pd.Series(
            [1 / len(criterios)] * len(criterios),
            index=criterios
        )
    else:
        pesos = desvios_padrao / desvios_padrao.sum()

    pontuacao_final = matriz_normalizada.dot(pesos)

    resultado = df.copy()

    resultado["Pontuação AHP Gaussiano"] = pontuacao_final

    resultado["Ranking"] = resultado[
        "Pontuação AHP Gaussiano"
    ].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    resultado = resultado.sort_values(
        by="Pontuação AHP Gaussiano",
        ascending=False
    )

    return matriz_normalizada, pesos, resultado


# =====================================================
# 7. CÁLCULO
# =====================================================
matriz_normalizada, pesos, resultado = calcular_ahp_gaussiano(
    df,
    criterios,
    tipo_criterio
)


# =====================================================
# 8. RESULTADO FINAL
# =====================================================
st.subheader("🏆 Resultado da Decisão")

melhor_produto = resultado.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Produto Recomendado",
    melhor_produto["Marca"]
)

col2.metric(
    "Pontuação",
    f"{melhor_produto['Pontuação AHP Gaussiano']:.4f}"
)

col3.metric(
    "Ranking",
    f"{melhor_produto['Ranking']}º lugar"
)

st.success(
    f"✅ Pelo método AHP Gaussiano, o melhor produto para comprar é: "
    f"**{melhor_produto['Marca']}**."
)


# =====================================================
# 9. RANKING FINAL
# =====================================================
st.subheader("📊 Ranking Final")

resultado_exibicao = resultado.copy()

resultado_exibicao["Custo (R$)"] = resultado_exibicao[
    "Custo (R$)"
].round(3)

resultado_exibicao["Quantidade (g)"] = resultado_exibicao[
    "Quantidade (g)"
].round(3)

resultado_exibicao["Pontuação AHP Gaussiano"] = resultado_exibicao[
    "Pontuação AHP Gaussiano"
].round(4)

st.dataframe(
    resultado_exibicao,
    use_container_width=True
)


# =====================================================
# 10. PESOS
# =====================================================
st.subheader("⚖️ Pesos Calculados")

df_pesos = pd.DataFrame({
    "Critério": pesos.index,
    "Peso": pesos.values.round(4)
})

st.dataframe(
    df_pesos,
    use_container_width=True
)

fig_pesos = px.bar(
    df_pesos,
    x="Critério",
    y="Peso",
    text="Peso",
    title="Pesos dos Critérios pelo AHP Gaussiano"
)

fig_pesos.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_pesos,
    use_container_width=True
)


# =====================================================
# 11. MATRIZ NORMALIZADA
# =====================================================
st.subheader("📐 Matriz Normalizada")

df_normalizada = matriz_normalizada.copy()

df_normalizada.insert(
    0,
    "Marca",
    df["Marca"].values
)

for coluna in criterios:
    df_normalizada[coluna] = df_normalizada[coluna].round(4)

st.dataframe(
    df_normalizada,
    use_container_width=True
)


# =====================================================
# 12. GRÁFICO DO RANKING
# =====================================================
st.subheader("📈 Comparação dos Produtos")

fig_ranking = px.bar(
    resultado,
    x="Marca",
    y="Pontuação AHP Gaussiano",
    text="Pontuação AHP Gaussiano",
    title="Pontuação Final por Produto"
)

fig_ranking.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

st.plotly_chart(
    fig_ranking,
    use_container_width=True
)


# =====================================================
# 13. EXPLICAÇÃO
# =====================================================
with st.expander("🧮 Como o cálculo foi feito"):
    st.markdown("""
    O método AHP Gaussiano calcula os pesos dos critérios
    automaticamente com base na dispersão dos dados.

    Etapas:

    1. Normaliza os critérios

    2. Inverte critérios onde menor valor é melhor

    3. Calcula o desvio-padrão de cada critério

    4. Converte os desvios-padrão em pesos

    5. Calcula a pontuação final de cada produto

    Fórmula dos pesos:

    $$
    peso_j = \\frac{\\sigma_j}{\\sum \\sigma_j}
    $$

    Fórmula da pontuação final:

    $$
    pontuacao_i =
    \\sum normalizado_{ij} \\times peso_j
    $$
    """)