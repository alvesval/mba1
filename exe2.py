import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(
    page_title="AHP Gaussiano - Escolha de Produto",
    layout="wide"
)

st.title("🧠 Método AHP Gaussiano - Decisão de Compra")
st.markdown("Sistema para escolher o melhor produto com base em múltiplos critérios.")


# =====================================================
# 1. BASE INICIAL
# =====================================================
dados_iniciais = pd.DataFrame({
    "Produto": [1, 2, 3, 4],
    "Marca": [
        "KIT NUTRI",
        "KIT WHEY",
        "WHEY PROTEIN ISOLADO",
        "WHEY PROTEIN ALPEX"
    ],
    "Custo (R$)": [0.170, 0.224, 0.360, 0.246],
    "Quantidade (g)": [0.188, 0.417, 0.208, 0.188],
    "Benefício Extra": [1, 0, 0, 0]
})


st.subheader("📋 Matriz de Decisão")

df = st.data_editor(
    dados_iniciais,
    num_rows="dynamic",
    use_container_width=True
)


# =====================================================
# 2. CONFIGURAÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")

criterios = ["Custo (R$)", "Quantidade (g)", "Benefício Extra"]

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
# 3. FUNÇÕES DO AHP GAUSSIANO
# =====================================================
def normalizar_coluna(serie, tipo):
    """
    Normalização Min-Max.
    Para critério de benefício: maior é melhor.
    Para critério de custo: menor é melhor.
    """
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series([1] * len(serie), index=serie.index)

    if tipo == "Maior é melhor":
        return (serie - minimo) / (maximo - minimo)
    else:
        return (maximo - serie) / (maximo - minimo)


def calcular_ahp_gaussiano(df, criterios, tipo_criterio):
    matriz_normalizada = pd.DataFrame()

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
    resultado["Ranking"] = resultado["Pontuação AHP Gaussiano"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    resultado = resultado.sort_values(
        by="Pontuação AHP Gaussiano",
        ascending=False
    )

    return matriz_normalizada, pesos, resultado


# =====================================================
# 4. CÁLCULO
# =====================================================
matriz_normalizada, pesos, resultado = calcular_ahp_gaussiano(
    df,
    criterios,
    tipo_criterio
)


# =====================================================
# 5. RESULTADO FINAL
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
# 6. TABELA FINAL
# =====================================================
st.subheader("📊 Ranking Final")

st.dataframe(
    resultado.style.format({
        "Custo (R$)": "{:.3f}",
        "Quantidade (g)": "{:.3f}",
        "Pontuação AHP Gaussiano": "{:.4f}"
    }),
    use_container_width=True
)


# =====================================================
# 7. PESOS CALCULADOS
# =====================================================
st.subheader("⚖️ Pesos Calculados pelo AHP Gaussiano")

df_pesos = pd.DataFrame({
    "Critério": pesos.index,
    "Peso": pesos.values
})

st.dataframe(
    df_pesos.style.format({"Peso": "{:.4f}"}),
    use_container_width=True
)

fig_pesos = px.bar(
    df_pesos,
    x="Critério",
    y="Peso",
    text="Peso",
    title="Peso dos Critérios pelo Desvio-Padrão Gaussiano"
)

fig_pesos.update_traces(texttemplate="%{text:.4f}", textposition="outside")

st.plotly_chart(fig_pesos, use_container_width=True)


# =====================================================
# 8. MATRIZ NORMALIZADA
# =====================================================
st.subheader("📐 Matriz Normalizada")

df_normalizada = matriz_normalizada.copy()
df_normalizada.insert(0, "Marca", df["Marca"])

st.dataframe(
    df_normalizada.style.format({
        "Custo (R$)": "{:.4f}",
        "Quantidade (g)": "{:.4f}",
        "Benefício Extra": "{:.4f}"
    }),
    use_container_width=True
)


# =====================================================
# 9. GRÁFICO DO RANKING
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

st.plotly_chart(fig_ranking, use_container_width=True)


# =====================================================
# 10. EXPLICAÇÃO DO MÉTODO
# =====================================================
with st.expander("🧮 Como o cálculo foi feito"):
    st.markdown("""
    O método AHP Gaussiano calcula os pesos dos critérios automaticamente com base
    na dispersão dos dados.

    Etapas:

    1. Normaliza os critérios.
    2. Inverte critérios onde menor valor é melhor, como custo.
    3. Calcula o desvio-padrão de cada critério.
    4. Transforma os desvios-padrão em pesos.
    5. Calcula a pontuação final de cada produto.

    Fórmula dos pesos:

    $$
    peso_j = \\frac{\\sigma_j}{\\sum \\sigma_j}
    $$

    Fórmula da pontuação final:

    $$
    pontuacao_i = \\sum normalizado_{ij} \\times peso_j
    $$
    """)