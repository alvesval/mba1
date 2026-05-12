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
    """
    Esta aplicação utiliza o **AHP Gaussiano** para apoiar a escolha de produtos.
    Os pesos dos critérios são calculados automaticamente a partir da dispersão dos dados,
    reduzindo a necessidade de julgamentos par a par.
    """
)

# =====================================================
# 1. BASE PADRÃO
# =====================================================
def carregar_dados_padrao():
    return pd.DataFrame({
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
    }).astype(str)

if "dados_produtos" not in st.session_state:
    st.session_state["dados_produtos"] = carregar_dados_padrao()

# =====================================================
# 2. CONTROLES
# =====================================================
col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("🔄 Restaurar matriz padrão"):
        st.session_state["dados_produtos"] = carregar_dados_padrao()
        st.rerun()

with col_b:
    st.info(
        "Edite os dados, inclua novos produtos e ajuste a direção de preferência de cada critério."
    )

# =====================================================
# 3. MATRIZ DE DECISÃO
# =====================================================
st.subheader("📋 Matriz de Decisão")

df_editado = st.data_editor(
    st.session_state["dados_produtos"].astype(str),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Produto": st.column_config.TextColumn("Produto"),
        "Marca": st.column_config.TextColumn("Marca", required=True),
        "Custo (R$)": st.column_config.TextColumn("Custo (R$)"),
        "Quantidade (g)": st.column_config.TextColumn("Quantidade (g)"),
        "Benefício Extra": st.column_config.TextColumn("Benefício Extra"),
    },
    key="editor_produtos"
)

st.session_state["dados_produtos"] = df_editado.copy()

# =====================================================
# 4. TRATAMENTO DOS DADOS
# =====================================================
def converter_numero(valor):
    """
    Converte valores digitados como:
    0.170, 0,170, 1.250, 1,250.
    Observação: quando houver pontos e vírgulas misturados, assume padrão brasileiro.
    """
    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip()
    if valor == "":
        return np.nan

    # Ex.: 1.250,50 -> 1250.50
    if "." in valor and "," in valor:
        valor = valor.replace(".", "").replace(",", ".")
    else:
        valor = valor.replace(",", ".")

    try:
        return float(valor)
    except ValueError:
        return np.nan


df = df_editado.copy()
df = df.dropna(subset=["Marca"])
df["Marca"] = df["Marca"].astype(str).str.strip()
df = df[df["Marca"] != ""]

criterios = ["Custo (R$)", "Quantidade (g)", "Benefício Extra"]

for criterio in criterios:
    df[criterio] = df[criterio].apply(converter_numero)

linhas_invalidas = df[df[criterios].isna().any(axis=1)]

if not linhas_invalidas.empty:
    st.warning(
        "Algumas linhas possuem valores numéricos vazios ou inválidos e serão ignoradas no cálculo."
    )
    st.dataframe(linhas_invalidas[["Produto", "Marca"] + criterios], use_container_width=True)

df = df.dropna(subset=criterios)

if df.empty:
    st.warning("Inclua pelo menos um produto válido na matriz.")
    st.stop()

if len(df) < 2:
    st.warning("Inclua pelo menos dois produtos para que o AHP Gaussiano consiga comparar alternativas.")
    st.stop()

# =====================================================
# 5. CONFIGURAÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")

col1, col2, col3 = st.columns(3)

with col1:
    tipo_custo = st.selectbox(
        "Custo (R$)",
        ["Menor é melhor", "Maior é melhor"],
        index=0,
        help="Use 'Menor é melhor' para preço, custo, tempo ou consumo."
    )

with col2:
    tipo_quantidade = st.selectbox(
        "Quantidade (g)",
        ["Maior é melhor", "Menor é melhor"],
        index=0,
        help="Use 'Maior é melhor' quando mais quantidade representa vantagem."
    )

with col3:
    tipo_beneficio = st.selectbox(
        "Benefício Extra",
        ["Maior é melhor", "Menor é melhor"],
        index=0,
        help="Pode representar brinde, dose extra, qualidade percebida ou outro benefício."
    )

tipo_criterio = {
    "Custo (R$)": tipo_custo,
    "Quantidade (g)": tipo_quantidade,
    "Benefício Extra": tipo_beneficio,
}

# =====================================================
# 6. FUNÇÕES DO AHP GAUSSIANO
# =====================================================
def normalizar_criterio_soma(serie, tipo):
    """
    Normalização por soma, compatível com a lógica usada em aplicações AHP:
    - Maior é melhor: valor / soma dos valores
    - Menor é melhor: inverso(valor) / soma dos inversos

    Para valores zero em critério de custo, aplica deslocamento pequeno para evitar divisão por zero.
    """
    serie = pd.to_numeric(serie, errors="coerce").astype(float)

    if tipo == "Maior é melhor":
        valores = serie.clip(lower=0)
        soma = valores.sum()
        if soma == 0:
            return pd.Series([1 / len(serie)] * len(serie), index=serie.index)
        return valores / soma

    # Menor é melhor
    minimo_positivo = serie[serie > 0].min()
    if pd.isna(minimo_positivo):
        return pd.Series([1 / len(serie)] * len(serie), index=serie.index)

    serie_ajustada = serie.copy()
    serie_ajustada[serie_ajustada <= 0] = minimo_positivo * 0.01

    inverso = 1 / serie_ajustada
    return inverso / inverso.sum()


def calcular_ahp_gaussiano(df, criterios, tipo_criterio):
    matriz_normalizada = pd.DataFrame(index=df.index)

    for criterio in criterios:
        matriz_normalizada[criterio] = normalizar_criterio_soma(
            df[criterio],
            tipo_criterio[criterio]
        )

    medias = matriz_normalizada.mean()
    desvios_padrao = matriz_normalizada.std(ddof=0)

    fator_gaussiano = desvios_padrao / medias.replace(0, np.nan)
    fator_gaussiano = fator_gaussiano.fillna(0)

    if fator_gaussiano.sum() == 0:
        pesos = pd.Series([1 / len(criterios)] * len(criterios), index=criterios)
    else:
        pesos = fator_gaussiano / fator_gaussiano.sum()

    pontuacao_final = matriz_normalizada.dot(pesos)

    resultado = df.copy()
    resultado["Pontuação AHP Gaussiano"] = pontuacao_final
    resultado["Ranking"] = (
        resultado["Pontuação AHP Gaussiano"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    contribuicoes = matriz_normalizada.mul(pesos, axis=1)
    contribuicoes.insert(0, "Marca", df["Marca"].values)

    resultado = resultado.sort_values(
        by="Pontuação AHP Gaussiano",
        ascending=False
    )

    diagnostico = pd.DataFrame({
        "Critério": criterios,
        "Tipo": [tipo_criterio[c] for c in criterios],
        "Média normalizada": medias.values,
        "Desvio-padrão": desvios_padrao.values,
        "Fator Gaussiano": fator_gaussiano.values,
        "Peso": pesos.values,
    }).sort_values("Peso", ascending=False)

    return matriz_normalizada, pesos, resultado, diagnostico, contribuicoes


matriz_normalizada, pesos, resultado, diagnostico, contribuicoes = calcular_ahp_gaussiano(
    df, criterios, tipo_criterio
)

# =====================================================
# 7. RESULTADO FINAL
# =====================================================
st.subheader("🏆 Resultado da Decisão")

melhor_produto = resultado.iloc[0]
segundo_produto = resultado.iloc[1] if len(resultado) > 1 else None

col1, col2, col3, col4 = st.columns(4)

col1.metric("Produto recomendado", melhor_produto["Marca"])
col2.metric("Pontuação", f"{melhor_produto['Pontuação AHP Gaussiano']:.4f}")
col3.metric("Ranking", f"{melhor_produto['Ranking']}º lugar")

if segundo_produto is not None:
    diferenca = (
        melhor_produto["Pontuação AHP Gaussiano"]
        - segundo_produto["Pontuação AHP Gaussiano"]
    )
    col4.metric("Vantagem sobre o 2º", f"{diferenca:.4f}")
else:
    diferenca = 0
    col4.metric("Vantagem sobre o 2º", "-")

st.success(
    f"✅ Pelo método AHP Gaussiano, o produto mais indicado é **{melhor_produto['Marca']}**, "
    f"com pontuação final de **{melhor_produto['Pontuação AHP Gaussiano']:.4f}**."
)

# Interpretação automática do resultado
criterio_mais_importante = diagnostico.iloc[0]["Critério"]
peso_mais_importante = diagnostico.iloc[0]["Peso"]

if segundo_produto is not None:
    if diferenca < 0.02:
        intensidade = "muito pequena"
        recomendacao_cautela = (
            "A diferença entre o primeiro e o segundo colocado é baixa. "
            "Recomenda-se revisar os dados, principalmente os critérios com maior peso."
        )
    elif diferenca < 0.08:
        intensidade = "moderada"
        recomendacao_cautela = (
            "A decisão apresenta vantagem moderada para o primeiro colocado."
        )
    else:
        intensidade = "alta"
        recomendacao_cautela = (
            "A decisão apresenta vantagem clara para o primeiro colocado."
        )

    st.markdown(
        f"""
        **Interpretação:** o produto **{melhor_produto['Marca']}** ficou à frente de
        **{segundo_produto['Marca']}** por uma diferença **{intensidade}** de
        **{diferenca:.4f}** ponto(s). O critério que mais influenciou o cálculo foi
        **{criterio_mais_importante}**, com peso de **{peso_mais_importante:.2%}**.

        {recomendacao_cautela}
        """
    )

# =====================================================
# 8. RANKING FINAL
# =====================================================
st.subheader("📊 Ranking Final")

resultado_exibicao = resultado.copy()
for coluna in criterios:
    resultado_exibicao[coluna] = resultado_exibicao[coluna].round(4)

resultado_exibicao["Pontuação AHP Gaussiano"] = resultado_exibicao[
    "Pontuação AHP Gaussiano"
].round(4)

st.dataframe(resultado_exibicao, use_container_width=True)

fig_ranking = px.bar(
    resultado,
    x="Marca",
    y="Pontuação AHP Gaussiano",
    text="Pontuação AHP Gaussiano",
    title="Pontuação Final por Produto"
)
fig_ranking.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig_ranking, use_container_width=True)

# =====================================================
# 9. PESOS E DIAGNÓSTICO
# =====================================================
st.subheader("⚖️ Pesos Calculados e Diagnóstico dos Critérios")

diagnostico_exibicao = diagnostico.copy()
for coluna in ["Média normalizada", "Desvio-padrão", "Fator Gaussiano", "Peso"]:
    diagnostico_exibicao[coluna] = diagnostico_exibicao[coluna].round(4)

st.dataframe(diagnostico_exibicao, use_container_width=True)

fig_pesos = px.bar(
    diagnostico_exibicao,
    x="Critério",
    y="Peso",
    text="Peso",
    title="Pesos dos Critérios pelo Fator Gaussiano"
)
fig_pesos.update_traces(textposition="outside")
st.plotly_chart(fig_pesos, use_container_width=True)

st.markdown(
    f"""
    **Leitura dos pesos:** o critério **{criterio_mais_importante}** recebeu o maior peso
    porque apresentou maior capacidade de discriminar os produtos após a normalização.
    No AHP Gaussiano, critérios com maior dispersão relativa tendem a ganhar mais influência
    na decisão.
    """
)

# =====================================================
# 10. MATRIZ NORMALIZADA
# =====================================================
st.subheader("📐 Matriz Normalizada")

df_normalizada = matriz_normalizada.copy()
df_normalizada.insert(0, "Marca", df["Marca"].values)

for coluna in criterios:
    df_normalizada[coluna] = df_normalizada[coluna].round(4)

st.dataframe(df_normalizada, use_container_width=True)

# =====================================================
# 11. CONTRIBUIÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("🧩 Contribuição de Cada Critério na Pontuação")

contribuicoes_exibicao = contribuicoes.copy()
for coluna in criterios:
    contribuicoes_exibicao[coluna] = contribuicoes_exibicao[coluna].round(4)

contribuicoes_exibicao["Pontuação total"] = contribuicoes[criterios].sum(axis=1).round(4)
contribuicoes_exibicao = contribuicoes_exibicao.sort_values("Pontuação total", ascending=False)

st.dataframe(contribuicoes_exibicao, use_container_width=True)

fig_contrib = px.bar(
    contribuicoes_exibicao,
    x="Marca",
    y=criterios,
    title="Composição da Pontuação Final por Critério",
    barmode="stack"
)
st.plotly_chart(fig_contrib, use_container_width=True)

# =====================================================
# 12. ANÁLISE INDIVIDUAL DO PRODUTO VENCEDOR
# =====================================================
st.subheader("🔎 Por que esse produto venceu?")

linha_melhor = df_normalizada[df_normalizada["Marca"] == melhor_produto["Marca"]]
contrib_melhor = contribuicoes_exibicao[
    contribuicoes_exibicao["Marca"] == melhor_produto["Marca"]
]

if not linha_melhor.empty and not contrib_melhor.empty:
    melhores_criterios = (
        contrib_melhor[criterios]
        .iloc[0]
        .sort_values(ascending=False)
    )

    st.markdown(
        f"""
        O produto **{melhor_produto['Marca']}** venceu porque apresentou a melhor combinação
        entre desempenho normalizado e peso dos critérios. A maior parcela da sua pontuação veio de:

        1. **{melhores_criterios.index[0]}**: contribuição de **{melhores_criterios.iloc[0]:.4f}**
        2. **{melhores_criterios.index[1]}**: contribuição de **{melhores_criterios.iloc[1]:.4f}**
        3. **{melhores_criterios.index[2]}**: contribuição de **{melhores_criterios.iloc[2]:.4f}**
        """
    )

# =====================================================
# 13. ALERTAS METODOLÓGICOS
# =====================================================
st.subheader("⚠️ Alertas de Qualidade dos Dados")

alertas = []

for criterio in criterios:
    if df[criterio].nunique() == 1:
        alertas.append(
            f"O critério **{criterio}** possui o mesmo valor para todos os produtos; "
            "ele não discrimina as alternativas."
        )

if pesos.max() > 0.70:
    alertas.append(
        "Um único critério concentrou mais de 70% do peso. "
        "A decisão pode estar muito dependente desse critério."
    )

if segundo_produto is not None and diferenca < 0.02:
    alertas.append(
        "A diferença entre os dois primeiros colocados é muito pequena. "
        "Considere revisar os dados ou incluir novos critérios."
    )

if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("Nenhum alerta crítico identificado nos dados atuais.")

# =====================================================
# 14. EXPLICAÇÃO METODOLÓGICA
# =====================================================
with st.expander("🧮 Como o cálculo foi feito"):
    st.markdown(
        """
        O **AHP Gaussiano** calcula os pesos dos critérios com base na dispersão dos dados.
        A ideia central é que um critério que diferencia mais as alternativas possui maior
        poder de discriminação.

        **Etapas utilizadas nesta aplicação:**

        1. Conversão dos dados digitados para valores numéricos.
        2. Definição da direção de preferência de cada critério:
           - **Maior é melhor**
           - **Menor é melhor**
        3. Normalização dos critérios.
        4. Cálculo da média e do desvio-padrão de cada critério normalizado.
        5. Cálculo do fator gaussiano.
        6. Transformação dos fatores gaussianos em pesos.
        7. Cálculo da pontuação final de cada produto.

        **Fator Gaussiano:**

        $$
        FG_j = \\frac{\\sigma_j}{\\bar{x}_j}
        $$

        **Peso do critério:**

        $$
        peso_j = \\frac{FG_j}{\\sum FG_j}
        $$

        **Pontuação final da alternativa:**

        $$
        pontuacao_i = \\sum_j normalizado_{ij} \\times peso_j
        $$

        **Observação:** diferentemente do AHP tradicional, esta aplicação não exige matriz
        de comparação par a par nem cálculo de razão de consistência. Os pesos são obtidos
        automaticamente a partir da própria matriz de decisão.
        """
    )

with st.expander("📌 Como interpretar o resultado"):
    st.markdown(
        """
        - **Pontuação AHP Gaussiano:** mede o desempenho global do produto considerando todos os critérios.
        - **Ranking:** ordena os produtos da maior para a menor pontuação.
        - **Peso:** mostra quanto cada critério influenciou a decisão.
        - **Matriz normalizada:** mostra os valores comparáveis entre critérios diferentes.
        - **Contribuição dos critérios:** mostra de onde veio a pontuação final de cada produto.

        Um produto pode vencer mesmo não sendo o melhor em todos os critérios, desde que apresente
        o melhor equilíbrio nos critérios mais relevantes.
        """
    )
