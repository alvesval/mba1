import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="AHP Gaussiano - Escolha de Produto",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Método AHP Gaussiano - Decisão de Compra")
st.markdown(
    """
    Esta aplicação implementa o **Método AHP Gaussiano** conforme proposto pelo **Prof. Dr. Marcos Santos**.
    Os pesos dos critérios são determinados objetivamente pela dispersão dos dados (Coeficiente de Variação),
    eliminando a subjetividade das matrizes de comparação par a par do AHP tradicional.
    """
)


# =====================================================
# 1. BASE PADRÃO
# =====================================================
def carregar_dados_padrao():
    return pd.DataFrame({
        "Produto": ["1", "2", "3", "4"],
        "Marca": ["KIT NUTRI", "KIT WHEY", "WHEY PROTEIN ISOLADO", "WHEY PROTEIN ALPEX"],
        "Custo (R$)": [0.170, 0.224, 0.360, 0.246],
        "Quantidade (g)": [0.188, 0.417, 0.208, 0.188],
        "Benefício Extra": [1.0, 0.0, 0.0, 0.0]
    })


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
        "Edite os dados, inclua novos produtos e ajuste a direção de preferência de cada critério. "
        "Valores numéricos podem usar ponto (.) ou vírgula (,) como separador decimal."
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
        "Produto": st.column_config.TextColumn("Produto", disabled=True),
        "Marca": st.column_config.TextColumn("Marca", required=True),
        "Custo (R$)": st.column_config.TextColumn("Custo (R$)"),
        "Quantidade (g)": st.column_config.TextColumn("Quantidade (g)"),
        "Benefício Extra": st.column_config.TextColumn("Benefício Extra"),
    },
    key="editor_produtos"
)

# Atualiza sessão
st.session_state["dados_produtos"] = df_editado.copy()


# =====================================================
# 4. TRATAMENTO DOS DADOS
# =====================================================
def converter_numero(valor):
    """Converte strings para float tratando formatos BR/US."""
    if pd.isna(valor):
        return np.nan
    valor = str(valor).strip().replace(" ", "")
    if valor == "":
        return np.nan

    # Formato brasileiro: 1.000,50 -> 1000.50
    if "." in valor and "," in valor:
        valor = valor.replace(".", "").replace(",", ".")
    else:
        valor = valor.replace(",", ".")

    try:
        return float(valor)
    except ValueError:
        return np.nan


# Limpeza e conversão
df = df_editado.copy()
df = df.dropna(subset=["Marca"])
df["Marca"] = df["Marca"].astype(str).str.strip()
df = df[df["Marca"] != ""]

criterios = ["Custo (R$)", "Quantidade (g)", "Benefício Extra"]
for criterio in criterios:
    df[criterio] = df[criterio].apply(converter_numero)

linhas_invalidas = df[df[criterios].isna().any(axis=1)]
if not linhas_invalidas.empty:
    st.warning("⚠️ Algumas linhas possuem valores numéricos inválidos ou vazios e serão ignoradas.")
    st.dataframe(linhas_invalidas[["Produto", "Marca"] + criterios], use_container_width=True)

df = df.dropna(subset=criterios).reset_index(drop=True)

if df.empty:
    st.error("❌ Nenhum produto válido encontrado. Preencha a matriz.")
    st.stop()
if len(df) < 2:
    st.warning("⚠️ O AHP Gaussiano exige pelo menos dois produtos para calcular a dispersão dos critérios.")
    st.stop()

# =====================================================
# 5. CONFIGURAÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")
col1, col2, col3 = st.columns(3)
with col1:
    tipo_custo = st.selectbox("Custo (R$)", ["Menor é melhor", "Maior é melhor"], index=0)
with col2:
    tipo_quantidade = st.selectbox("Quantidade (g)", ["Maior é melhor", "Menor é melhor"], index=0)
with col3:
    tipo_beneficio = st.selectbox("Benefício Extra", ["Maior é melhor", "Menor é melhor"], index=0)

tipo_criterio = {
    "Custo (R$)": tipo_custo,
    "Quantidade (g)": tipo_quantidade,
    "Benefício Extra": tipo_beneficio,
}


# =====================================================
# 6. FUNÇÕES DO AHP GAUSSIANO (Corrigido e Academicamente Alinhado)
# =====================================================
def normalizar_criterio_soma(serie, tipo):
    """
    Normalização por soma (padrão AHP):
    - Maior é melhor: x / Σx
    - Menor é melhor: (1/x) / Σ(1/x)
    Proteção contra divisão por zero mantendo a proporcionalidade.
    """
    valores = pd.to_numeric(serie, errors="coerce").astype(float)
    eps = 1e-9  # epsilon para estabilidade numérica

    if tipo == "Maior é melhor":
        v = valores.clip(lower=0)
        soma = v.sum()
        if soma <= 0:
            return pd.Series([1 / len(v)] * len(v), index=v.index)
        return v / soma

    # Menor é melhor
    v = valores.copy()
    # Substitui zeros ou negativos pelo menor positivo encontrado ou epsilon
    min_pos = v[v > 0].min()
    if pd.isna(min_pos) or min_pos <= 0:
        min_pos = eps
    v[v <= 0] = min_pos * 0.5

    inv = 1.0 / v
    soma_inv = inv.sum()
    if soma_inv <= 0:
        return pd.Series([1 / len(v)] * len(v), index=v.index)
    return inv / soma_inv


def calcular_ahp_gaussiano(df, criterios, tipo_criterio):
    """
    Implementação rigorosa do AHP Gaussiano (Santos, M. et al.)
    Etapas:
    1. Normalização da Matriz de Decisão
    2. Cálculo da Média e Desvio-Padrão por coluna
    3. Fator Gaussiano (Coeficiente de Variação): FG = σ / μ
    4. Ponderação: w_j = FG_j / ΣFG_j
    5. Pontuação Final: S_i = Σ(r_ij * w_j)
    """
    matriz_normalizada = pd.DataFrame(index=df.index)
    for criterio in criterios:
        matriz_normalizada[criterio] = normalizar_criterio_soma(
            df[criterio], tipo_criterio[criterio]
        )

    medias = matriz_normalizada.mean()
    # Desvio-padrão populacional (ddof=0) conforme padrão acadêmico do método
    desvios = matriz_normalizada.std(ddof=0)

    # Fator Gaussiano = Coeficiente de Variação
    fator_gaussiano = desvios / medias.replace(0, np.nan)
    fator_gaussiano = fator_gaussiano.fillna(0)

    if fator_gaussiano.sum() == 0:
        pesos = pd.Series([1 / len(criterios)] * len(criterios), index=criterios)
    else:
        pesos = fator_gaussiano / fator_gaussiano.sum()

    # Pontuação final
    pontuacao_final = matriz_normalizada.dot(pesos)

    resultado = df.copy()
    resultado["Pontuação AHP Gaussiano"] = pontuacao_final.values
    resultado["Ranking"] = (
        resultado["Pontuação AHP Gaussiano"]
            .rank(ascending=False, method="dense")
            .astype(int)
    )

    contribuicoes = matriz_normalizada.mul(pesos, axis=1)
    contribuicoes.insert(0, "Marca", df["Marca"].values)

    resultado = resultado.sort_values("Pontuação AHP Gaussiano", ascending=False)

    diagnostico = pd.DataFrame({
        "Critério": criterios,
        "Tipo": [tipo_criterio[c] for c in criterios],
        "Média normalizada": medias.values,
        "Desvio-padrão": desvios.values,
        "Fator Gaussiano (CV)": fator_gaussiano.values,
        "Peso Calculado": pesos.values,
    }).sort_values("Peso Calculado", ascending=False)

    return matriz_normalizada, pesos, resultado, diagnostico, contribuicoes


# Executa cálculo
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
col3.metric("Ranking", f"{int(melhor_produto['Ranking'])}º lugar")

diferenca = 0
if segundo_produto is not None:
    diferenca = melhor_produto["Pontuação AHP Gaussiano"] - segundo_produto["Pontuação AHP Gaussiano"]
    col4.metric("Vantagem sobre o 2º", f"{diferenca:.4f}")
else:
    col4.metric("Vantagem sobre o 2º", "-")

st.success(
    f"✅ Pelo método AHP Gaussiano, o produto mais indicado é **{melhor_produto['Marca']}**, "
    f"com pontuação final de **{melhor_produto['Pontuação AHP Gaussiano']:.4f}**."
)

# Interpretação automática
criterio_mais_importante = diagnostico.iloc[0]["Critério"]
peso_mais_importante = diagnostico.iloc[0]["Peso Calculado"]

if segundo_produto is not None:
    if diferenca < 0.02:
        intensidade = "muito pequena"
        recomendacao_cautela = "A diferença entre o primeiro e o segundo colocado é baixa. Recomenda-se revisar os dados ou critérios."
    elif diferenca < 0.08:
        intensidade = "moderada"
        recomendacao_cautela = "A decisão apresenta vantagem moderada para o primeiro colocado."
    else:
        intensidade = "alta"
        recomendacao_cautela = "A decisão apresenta vantagem clara para o primeiro colocado."

    st.markdown(
        f"""
        **Interpretação Acadêmica:** O produto **{melhor_produto['Marca']}** ficou à frente de 
        **{segundo_produto['Marca']}** por uma diferença **{intensidade}** de **{diferenca:.4f}** ponto(s). 
        O critério que mais influenciou o cálculo foi **{criterio_mais_importante}**, com peso de **{peso_mais_importante:.2%}**.
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
resultado_exibicao["Pontuação AHP Gaussiano"] = resultado_exibicao["Pontuação AHP Gaussiano"].round(4)
st.dataframe(resultado_exibicao, use_container_width=True)

fig_ranking = px.bar(
    resultado, x="Marca", y="Pontuação AHP Gaussiano",
    text="Pontuação AHP Gaussiano", title="Pontuação Final por Produto"
)
fig_ranking.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig_ranking, use_container_width=True)

# =====================================================
# 9. PESOS E DIAGNÓSTICO
# =====================================================
st.subheader("⚖️ Pesos Calculados e Diagnóstico dos Critérios")
diagnostico_exibicao = diagnostico.copy()
for coluna in ["Média normalizada", "Desvio-padrão", "Fator Gaussiano (CV)", "Peso Calculado"]:
    diagnostico_exibicao[coluna] = diagnostico_exibicao[coluna].round(4)
st.dataframe(diagnostico_exibicao, use_container_width=True)

fig_pesos = px.bar(
    diagnostico_exibicao, x="Critério", y="Peso Calculado",
    text="Peso Calculado", title="Pesos dos Critérios pelo Fator Gaussiano"
)
fig_pesos.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig_pesos, use_container_width=True)

st.markdown(
    f"""
    **Leitura Metodológica:** O critério **{criterio_mais_importante}** recebeu o maior peso 
    porque apresentou maior capacidade de discriminação (maior dispersão relativa) entre as alternativas. 
    No AHP Gaussiano, critérios com maior Coeficiente de Variação assumem maior influência objetiva na decisão.
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
    contribuicoes_exibicao, x="Marca", y=criterios,
    title="Composição da Pontuação Final por Critério", barmode="stack"
)
st.plotly_chart(fig_contrib, use_container_width=True)

# =====================================================
# 12. ANÁLISE INDIVIDUAL DO PRODUTO VENCEDOR
# =====================================================
st.subheader("🔎 Por que esse produto venceu?")
contrib_melhor = contribuicoes_exibicao[contribuicoes_exibicao["Marca"] == melhor_produto["Marca"]]
if not contrib_melhor.empty:
    melhores_criterios = contrib_melhor[criterios].iloc[0].sort_values(ascending=False)
    st.markdown(
        f"""
        O produto **{melhor_produto['Marca']}** venceu porque apresentou a melhor combinação 
        entre desempenho normalizado e peso objetivo dos critérios. A maior parcela da sua pontuação veio de:
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
            f"O critério `{criterio}` possui o mesmo valor para todos os produtos; ele não discrimina as alternativas.")
if pesos.max() > 0.70:
    alertas.append(
        "Um único critério concentrou mais de 70% do peso. A decisão pode estar excessivamente dependente desse critério.")
if segundo_produto is not None and diferenca < 0.02:
    alertas.append(
        "A diferença entre os dois primeiros colocados é muito pequena (<0.02). Considere revisar os dados ou incluir novos critérios.")
if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("✅ Nenhum alerta crítico identificado. Os dados estão adequados para aplicação do método.")

# =====================================================
# 14. EXPLICAÇÃO METODOLÓGICA (Referência Acadêmica)
# =====================================================
with st.expander("🧮 Como o cálculo foi feito (Metodologia AHP Gaussiano)"):
    st.markdown(
        """
        O **AHP Gaussiano** (Santos, M. et al.) substitui a subjetividade da matriz de comparação par a par 
        pela análise estatística da dispersão dos dados. A premissa central é: **critérios com maior variabilidade 
        entre as alternativas carregam mais informação e, portanto, devem receber maior peso objetivo.**

        **Etapas de Cálculo:**
        1. **Normalização por Soma:** Padroniza as unidades de medida para uma escala adimensional `[0,1]`.
        2. **Estatísticas Descritivas:** Calcula-se a média ($\\bar{x}_j$) e o desvio-padrão populacional ($\\sigma_j$) de cada coluna normalizada.
        3. **Fator Gaussiano (FG):** Corresponde ao Coeficiente de Variação.
           $$ FG_j = \\frac{\\sigma_j}{\\bar{x}_j} $$
        4. **Ponderação Objetiva:** Os pesos são normalizados para somar 100%.
           $$ w_j = \\frac{FG_j}{\\sum_{k=1}^{n} FG_k} $$
        5. **Pontuação Final (S):** Combinação linear ponderada da matriz normalizada.
           $$ S_i = \\sum_{j=1}^{m} r_{ij} \\times w_j $$

        *Referência:* Método proposto para decisão multicritério objetiva, eliminando inconsistências de julgamento humano 
        típicas do AHP clássico (Saaty, 1980).
        """
    )

with st.expander("📌 Como interpretar o resultado"):
    st.markdown(
        """
        - **Pontuação AHP Gaussiano:** Mede o desempenho global do produto considerando todos os critérios ponderados pela dispersão.
        - **Ranking:** Ordenação decrescente das pontuações.
        - **Peso Calculado:** Representa a influência objetiva de cada critério. Valores mais altos indicam maior poder de discriminação.
        - **Matriz Normalizada:** Valores comparáveis entre critérios de naturezas distintas.
        - **Contribuição dos Critérios:** Decomposição da pontuação final. Um produto pode vencer sem ser o melhor em todos os critérios, 
          desde que apresente o melhor equilíbrio nos critérios mais discriminantes.
        """
    )