import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="AHP Gaussiano - Decisão de Compra",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Método AHP-Gaussiano - Decisão de Compra")

st.markdown("""
Aplicação do **Método AHP-Gaussiano** para apoio à decisão multicritério.

O método calcula os pesos dos critérios a partir da própria matriz de decisão,
usando o **Fator Gaussiano**, representado pelo coeficiente de variação:

`FG = σ / μ`

Depois, os fatores gaussianos são normalizados para formar os pesos dos critérios.
""")


# =====================================================
# BASE PADRÃO
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
        "Custo (R$)": [137.75, 104.09, 64.90, 94.91],
        "Quantidade (g)": [900, 2000, 1000, 900],
        "Benefício Extra": [1.0, 0.0, 0.0, 0.0],
        "Pontuação": [4.9, 4.7, 4.8, 5.0]
    })


if "dados_produtos" not in st.session_state:
    st.session_state["dados_produtos"] = carregar_dados_padrao()


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================
def converter_numero(valor):
    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip().replace(" ", "")

    if valor == "":
        return np.nan

    if "." in valor and "," in valor:
        valor = valor.replace(".", "").replace(",", ".")
    else:
        valor = valor.replace(",", ".")

    try:
        return float(valor)
    except ValueError:
        return np.nan


def validar_dados(df, criterios):
    erros = []

    if df.empty:
        erros.append("Nenhum produto válido encontrado.")

    if len(df) < 2:
        erros.append("O método exige pelo menos duas alternativas.")

    for criterio in criterios:
        if criterio not in df.columns:
            erros.append(f"Critério ausente na matriz: {criterio}")

    for criterio in criterios:
        if criterio in df.columns and df[criterio].isna().any():
            erros.append(f"O critério {criterio} possui valores vazios ou inválidos.")

    return erros


def preparar_matriz_decisao(df, criterios, tipo_criterio):
    """
    Etapa anterior à normalização.

    Para critérios 'Maior é melhor':
        usa o valor original.

    Para critérios 'Menor é melhor':
        aplica transformação inversa 1/x.

    Observação:
    O AHP-Gaussiano trabalha com direção MAX na matriz normalizada.
    Portanto, critérios de minimização precisam ser convertidos antes.
    """

    matriz = df[criterios].copy().astype(float)

    alertas = []

    for criterio in criterios:
        direcao = tipo_criterio[criterio]

        if "Menor" in direcao:
            if (matriz[criterio] <= 0).any():
                alertas.append(
                    f"O critério '{criterio}' possui valor menor ou igual a zero. "
                    "Não é possível aplicar 1/x com segurança."
                )

            matriz[criterio] = np.where(
                matriz[criterio] > 0,
                1 / matriz[criterio],
                np.nan
            )

    return matriz, alertas


def normalizar_por_soma(matriz):
    """
    Normalização:
        r_ij = x_ij / soma(x_j)
    """

    matriz_norm = pd.DataFrame(index=matriz.index)

    for col in matriz.columns:
        soma = matriz[col].sum(skipna=True)

        if pd.isna(soma) or soma == 0:
            matriz_norm[col] = np.nan
        else:
            matriz_norm[col] = matriz[col] / soma

    return matriz_norm


def calcular_ahp_gaussiano(df, criterios, tipo_criterio, id_col="Marca"):
    matriz_original = df[criterios].copy().astype(float)

    matriz_ajustada, alertas_direcao = preparar_matriz_decisao(
        df=df,
        criterios=criterios,
        tipo_criterio=tipo_criterio
    )

    if matriz_ajustada.isna().any().any():
        raise ValueError(
            "A matriz ajustada possui valores inválidos. "
            "Verifique zeros em critérios configurados como 'Menor é melhor'."
        )

    matriz_norm = normalizar_por_soma(matriz_ajustada)

    if matriz_norm.isna().any().any():
        raise ValueError(
            "A matriz normalizada possui valores inválidos. "
            "Verifique critérios com soma igual a zero."
        )

    # Média das alternativas em cada critério
    medias = matriz_norm[criterios].mean(axis=0)

    # Desvio-padrão AMOSTRAL para aderir ao padrão DESVPAD usado nos exemplos acadêmicos
    desvios = matriz_norm[criterios].std(axis=0, ddof=1)

    # Fator Gaussiano
    fator_gaussiano = desvios / medias.replace(0, np.nan)
    fator_gaussiano = fator_gaussiano.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Peso normalizado do critério
    soma_fg = fator_gaussiano.sum()

    if soma_fg > 0:
        pesos = fator_gaussiano / soma_fg
    else:
        pesos = pd.Series(
            [1 / len(criterios)] * len(criterios),
            index=criterios
        )

    # Pontuação final das alternativas
    contribuicoes = matriz_norm[criterios].mul(pesos, axis=1)
    scores = contribuicoes.sum(axis=1)

    resultado = df.copy()
    resultado["Pontuação AHP-Gaussiano"] = scores.values
    resultado["Ranking"] = (
        resultado["Pontuação AHP-Gaussiano"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    resultado["Classificação"] = np.select(
        [
            resultado["Ranking"] == 1,
            resultado["Ranking"] == 2,
            resultado["Ranking"] == 3
        ],
        [
            "Melhor escolha",
            "Segunda opção",
            "Terceira opção"
        ],
        default="Alternativa complementar"
    )

    diagnostico = pd.DataFrame({
        "Critério": criterios,
        "Direção": [tipo_criterio[c] for c in criterios],
        "Média Normalizada": medias.values,
        "Desvio-Padrão Amostral (σ)": desvios.values,
        "Fator Gaussiano (σ/μ)": fator_gaussiano.values,
        "Peso Normalizado": pesos.values,
        "Peso Calculado (%)": pesos.values * 100
    }).sort_values(
        "Peso Calculado (%)",
        ascending=False
    ).reset_index(drop=True)

    contribuicoes.insert(0, id_col, df[id_col].values)
    contribuicoes["Pontuação Total"] = scores.values

    resultado = resultado.sort_values(
        "Pontuação AHP-Gaussiano",
        ascending=False
    ).reset_index(drop=True)

    matriz_norm_exib = matriz_norm.copy()
    matriz_norm_exib.insert(0, id_col, df[id_col].values)

    matriz_ajustada_exib = matriz_ajustada.copy()
    matriz_ajustada_exib.insert(0, id_col, df[id_col].values)

    return {
        "resultado": resultado,
        "diagnostico": diagnostico,
        "matriz_original": matriz_original,
        "matriz_ajustada": matriz_ajustada_exib,
        "matriz_normalizada": matriz_norm_exib,
        "contribuicoes": contribuicoes,
        "pesos": pesos,
        "alertas_direcao": alertas_direcao
    }


# =====================================================
# CONTROLES
# =====================================================
col_a, col_b = st.columns([1, 4])

with col_a:
    if st.button("🔄 Restaurar matriz padrão"):
        st.session_state["dados_produtos"] = carregar_dados_padrao()
        st.rerun()

with col_b:
    st.info(
        "Edite os produtos, altere os critérios e escolha se cada critério é de maximização "
        "ou minimização. O cálculo seguirá a sequência: matriz ajustada → normalização → "
        "média → desvio padrão → fator gaussiano → pesos → ranking."
    )


# =====================================================
# MATRIZ DE DECISÃO
# =====================================================
st.subheader("📋 Matriz de Decisão")

df_editado = st.data_editor(
    st.session_state["dados_produtos"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Produto": st.column_config.TextColumn(
            "Produto",
            help="Identificador do produto"
        ),
        "Marca": st.column_config.TextColumn(
            "Marca",
            required=True,
            help="Nome ou descrição do produto"
        ),
        "Custo (R$)": st.column_config.NumberColumn(
            "Custo (R$)",
            format="%.2f",
            min_value=0.0,
            help="Preço do produto"
        ),
        "Quantidade (g)": st.column_config.NumberColumn(
            "Quantidade (g)",
            format="%.0f",
            min_value=0.0,
            help="Quantidade total em gramas"
        ),
        "Benefício Extra": st.column_config.NumberColumn(
            "Benefício Extra",
            format="%.2f",
            min_value=0.0,
            help="Benefício adicional do produto"
        ),
        "Pontuação": st.column_config.NumberColumn(
            "Pontuação",
            format="%.2f",
            min_value=0.0,
            max_value=10.0,
            help="Nota, avaliação ou reputação do produto"
        ),
    },
    key="editor_produtos"
)

st.session_state["dados_produtos"] = df_editado.copy()


# =====================================================
# TRATAMENTO DOS DADOS
# =====================================================
df = df_editado.copy()

df = df.dropna(subset=["Marca"])
df["Marca"] = df["Marca"].astype(str).str.strip()
df = df[df["Marca"] != ""]

criterios = [
    "Custo (R$)",
    "Quantidade (g)",
    "Benefício Extra",
    "Pontuação"
]

for criterio in criterios:
    df[criterio] = df[criterio].apply(converter_numero)

linhas_invalidas = df[df[criterios].isna().any(axis=1)]

if not linhas_invalidas.empty:
    st.warning("⚠️ Algumas linhas possuem valores inválidos e serão ignoradas.")
    st.dataframe(
        linhas_invalidas[["Produto", "Marca"] + criterios],
        use_container_width=True
    )

df = df.dropna(subset=criterios).reset_index(drop=True)

erros_validacao = validar_dados(df, criterios)

if erros_validacao:
    for erro in erros_validacao:
        st.error(f"❌ {erro}")
    st.stop()


# =====================================================
# CONFIGURAÇÃO DOS CRITÉRIOS
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")

col1, col2, col3, col4 = st.columns(4)

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

with col4:
    tipo_pontuacao = st.selectbox(
        "Pontuação",
        ["Maior é melhor", "Menor é melhor"],
        index=0
    )

tipo_criterio = {
    "Custo (R$)": tipo_custo,
    "Quantidade (g)": tipo_quantidade,
    "Benefício Extra": tipo_beneficio,
    "Pontuação": tipo_pontuacao,
}


# =====================================================
# EXECUTAR CÁLCULO
# =====================================================
st.subheader("🚀 Executar Cálculo")

if st.button("Calcular AHP-Gaussiano", type="primary"):
    try:
        resultado_calculo = calcular_ahp_gaussiano(
            df=df,
            criterios=criterios,
            tipo_criterio=tipo_criterio
        )

        st.session_state["ahp_resultado_calculo"] = resultado_calculo
        st.success("✅ Cálculo concluído com sucesso.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Erro no cálculo: {type(e).__name__}: {e}")
        st.exception(e)


# =====================================================
# RESULTADOS
# =====================================================
if "ahp_resultado_calculo" not in st.session_state:
    st.info("👆 Clique em **Calcular AHP-Gaussiano** para visualizar os resultados.")
    st.stop()


dados = st.session_state["ahp_resultado_calculo"]

resultado = dados["resultado"]
diagnostico = dados["diagnostico"]
matriz_normalizada = dados["matriz_normalizada"]
matriz_ajustada = dados["matriz_ajustada"]
contribuicoes = dados["contribuicoes"]
pesos = dados["pesos"]
alertas_direcao = dados["alertas_direcao"]


# =====================================================
# RESULTADO EXECUTIVO
# =====================================================
st.subheader("🏆 Resultado da Decisão")

melhor = resultado.iloc[0]
segundo = resultado.iloc[1] if len(resultado) > 1 else None

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Produto recomendado", melhor["Marca"])
col2.metric("Pontuação final", f"{melhor['Pontuação AHP-Gaussiano']:.4f}")
col3.metric("Ranking", f"{melhor['Ranking']}º")
col4.metric("Nota do produto", f"{melhor['Pontuação']:.2f}")

if segundo is not None:
    diferenca = melhor["Pontuação AHP-Gaussiano"] - segundo["Pontuação AHP-Gaussiano"]
    col5.metric("Vantagem sobre 2º", f"{diferenca:.4f}")
else:
    diferenca = np.nan
    col5.metric("Vantagem sobre 2º", "-")

st.success(
    f"✅ O produto recomendado pelo método AHP-Gaussiano é "
    f"**{melhor['Marca']}**, com pontuação final de "
    f"**{melhor['Pontuação AHP-Gaussiano']:.4f}**."
)

criterio_principal = diagnostico.iloc[0]["Critério"]
peso_principal = diagnostico.iloc[0]["Peso Calculado (%)"]
fg_principal = diagnostico.iloc[0]["Fator Gaussiano (σ/μ)"]

st.markdown(f"""
### Leitura técnica

O critério com maior influência foi **{criterio_principal}**, com peso de
**{peso_principal:.2f}%**.

Isso ocorreu porque ele apresentou o maior **Fator Gaussiano**:

`σ / μ = {fg_principal:.4f}`

Ou seja, foi o critério que mais diferenciou as alternativas na matriz normalizada.
""")


# =====================================================
# RANKING FINAL
# =====================================================
st.subheader("📊 Ranking Final")

cols_ranking = [
    "Ranking",
    "Marca",
    "Classificação",
    "Pontuação AHP-Gaussiano"
] + criterios

st.dataframe(
    resultado[cols_ranking].round(4),
    use_container_width=True
)

fig_rank = px.bar(
    resultado.sort_values("Pontuação AHP-Gaussiano"),
    x="Marca",
    y="Pontuação AHP-Gaussiano",
    text="Pontuação AHP-Gaussiano",
    title="Pontuação Final por Produto",
    color="Pontuação AHP-Gaussiano",
    color_continuous_scale="Viridis"
)

fig_rank.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_rank.update_layout(
    showlegend=False,
    yaxis_title="Pontuação final",
    xaxis_title="Produto"
)

st.plotly_chart(fig_rank, use_container_width=True)


# =====================================================
# PESOS DOS CRITÉRIOS
# =====================================================
st.subheader("⚖️ Pesos Calculados dos Critérios")

st.dataframe(
    diagnostico.round(6),
    use_container_width=True
)

fig_pesos = px.bar(
    diagnostico,
    x="Critério",
    y="Peso Calculado (%)",
    text="Peso Calculado (%)",
    title="Peso Objetivo dos Critérios",
    color="Peso Calculado (%)",
    color_continuous_scale="Blues"
)

fig_pesos.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_pesos.update_layout(
    showlegend=False,
    yaxis_title="Peso (%)",
    xaxis_title="Critério"
)

st.plotly_chart(fig_pesos, use_container_width=True)


# =====================================================
# MATRIZES DO MÉTODO
# =====================================================
st.subheader("📐 Matrizes do Cálculo")

with st.expander("1️⃣ Matriz ajustada pela direção de preferência"):
    st.dataframe(
        matriz_ajustada.round(6),
        use_container_width=True
    )

    st.caption(
        "Critérios do tipo 'Menor é melhor' são convertidos por 1/x antes da normalização."
    )

with st.expander("2️⃣ Matriz normalizada"):
    st.dataframe(
        matriz_normalizada.round(6),
        use_container_width=True
    )

    st.caption(
        "Normalização por soma: rᵢⱼ = xᵢⱼ / Σxᵢⱼ"
    )


# =====================================================
# CONTRIBUIÇÕES
# =====================================================
st.subheader("🧩 Contribuição de Cada Critério na Pontuação")

st.dataframe(
    contribuicoes.round(6),
    use_container_width=True
)

fig_contrib = px.bar(
    contribuicoes.sort_values("Pontuação Total"),
    x="Marca",
    y=criterios,
    title="Composição da Pontuação Final por Critério",
    barmode="stack"
)

fig_contrib.update_layout(
    yaxis_title="Contribuição na pontuação",
    xaxis_title="Produto"
)

st.plotly_chart(fig_contrib, use_container_width=True)


# =====================================================
# POR QUE O PRODUTO VENCEU
# =====================================================
st.subheader("🔎 Por que este produto venceu?")

contrib_vencedor = contribuicoes[
    contribuicoes["Marca"] == melhor["Marca"]
].iloc[0]

top_contrib = contrib_vencedor[criterios].sort_values(ascending=False)

df_top = pd.DataFrame({
    "Critério": top_contrib.index,
    "Contribuição": top_contrib.values,
    "Peso do Critério (%)": [pesos[c] * 100 for c in top_contrib.index]
})

st.markdown(
    f"**{melhor['Marca']}** venceu porque obteve a maior soma ponderada "
    f"na matriz normalizada."
)

st.dataframe(
    df_top.round(6),
    use_container_width=True
)

st.info(
    "No AHP-Gaussiano, uma alternativa não precisa ser a melhor em todos os critérios. "
    "Ela precisa performar bem principalmente nos critérios com maior dispersão, pois são eles "
    "que recebem maior peso objetivo."
)


# =====================================================
# ALERTAS DE QUALIDADE
# =====================================================
st.subheader("⚠️ Alertas de Qualidade dos Dados")

alertas = []

for alerta in alertas_direcao:
    alertas.append(alerta)

for criterio in criterios:
    if df[criterio].nunique() == 1:
        alertas.append(
            f"O critério '{criterio}' possui o mesmo valor para todas as alternativas. "
            "Ele não discrimina os produtos."
        )

for _, row in diagnostico.iterrows():
    if row["Peso Calculado (%)"] > 70:
        alertas.append(
            f"O critério '{row['Critério']}' concentra {row['Peso Calculado (%)']:.2f}% "
            "do peso total. A decisão está muito dependente desse critério."
        )

if segundo is not None:
    if diferenca < 0.02:
        alertas.append(
            "A diferença entre o 1º e o 2º colocado é muito pequena. "
            "Recomenda-se revisar critérios ou incluir mais variáveis."
        )

if alertas:
    for alerta in alertas:
        st.warning(f"⚠️ {alerta}")
else:
    st.success("✅ Nenhum alerta crítico encontrado.")


# =====================================================
# METODOLOGIA
# =====================================================
with st.expander("🧮 Metodologia aplicada"):
    st.markdown("""
    ## Método AHP-Gaussiano

    ### 1. Matriz de decisão

    Cada produto é uma alternativa e cada coluna numérica é um critério.

    ### 2. Direção de preferência

    Critérios podem ser:

    - **Maior é melhor**
    - **Menor é melhor**

    Para critérios de minimização, a aplicação converte o valor usando:

    ```text
    x' = 1 / x
    ```

    ### 3. Normalização

    Cada critério é normalizado por soma:

    ```text
    rᵢⱼ = xᵢⱼ / Σxᵢⱼ
    ```

    ### 4. Média normalizada

    ```text
    μⱼ = média dos valores normalizados do critério j
    ```

    ### 5. Desvio padrão amostral

    ```text
    σⱼ = desvio padrão amostral dos valores normalizados do critério j
    ```

    ### 6. Fator Gaussiano

    ```text
    FGⱼ = σⱼ / μⱼ
    ```

    ### 7. Peso normalizado do critério

    ```text
    wⱼ = FGⱼ / ΣFGⱼ
    ```

    ### 8. Pontuação final

    ```text
    Sᵢ = Σ(rᵢⱼ × wⱼ)
    ```

    A alternativa com maior `Sᵢ` recebe a primeira posição no ranking.
    """)


# =====================================================
# RODAPÉ
# =====================================================
st.markdown("---")
st.caption(
    "Aplicação acadêmica | Método AHP-Gaussiano | Decisão multicritério"
)