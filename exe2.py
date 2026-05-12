import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="AHP Gaussiano - Escolha de Produto",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Método AHP Gaussiano - Decisão de Compra")

st.markdown("""
Esta aplicação implementa o **Método AHP Gaussiano** para apoio à decisão multicritério.

Os pesos dos critérios são calculados objetivamente pela dispersão dos dados, usando o
**Coeficiente de Variação**, reduzindo a subjetividade do AHP tradicional.
""")


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
        "Custo (R$)": [137.75, 104.09, 64.90, 94.91],
        "Quantidade (g)": [900, 2000, 1000, 900],
        "Benefício Extra": [1.0, 0.0, 0.0, 0.0],
        "Pontuação": [4.9, 4.7, 4.8, 5.0]
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
        "Agora a coluna Pontuação também participa do cálculo final."
    )


# =====================================================
# 3. MATRIZ DE DECISÃO
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
# 4. TRATAMENTO DOS DADOS
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
    st.warning("⚠️ Algumas linhas possuem valores numéricos inválidos ou vazios e serão ignoradas.")
    st.dataframe(
        linhas_invalidas[["Produto", "Marca"] + criterios],
        use_container_width=True
    )

df = df.dropna(subset=criterios).reset_index(drop=True)

if df.empty:
    st.error("❌ Nenhum produto válido encontrado.")
    st.stop()

if len(df) < 2:
    st.warning("⚠️ O AHP Gaussiano exige pelo menos dois produtos.")
    st.stop()


# =====================================================
# 5. CONFIGURAÇÃO DOS CRITÉRIOS
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
# 6. CÁLCULO AHP GAUSSIANO
# =====================================================
def calcular_ahp_gaussiano(df, criterios, tipo_criterio, id_col="Marca"):

    matriz = df[criterios].copy().astype(float)

    matriz_original = matriz.copy()

    for col in criterios:
        if "Menor" in tipo_criterio[col]:
            vals = matriz[col].replace(0, np.nan)
            matriz[col] = 1.0 / vals
            matriz[col] = matriz[col].fillna(matriz[col].max())

    matriz_norm = pd.DataFrame(index=matriz.index)

    for col in criterios:
        soma = matriz[col].sum()

        if soma > 0:
            matriz_norm[col] = matriz[col] / soma
        else:
            matriz_norm[col] = 1.0 / len(matriz)

    medias = matriz_norm[criterios].mean()
    desvios = matriz_norm[criterios].std(ddof=0)

    fator_gaussiano = (desvios / medias.replace(0, np.nan)).fillna(0)

    if fator_gaussiano.sum() > 0:
        pesos = fator_gaussiano / fator_gaussiano.sum()
    else:
        pesos = pd.Series(
            [1 / len(criterios)] * len(criterios),
            index=criterios
        )

    scores = matriz_norm[criterios].dot(pesos)

    resultado = df.copy()
    resultado["Pontuação AHP Gaussiano"] = scores.values

    resultado["Ranking"] = (
        resultado["Pontuação AHP Gaussiano"]
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
        "Tipo": [tipo_criterio[c] for c in criterios],
        "Média Normalizada": medias.values,
        "Desvio-Padrão (σ)": desvios.values,
        "Fator Gaussiano (CV)": fator_gaussiano.values,
        "Peso Calculado (%)": (pesos.values * 100).round(2),
    }).sort_values(
        "Peso Calculado (%)",
        ascending=False
    ).reset_index(drop=True)

    contribuicoes = matriz_norm[criterios].mul(pesos, axis=1)
    contribuicoes.insert(0, id_col, df[id_col].values)

    resultado = resultado.sort_values(
        "Pontuação AHP Gaussiano",
        ascending=False
    ).reset_index(drop=True)

    return matriz_norm, matriz_original, pesos, resultado, diagnostico, contribuicoes


# =====================================================
# 7. EXECUÇÃO
# =====================================================
st.subheader("🚀 Executar Cálculo")

if st.button("Calcular AHP Gaussiano", type="primary"):
    with st.spinner("Processando método AHP Gaussiano..."):
        try:
            (
                matriz_norm,
                matriz_original,
                pesos,
                resultado,
                diagnostico,
                contribuicoes
            ) = calcular_ahp_gaussiano(
                df,
                criterios,
                tipo_criterio
            )

            st.session_state.update({
                "resultado_ahp": resultado,
                "diagnostico_ahp": diagnostico,
                "matriz_norm_ahp": matriz_norm,
                "matriz_original_ahp": matriz_original,
                "contribuicoes_ahp": contribuicoes,
                "pesos_ahp": pesos
            })

            st.success("✅ Cálculo concluído com sucesso!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erro: {type(e).__name__}: {e}")
            st.exception(e)


# =====================================================
# 8. RESULTADOS
# =====================================================
if "resultado_ahp" in st.session_state:

    resultado = st.session_state["resultado_ahp"]
    diagnostico = st.session_state["diagnostico_ahp"]
    matriz_norm = st.session_state["matriz_norm_ahp"]
    contribuicoes = st.session_state["contribuicoes_ahp"]
    pesos = st.session_state["pesos_ahp"]

    st.subheader("🏆 Resultado da Decisão")

    melhor = resultado.iloc[0]
    segundo = resultado.iloc[1] if len(resultado) > 1 else None

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Produto recomendado", melhor["Marca"])
    col2.metric("Pontuação AHP", f"{melhor['Pontuação AHP Gaussiano']:.4f}")
    col3.metric("Nota do Produto", f"{melhor['Pontuação']:.2f}")
    col4.metric("Ranking", f"{melhor['Ranking']}º")

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        col5.metric("Vantagem sobre 2º", f"{diff:.4f}")
    else:
        col5.metric("Vantagem sobre 2º", "-")

    st.success(
        f"✅ Pelo método AHP Gaussiano, o produto mais indicado é "
        f"**{melhor['Marca']}**, com pontuação final de "
        f"**{melhor['Pontuação AHP Gaussiano']:.4f}**."
    )

    crit_principal = diagnostico.loc[0, "Critério"]
    peso_principal = diagnostico.loc[0, "Peso Calculado (%)"]

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]

        if diff < 0.02:
            intensidade = "muito pequena"
            recomendacao = "⚠️ Recomenda-se revisar os critérios ou incluir novos dados."
        elif diff < 0.08:
            intensidade = "moderada"
            recomendacao = "✅ Existe vantagem moderada para o primeiro colocado."
        else:
            intensidade = "alta"
            recomendacao = "✅ Existe vantagem clara para o primeiro colocado."

        st.markdown(f"""
        **📚 Interpretação Acadêmica**

        O produto **{melhor['Marca']}** ficou à frente de **{segundo['Marca']}**
        por uma diferença **{intensidade}** de **{diff:.4f}** ponto(s).

        O critério com maior influência foi **{crit_principal}**, com peso objetivo de
        **{peso_principal:.2f}%**.

        {recomendacao}
        """)

    st.subheader("📊 Ranking Final")

    cols_exib = [
        "Ranking",
        "Marca",
        "Classificação",
        "Pontuação AHP Gaussiano"
    ] + criterios

    st.dataframe(
        resultado[cols_exib].round(4),
        use_container_width=True
    )

    fig_rank = px.bar(
        resultado.sort_values("Pontuação AHP Gaussiano"),
        x="Marca",
        y="Pontuação AHP Gaussiano",
        text_auto=".4f",
        title="Pontuação Final por Produto",
        color="Pontuação AHP Gaussiano",
        color_continuous_scale="Viridis"
    )

    fig_rank.update_layout(
        showlegend=False,
        yaxis_title="Pontuação AHP",
        xaxis_title="Produto"
    )

    st.plotly_chart(fig_rank, use_container_width=True)


    # =====================================================
    # PESOS
    # =====================================================
    st.subheader("⚖️ Pesos Calculados dos Critérios")

    st.dataframe(
        diagnostico.round(4),
        use_container_width=True
    )

    fig_pesos = px.bar(
        diagnostico,
        x="Critério",
        y="Peso Calculado (%)",
        text_auto=".1f",
        title="Peso Objetivo de Cada Critério",
        color="Peso Calculado (%)",
        color_continuous_scale="Blues"
    )

    fig_pesos.update_layout(
        showlegend=False,
        yaxis_title="Peso (%)"
    )

    st.plotly_chart(fig_pesos, use_container_width=True)

    cv_valor = diagnostico.loc[0, "Fator Gaussiano (CV)"]

    st.markdown(
        f"> **🔍 Leitura metodológica:** o critério **{crit_principal}** recebeu o maior peso "
        f"porque apresentou maior coeficiente de variação "
        f"**σ/μ = {cv_valor:.4f}**, indicando maior poder de discriminação entre os produtos."
    )


    # =====================================================
    # MATRIZ NORMALIZADA
    # =====================================================
    with st.expander("📐 Ver Matriz Normalizada"):
        df_norm_exib = matriz_norm.copy()
        df_norm_exib.insert(0, "Marca", df["Marca"].values)

        st.dataframe(
            df_norm_exib.round(4),
            use_container_width=True
        )

        st.caption("Normalização por soma: rᵢⱼ = xᵢⱼ / Σxᵢⱼ")


    # =====================================================
    # CONTRIBUIÇÕES
    # =====================================================
    st.subheader("🧩 Contribuição de Cada Critério na Pontuação")

    contrib_exib = contribuicoes.copy()
    contrib_exib["Pontuação Total"] = contrib_exib[criterios].sum(axis=1)

    st.dataframe(
        contrib_exib.round(4),
        use_container_width=True
    )

    fig_contrib = px.bar(
        contrib_exib.sort_values("Pontuação Total"),
        x="Marca",
        y=criterios,
        title="Composição da Pontuação Final por Critério",
        barmode="stack"
    )

    st.plotly_chart(fig_contrib, use_container_width=True)


    # =====================================================
    # ANÁLISE DA PONTUAÇÃO
    # =====================================================
    st.subheader("⭐ Análise da Coluna Pontuação")

    peso_pontuacao = pesos["Pontuação"] * 100
    nota_media = resultado["Pontuação"].mean()
    melhor_nota = resultado.loc[resultado["Pontuação"].idxmax()]

    colp1, colp2, colp3 = st.columns(3)

    colp1.metric("Peso da Pontuação", f"{peso_pontuacao:.2f}%")
    colp2.metric("Nota média dos produtos", f"{nota_media:.2f}")
    colp3.metric("Maior nota", f"{melhor_nota['Pontuação']:.2f}")

    st.markdown(f"""
    A coluna **Pontuação** foi tratada como critério qualitativo/estratégico.
    Ela representa avaliação, reputação ou percepção de qualidade do produto.

    Produto com maior nota: **{melhor_nota['Marca']}**, com **{melhor_nota['Pontuação']:.2f}**.

    Peso calculado da Pontuação no modelo: **{peso_pontuacao:.2f}%**.
    """)

    if peso_pontuacao >= 30:
        st.info(
            "A Pontuação teve influência alta na decisão. "
            "Isso indica que a diferença de avaliação entre os produtos foi relevante."
        )
    elif peso_pontuacao >= 15:
        st.info(
            "A Pontuação teve influência moderada. "
            "Ela ajudou na decisão, mas não dominou o resultado."
        )
    else:
        st.info(
            "A Pontuação teve baixa influência. "
            "As notas dos produtos estão próximas ou outros critérios discriminaram melhor."
        )


    # =====================================================
    # POR QUE VENCEU
    # =====================================================
    st.subheader("🔎 Por que este produto venceu?")

    contrib_vencedor = contrib_exib[
        contrib_exib["Marca"] == melhor["Marca"]
    ].iloc[0]

    top_contrib = contrib_vencedor[criterios].sort_values(ascending=False)

    st.markdown(f"""
    **{melhor['Marca']}** venceu porque apresentou o melhor equilíbrio ponderado entre os critérios.

    | Critério | Contribuição | Peso do Critério |
    |---|---:|---:|
    | **{top_contrib.index[0]}** | {top_contrib.iloc[0]:.4f} | {pesos[top_contrib.index[0]] * 100:.2f}% |
    | **{top_contrib.index[1]}** | {top_contrib.iloc[1]:.4f} | {pesos[top_contrib.index[1]] * 100:.2f}% |
    | **{top_contrib.index[2]}** | {top_contrib.iloc[2]:.4f} | {pesos[top_contrib.index[2]] * 100:.2f}% |
    | **{top_contrib.index[3]}** | {top_contrib.iloc[3]:.4f} | {pesos[top_contrib.index[3]] * 100:.2f}% |

    Um produto não precisa ser o melhor em todos os critérios.
    Ele precisa ser competitivo nos critérios que mais diferenciam as alternativas.
    """)


    # =====================================================
    # ALERTAS
    # =====================================================
    st.subheader("⚠️ Alertas de Qualidade dos Dados")

    alertas = []

    for crit in criterios:
        if df[crit].nunique() == 1:
            alertas.append(
                f"`{crit}` possui o mesmo valor para todos os produtos e não discrimina alternativas."
            )

    pesos_array = np.array(list(pesos.values))

    if pesos_array.max() > 0.70:
        crit_max = pesos.idxmax()
        alertas.append(
            f"O critério `{crit_max}` concentra {pesos_array.max() * 100:.2f}% do peso. "
            "A decisão está muito dependente desse critério."
        )

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]

        if diff < 0.02:
            alertas.append(
                "Diferença muito pequena entre 1º e 2º colocado. "
                "Pode ser necessário incluir novos critérios."
            )

    if resultado["Pontuação"].min() < 3:
        alertas.append(
            "Existe produto com Pontuação abaixo de 3. "
            "Avalie se ele deveria continuar na análise."
        )

    if alertas:
        for alerta in alertas:
            st.warning(f"⚠️ {alerta}")
    else:
        st.info("✅ Nenhum alerta crítico encontrado.")


    # =====================================================
    # METODOLOGIA
    # =====================================================
    with st.expander("🧮 Como funciona o AHP Gaussiano"):
        st.markdown("""
        ## Método AHP Gaussiano

        O AHP Gaussiano calcula pesos de forma objetiva usando a variabilidade dos dados.

        ### Etapas

        **1. Definição dos critérios**

        São definidos critérios de decisão, como:

        - Custo
        - Quantidade
        - Benefício Extra
        - Pontuação

        **2. Direção de preferência**

        Cada critério pode ser configurado como:

        - Maior é melhor
        - Menor é melhor

        Critérios do tipo “menor é melhor” são invertidos usando:

        ```
        x' = 1 / x
        ```

        **3. Normalização**

        Os dados são normalizados por soma:

        ```
        rᵢⱼ = xᵢⱼ / Σxᵢⱼ
        ```

        **4. Fator Gaussiano**

        Para cada critério, calcula-se:

        ```
        FGⱼ = σⱼ / μⱼ
        ```

        Onde:

        - σ = desvio-padrão
        - μ = média

        **5. Peso do critério**

        ```
        wⱼ = FGⱼ / ΣFGⱼ
        ```

        **6. Pontuação final**

        ```
        Sᵢ = Σ(rᵢⱼ × wⱼ)
        ```

        O produto com maior `Sᵢ` é o mais recomendado.
        """)

else:
    st.info("👆 Clique em **Calcular AHP Gaussiano** para visualizar os resultados.")


# =====================================================
# RODAPÉ
# =====================================================
st.markdown("---")

st.caption(
    "🎓 Aplicação acadêmica | Método AHP Gaussiano | Decisão multicritério com Pontuação integrada"
)