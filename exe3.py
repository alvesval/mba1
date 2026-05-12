import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="AHP Gaussiano - Escolha de Veículo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🚗 Método AHP Gaussiano - Decisão de Compra de Veículo")
st.markdown(
    """
    Esta aplicação implementa o **Método AHP Gaussiano** conforme proposto pelo **Prof. Dr. Marcos Santos**.
    Os pesos dos critérios são determinados objetivamente pela dispersão dos dados (Coeficiente de Variação),
    eliminando a subjetividade das matrizes de comparação par a par do AHP tradicional.
    """
)


# =====================================================
# 1. BASE PADRÃO - TABELA DE VEÍCULOS
# =====================================================
def carregar_dados_padrao():
    return pd.DataFrame({
        "Produto": ["1", "2", "3", "4"],
        "Marca": ["Honda Civic", "Toyota Corolla", "Volkswagen Jetta", "Ford Focus"],
        "Preço": [120, 115, 130, 110],
        "Consumo": [14.5, 15.2, 13.8, 14.0],
        "Segurança": [8.5, 9.0, 8.8, 8.2],
        "Conforto": [7.8, 8.2, 8.5, 7.5],
        "Potência": [143, 132, 150, 125]
    })


if "dados_produtos" not in st.session_state:
    st.session_state["dados_produtos"] = carregar_dados_padrao()

# =====================================================
# 2. CONTROLES E MATRIZ DE DECISÃO
# =====================================================
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🔄 Restaurar matriz padrão"):
        st.session_state["dados_produtos"] = carregar_dados_padrao()
        st.rerun()
with col_b:
    st.info(
        "Edite os dados, inclua novos veículos e ajuste a direção de preferência de cada critério. "
        "Valores numéricos podem usar ponto (.) ou vírgula (,) como separador decimal."
    )

st.subheader("📋 Matriz de Decisão")
df_editado = st.data_editor(
    st.session_state["dados_produtos"].astype(str),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Produto": st.column_config.TextColumn("Produto", disabled=True),
        "Marca": st.column_config.TextColumn("Marca", required=True),
        "Preço": st.column_config.NumberColumn("Preço (R$ mil)", format="%.2f"),
        "Consumo": st.column_config.NumberColumn("Consumo (km/l)", format="%.1f"),
        "Segurança": st.column_config.NumberColumn("Segurança (nota)", format="%.1f"),
        "Conforto": st.column_config.NumberColumn("Conforto (nota)", format="%.1f"),
        "Potência": st.column_config.NumberColumn("Potência (cv)", format="%.0f"),
    },
    key="editor_produtos"
)
st.session_state["dados_produtos"] = df_editado.copy()


# =====================================================
# 3. TRATAMENTO DOS DADOS
# =====================================================
def converter_numero(valor):
    """Converte strings para float tratando formatos BR/US."""
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

# ✅ CRITÉRIOS ATUALIZADOS PARA VEÍCULOS
criterios = ["Preço", "Consumo", "Segurança", "Conforto", "Potência"]
for criterio in criterios:
    df[criterio] = df[criterio].apply(converter_numero)

linhas_invalidas = df[df[criterios].isna().any(axis=1)]
if not linhas_invalidas.empty:
    st.warning("⚠️ Algumas linhas possuem valores numéricos inválidos ou vazios e serão ignoradas.")
    st.dataframe(linhas_invalidas[["Produto", "Marca"] + criterios], use_container_width=True)

df = df.dropna(subset=criterios).reset_index(drop=True)
if df.empty:
    st.error("❌ Nenhum veículo válido encontrado. Preencha a matriz.")
    st.stop()
if len(df) < 2:
    st.warning("⚠️ O AHP Gaussiano exige pelo menos dois veículos para calcular a dispersão dos critérios.")
    st.stop()

# =====================================================
# 4. CONFIGURAÇÃO DOS CRITÉRIOS - ATUALIZADO
# =====================================================
st.subheader("⚙️ Configuração dos Critérios")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    tipo_preco = st.selectbox("Preço", ["Menor é melhor", "Maior é melhor"], index=0)  # Custo: menor melhor
with col2:
    tipo_consumo = st.selectbox("Consumo (km/l)", ["Maior é melhor", "Menor é melhor"], index=0)  # Benefício
with col3:
    tipo_seguranca = st.selectbox("Segurança", ["Maior é melhor", "Menor é melhor"], index=0)
with col4:
    tipo_conforto = st.selectbox("Conforto", ["Maior é melhor", "Menor é melhor"], index=0)
with col5:
    tipo_potencia = st.selectbox("Potência", ["Maior é melhor", "Menor é melhor"], index=0)

tipo_criterio = {
    "Preço": tipo_preco,
    "Consumo": tipo_consumo,
    "Segurança": tipo_seguranca,
    "Conforto": tipo_conforto,
    "Potência": tipo_potencia,
}


# =====================================================
# 5. IMPLEMENTAÇÃO DO AHP GAUSSIANO (Método Completo)
# =====================================================
def calcular_ahp_gaussiano(df, criterios, tipo_criterio, id_col="Marca"):
    """
    Implementação rigorosa do AHP-Gaussiano (Santos et al.)
    """
    # === ETAPA 1: Pré-processamento (inversão de custos) ===
    matriz = df[criterios].copy().astype(float)
    for col in criterios:
        if "Menor" in tipo_criterio[col]:
            vals = matriz[col].replace(0, np.nan)
            matriz[col] = 1.0 / vals
            matriz[col] = matriz[col].fillna(matriz[col].max())

    # === ETAPA 2: Normalização por Soma ===
    matriz_norm = pd.DataFrame(index=matriz.index)
    for col in criterios:
        soma = matriz[col].sum()
        if soma > 0:
            matriz_norm[col] = matriz[col] / soma
        else:
            matriz_norm[col] = 1.0 / len(matriz)

    # === ETAPA 3: Estatísticas Descritivas ===
    medias = matriz_norm[criterios].mean()
    desvios = matriz_norm[criterios].std(ddof=0)

    # === ETAPA 4: Fator Gaussiano (Coeficiente de Variação) ===
    fator_gaussiano = (desvios / medias.replace(0, np.nan)).fillna(0)

    # === ETAPA 5: Ponderação Objetiva ===
    if fator_gaussiano.sum() > 0:
        pesos = fator_gaussiano / fator_gaussiano.sum()
    else:
        pesos = pd.Series([1 / len(criterios)] * len(criterios), index=criterios)

    # === ETAPA 6: Pontuação Final ===
    scores = matriz_norm[criterios].dot(pesos)

    # === ETAPA 7: Preparar Resultados ===
    resultado = df.copy()
    resultado["Pontuação AHP Gaussiano"] = scores.values
    resultado["Ranking"] = (
        resultado["Pontuação AHP Gaussiano"]
            .rank(ascending=False, method="dense")
            .astype(int)
    )

    diagnostico = pd.DataFrame({
        "Critério": criterios,
        "Tipo": [tipo_criterio[c] for c in criterios],
        "Média Normalizada": medias.values,
        "Desvio-Padrão (σ)": desvios.values,
        "Fator Gaussiano (CV)": fator_gaussiano.values,
        "Peso Calculado (%)": (pesos.values * 100).round(2),
    }).sort_values("Peso Calculado (%)", ascending=False).reset_index(drop=True)

    contribuicoes = matriz_norm[criterios].mul(pesos, axis=1)
    contribuicoes.insert(0, id_col, df[id_col].values)
    resultado = resultado.sort_values("Pontuação AHP Gaussiano", ascending=False).reset_index(drop=True)

    return matriz_norm, pesos, resultado, diagnostico, contribuicoes


# =====================================================
# 6. EXECUÇÃO DO CÁLCULO
# =====================================================
st.subheader("🚀 Executar Cálculo")
if st.button("Calcular AHP Gaussiano", type="primary"):
    with st.spinner("Processando método AHP Gaussiano..."):
        try:
            matriz_norm, pesos, resultado, diagnostico, contribuicoes = calcular_ahp_gaussiano(
                df, criterios, tipo_criterio
            )
            st.success("✅ Cálculo concluído com sucesso!")

            st.session_state.update({
                "resultado_ahp": resultado,
                "diagnostico_ahp": diagnostico,
                "matriz_norm_ahp": matriz_norm,
                "contribuicoes_ahp": contribuicoes,
                "pesos_ahp": pesos
            })
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erro: {type(e).__name__}: {e}")
            with st.expander("🔍 Detalhes do erro"):
                st.exception(e)

# =====================================================
# 7. EXIBIÇÃO DOS RESULTADOS
# =====================================================
if "resultado_ahp" in st.session_state:
    resultado = st.session_state["resultado_ahp"]
    diagnostico = st.session_state["diagnostico_ahp"]
    matriz_norm = st.session_state["matriz_norm_ahp"]
    contribuicoes = st.session_state["contribuicoes_ahp"]
    pesos = st.session_state["pesos_ahp"]

    # 🏆 RESULTADO PRINCIPAL
    st.subheader("🏆 Veículo Recomendado")
    melhor = resultado.iloc[0]
    segundo = resultado.iloc[1] if len(resultado) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Veículo recomendado", melhor["Marca"])
    col2.metric("Pontuação", f"{melhor['Pontuação AHP Gaussiano']:.4f}")
    col3.metric("Ranking", f"{melhor['Ranking']}º")

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        col4.metric("Vantagem sobre 2º", f"{diff:.4f}")
    else:
        col4.metric("Vantagem sobre 2º", "-")

    st.success(
        f"✅ Pelo método AHP Gaussiano, o veículo mais indicado é **{melhor['Marca']}**, "
        f"com pontuação final de **{melhor['Pontuação AHP Gaussiano']:.4f}**."
    )

    # Interpretação acadêmica
    crit_principal = diagnostico.loc[0, "Critério"]
    peso_principal = diagnostico.loc[0, "Peso Calculado (%)"] / 100

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        if diff < 0.02:
            intensidade, rec = "muito pequena", "⚠️ Revise dados ou inclua novos critérios."
        elif diff < 0.08:
            intensidade, rec = "moderada", "✅ Vantagem moderada."
        else:
            intensidade, rec = "alta", "✅ Vantagem clara e robusta."

        st.markdown(
            f"""
            **📚 Interpretação:**  
            **{melhor['Marca']}** ficou à frente de **{segundo['Marca']}** por diferença **{intensidade}** 
            de **{diff:.4f}** ponto(s). O critério mais influente foi **{crit_principal}** 
            com peso de **{peso_principal:.2%}**. {rec}
            """
        )

    # 📊 RANKING FINAL
    st.subheader("📊 Ranking Final dos Veículos")
    cols_exib = ["Marca", "Pontuação AHP Gaussiano", "Ranking"] + criterios
    st.dataframe(resultado[cols_exib].round(4), use_container_width=True)

    fig_rank = px.bar(
        resultado.sort_values("Pontuação AHP Gaussiano"),
        x="Marca", y="Pontuação AHP Gaussiano",
        text_auto=".4f", title="Pontuação Final por Veículo",
        color="Pontuação AHP Gaussiano", color_continuous_scale="Viridis"
    )
    fig_rank.update_layout(showlegend=False, yaxis_title="Pontuação")
    st.plotly_chart(fig_rank, use_container_width=True)

    # ⚖️ PESOS DOS CRITÉRIOS
    st.subheader("⚖️ Pesos Calculados (Fator Gaussiano)")
    st.dataframe(diagnostico.round(4), use_container_width=True)

    fig_pesos = px.bar(
        diagnostico, x="Critério", y="Peso Calculado (%)",
        text_auto=".1f%", title="Influência Objetiva de Cada Critério",
        color="Peso Calculado (%)", color_continuous_scale="Blues"
    )
    fig_pesos.update_layout(showlegend=False, yaxis_title="Peso (%)")
    st.plotly_chart(fig_pesos, use_container_width=True)

    # Nota metodológica
    crit_principal = diagnostico.loc[0, "Critério"]
    cv_valor = diagnostico.loc[0, "Fator Gaussiano (CV)"]
    st.markdown(
        f"> **🔍 Leitura:** O critério **{crit_principal}** recebeu maior peso "
        f"por apresentar maior Coeficiente de Variação (σ/μ = {cv_valor:.4f}), "
        f"indicando maior poder de discriminação entre os veículos."
    )

    # 📐 MATRIZ NORMALIZADA
    with st.expander("📐 Ver Matriz Normalizada"):
        df_norm_exib = matriz_norm.copy()
        df_norm_exib.insert(0, "Marca", df["Marca"].values)
        st.dataframe(df_norm_exib.round(4), use_container_width=True)
        st.caption("Valores normalizados: rᵢⱼ = xᵢⱼ / Σxᵢⱼ")

    # 🧩 CONTRIBUIÇÕES POR CRITÉRIO
    st.subheader("🧩 Contribuição de Cada Critério na Pontuação")
    contrib_exib = contribuicoes.copy()
    contrib_exib["Pontuação Total"] = contrib_exib[criterios].sum(axis=1).round(4)
    st.dataframe(contrib_exib.round(4), use_container_width=True)

    fig_contrib = px.bar(
        contrib_exib.sort_values("Pontuação Total"),
        x="Marca", y=criterios,
        title="Composição da Pontuação por Critério",
        barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    # 🔎 ANÁLISE DO VENCEDOR
    st.subheader("🔎 Por que este veículo venceu?")
    contrib_vencedor = contrib_exib[contrib_exib["Marca"] == melhor["Marca"]].iloc[0]
    top_contrib = contrib_vencedor[criterios].sort_values(ascending=False)

    st.markdown(
        f"""
        **{melhor['Marca']}** obteve a melhor pontuação pelo equilíbrio entre:

        | Critério | Contribuição | Peso do Critério |
        |----------|-------------|-----------------|
        | **{top_contrib.index[0]}** | {top_contrib.iloc[0]:.4f} | {pesos[top_contrib.index[0]] * 100:.1f}% |
        | **{top_contrib.index[1]}** | {top_contrib.iloc[1]:.4f} | {pesos[top_contrib.index[1]] * 100:.1f}% |
        | **{top_contrib.index[2]}** | {top_contrib.iloc[2]:.4f} | {pesos[top_contrib.index[2]] * 100:.1f}% |

        💡 *Um veículo pode vencer sem ser o melhor em tudo, desde que tenha equilíbrio nos critérios mais discriminantes.*
        """
    )

    # ⚠️ ALERTAS METODOLÓGICOS
    st.subheader("⚠️ Alertas de Qualidade dos Dados")
    alertas = []
    for crit in criterios:
        if df[crit].nunique() == 1:
            alertas.append(f"`{crit}` tem valor constante → não discrimina alternativas.")

    pesos_array = np.array(list(pesos.values))
    if pesos_array.max() > 0.70:
        crit_max = pesos.idxmax()
        alertas.append(
            f"`{crit_max}` concentra {pesos_array.max() * 100:.1f}% do peso → decisão muito dependente dele.")

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        if diff < 0.02:
            alertas.append("Diferença <0.02 entre 1º e 2º → considere revisar critérios.")

    if alertas:
        for alerta in alertas:
            st.warning(f"⚠️ {alerta}")
    else:
        st.info("✅ Dados adequados para aplicação do método AHP Gaussiano.")

    # 🧮 EXPLICAÇÃO METODOLÓGICA
    with st.expander("🧮 Como funciona o AHP Gaussiano"):
        st.markdown("""
        ## 📚 Fundamentação Teórica

        O **AHP Gaussiano** (Santos et al.) calcula pesos **objetivamente** pelo **Coeficiente de Variação** (CV = σ/μ).

        ### 📐 Etapas de Cálculo
        1. **Pré-processamento**: Inverte critérios de custo (1/x)
        2. **Normalização**: rᵢⱼ = xᵢⱼ / Σxᵢⱼ → escala [0,1]
        3. **Estatísticas**: Média (μ) e desvio-padrão (σ) por critério
        4. **Fator Gaussiano**: FGⱼ = σⱼ/μⱼ → poder de discriminação
        5. **Ponderação**: wⱼ = FGⱼ/ΣFGₖ → pesos normalizados
        6. **Pontuação Final**: Sᵢ = Σ(rᵢⱼ × wⱼ)

        ### ✅ Vantagens
        | AHP Clássico | AHP Gaussiano |
        |-------------|---------------|
        | Pesos subjetivos | Pesos objetivos pelos dados |
        | Risco de inconsistência | Consistência matemática |
        | Matriz par a par | Usa matriz de decisão direta |

        *Referência: Método para decisão multicritério objetiva.*
        """)

else:
    st.info("👆 Clique em **Calcular AHP Gaussiano** para ver os resultados.")

# =====================================================
# RODAPÉ
# =====================================================
st.markdown("---")
st.caption(
    "🎓 Aplicação acadêmica | Método AHP Gaussiano (Santos et al.) | "
    "Implementação manual validada"
)