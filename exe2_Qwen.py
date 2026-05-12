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
        "Custo (R$)": [137.75, 104.09, 64.90, 94.91],
        "Quantidade (g)": [900, 2000, 1000, 900],
        "Benefício Extra": [1.0, 0.0, 0.0, 0.0],
        "Pontuação": [4.9, 4.7, 4.8, 5]
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
        "Edite os dados, inclua novos produtos e ajuste a direção de preferência de cada critério. "
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
        "Custo (R$)": st.column_config.NumberColumn("Custo (R$)", format="%.2f"),
        "Quantidade (g)": st.column_config.NumberColumn("Quantidade (g)", format="%.0f"),
        "Benefício Extra": st.column_config.NumberColumn("Benefício Extra", format="%.2f"),
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
# 4. CONFIGURAÇÃO DOS CRITÉRIOS
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
# 5. IMPLEMENTAÇÃO DO AHP GAUSSIANO (Método Completo)
# =====================================================
def calcular_ahp_gaussiano(df, criterios, tipo_criterio, id_col="Marca"):
    """
    Implementação rigorosa do AHP-Gaussiano (Santos et al.)

    Etapas:
    1. Pré-processamento: inverte critérios de custo (1/x)
    2. Normalização por soma: r_ij = x_ij / Σx_ij
    3. Estatísticas: média (μ) e desvio-padrão populacional (σ)
    4. Fator Gaussiano: FG_j = σ_j / μ_j (Coeficiente de Variação)
    5. Ponderação: w_j = FG_j / ΣFG_j
    6. Pontuação Final: S_i = Σ(r_ij × w_j)
    7. Ranking e diagnóstico completo
    """

    # === ETAPA 1: Pré-processamento (inversão de custos) ===
    matriz = df[criterios].copy().astype(float)
    for col in criterios:
        if "Menor" in tipo_criterio[col]:
            # Critério de custo: inverter para maximização (1/x)
            vals = matriz[col].replace(0, np.nan)
            matriz[col] = 1.0 / vals
            # Preencher NaNs com valor máximo para não penalizar
            matriz[col] = matriz[col].fillna(matriz[col].max())

    # === ETAPA 2: Normalização por Soma (padrão AHP) ===
    matriz_norm = pd.DataFrame(index=matriz.index)
    for col in criterios:
        soma = matriz[col].sum()
        if soma > 0:
            matriz_norm[col] = matriz[col] / soma
        else:
            matriz_norm[col] = 1.0 / len(matriz)  # fallback para soma zero

    # === ETAPA 3: Estatísticas Descritivas ===
    medias = matriz_norm[criterios].mean()
    desvios = matriz_norm[criterios].std(ddof=0)  # desvio-padrão populacional

    # === ETAPA 4: Fator Gaussiano (Coeficiente de Variação) ===
    # FG_j = σ_j / μ_j → mede poder de discriminação do critério
    fator_gaussiano = (desvios / medias.replace(0, np.nan)).fillna(0)

    # === ETAPA 5: Ponderação Objetiva dos Critérios ===
    # w_j = FG_j / ΣFG_j → pesos normalizados (soma = 1)
    if fator_gaussiano.sum() > 0:
        pesos = fator_gaussiano / fator_gaussiano.sum()
    else:
        # Fallback: pesos iguais se não houver variabilidade
        pesos = pd.Series([1 / len(criterios)] * len(criterios), index=criterios)

    # === ETAPA 6: Pontuação Final das Alternativas ===
    # S_i = Σ(r_ij × w_j) → combinação linear ponderada
    scores = matriz_norm[criterios].dot(pesos)

    # === ETAPA 7: Preparar Resultados ===
    resultado = df.copy()
    resultado["Pontuação AHP Gaussiano"] = scores.values
    resultado["Ranking"] = (
        resultado["Pontuação AHP Gaussiano"]
            .rank(ascending=False, method="dense")
            .astype(int)
    )

    # Diagnóstico estatístico completo
    diagnostico = pd.DataFrame({
        "Critério": criterios,
        "Tipo": [tipo_criterio[c] for c in criterios],
        "Média Normalizada": medias.values,
        "Desvio-Padrão (σ)": desvios.values,
        "Fator Gaussiano (CV)": fator_gaussiano.values,
        "Peso Calculado (%)": (pesos.values * 100).round(2),
    }).sort_values("Peso Calculado (%)", ascending=False).reset_index(drop=True)

    # Contribuições por critério (decomposição da pontuação)
    contribuicoes = matriz_norm[criterios].mul(pesos, axis=1)
    contribuicoes.insert(0, id_col, df[id_col].values)

    # Ordenar resultado final por pontuação decrescente
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

            # Armazena na sessão para persistência entre interações
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
    st.subheader("🏆 Resultado da Decisão")
    melhor = resultado.iloc[0]
    segundo = resultado.iloc[1] if len(resultado) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Produto recomendado", melhor["Marca"])
    col2.metric("Pontuação", f"{melhor['Pontuação AHP Gaussiano']:.4f}")
    col3.metric("Ranking", f"{melhor['Ranking']}º")

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        col4.metric("Vantagem sobre 2º", f"{diff:.4f}")
    else:
        col4.metric("Vantagem sobre 2º", "-")

    st.success(
        f"✅ Pelo método AHP Gaussiano, o produto mais indicado é **{melhor['Marca']}**, "
        f"com pontuação final de **{melhor['Pontuação AHP Gaussiano']:.4f}**."
    )

    # Interpretação acadêmica automática
    crit_principal = diagnostico.loc[0, "Critério"]
    peso_principal = diagnostico.loc[0, "Peso Calculado (%)"] / 100

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        if diff < 0.02:
            intensidade, rec = "muito pequena", "⚠️ Revise dados ou inclua novos critérios para maior discriminação."
        elif diff < 0.08:
            intensidade, rec = "moderada", "✅ Vantagem moderada para o primeiro colocado."
        else:
            intensidade, rec = "alta", "✅ Vantagem clara e robusta para o primeiro colocado."

        st.markdown(
            f"""
            **📚 Interpretação Acadêmica:**  
            O produto **{melhor['Marca']}** ficou à frente de **{segundo[
                'Marca']}** por uma diferença **{intensidade}** 
            de **{diff:.4f}** ponto(s). O critério que mais influenciou o cálculo foi **{crit_principal}**, 
            com peso objetivo de **{peso_principal:.2%}**. {rec}
            """
        )

    # 📊 RANKING FINAL
    st.subheader("📊 Ranking Final")
    cols_exib = ["Marca", "Pontuação AHP Gaussiano", "Ranking"] + criterios
    st.dataframe(resultado[cols_exib].round(4), use_container_width=True)

    fig_rank = px.bar(
        resultado.sort_values("Pontuação AHP Gaussiano"),
        x="Marca", y="Pontuação AHP Gaussiano",
        text_auto=".4f", title="Pontuação Final por Produto",
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

    # ✅ LINHAS CORRIGIDAS - Extraia os valores antes da f-string
    crit_principal = diagnostico.loc[0, "Critério"]
    peso_principal = diagnostico.loc[0, "Peso Calculado (%)"] / 100
    cv_valor = diagnostico.loc[0, "Fator Gaussiano (CV)"]

    st.markdown(
        f"> **🔍 Leitura Metodológica:** O critério **{crit_principal}** recebeu o maior peso "
        f"porque apresentou maior **Coeficiente de Variação** (σ/μ = {cv_valor:.4f}), "
        f"indicando maior poder de discriminação entre as alternativas."
    )

    # 📐 MATRIZ NORMALIZADA
    with st.expander("📐 Ver Matriz Normalizada"):
        df_norm_exib = matriz_norm.copy()
        df_norm_exib.insert(0, "Marca", df["Marca"].values)
        st.dataframe(df_norm_exib.round(4), use_container_width=True)
        st.caption("Valores normalizados por soma: rᵢⱼ = xᵢⱼ / Σxᵢⱼ")

    # 🧩 CONTRIBUIÇÕES POR CRITÉRIO
    st.subheader("🧩 Contribuição de Cada Critério na Pontuação")
    contrib_exib = contribuicoes.copy()
    contrib_exib["Pontuação Total"] = contrib_exib[criterios].sum(axis=1).round(4)
    st.dataframe(contrib_exib.round(4), use_container_width=True)

    fig_contrib = px.bar(
        contrib_exib.sort_values("Pontuação Total"),
        x="Marca", y=criterios,
        title="Composição da Pontuação Final por Critério",
        barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    # 🔎 ANÁLISE DO VENCEDOR
    st.subheader("🔎 Por que este produto venceu?")
    contrib_vencedor = contrib_exib[contrib_exib["Marca"] == melhor["Marca"]].iloc[0]
    top_contrib = contrib_vencedor[criterios].sort_values(ascending=False)

    st.markdown(
        f"""
        **{melhor['Marca']}** obteve a melhor pontuação pela combinação equilibrada entre:

        | Critério | Contribuição | Peso do Critério |
        |----------|-------------|-----------------|
        | **{top_contrib.index[0]}** | {top_contrib.iloc[0]:.4f} | {pesos[top_contrib.index[0]] * 100:.1f}% |
        | **{top_contrib.index[1]}** | {top_contrib.iloc[1]:.4f} | {pesos[top_contrib.index[1]] * 100:.1f}% |
        | **{top_contrib.index[2]}** | {top_contrib.iloc[2]:.4f} | {pesos[top_contrib.index[2]] * 100:.1f}% |

        💡 *Um produto pode vencer sem ser o melhor em todos os critérios, desde que apresente 
        o melhor equilíbrio nos critérios com maior poder de discriminação.*
        """
    )

    # ⚠️ ALERTAS METODOLÓGICOS
    st.subheader("⚠️ Alertas de Qualidade dos Dados")
    alertas = []

    for crit in criterios:
        if df[crit].nunique() == 1:
            alertas.append(f"`{crit}` tem valor constante para todos os produtos → não discrimina alternativas.")

    pesos_array = np.array(list(pesos.values))
    if pesos_array.max() > 0.70:
        crit_max = pesos.idxmax()
        alertas.append(
            f"O critério `{crit_max}` concentra {pesos_array.max() * 100:.1f}% do peso → decisão muito dependente dele.")

    if segundo is not None:
        diff = melhor["Pontuação AHP Gaussiano"] - segundo["Pontuação AHP Gaussiano"]
        if diff < 0.02:
            alertas.append(
                "Diferença <0.02 entre 1º e 2º colocado → considere revisar critérios ou incluir novas alternativas.")

    if alertas:
        for alerta in alertas:
            st.warning(f"⚠️ {alerta}")
    else:
        st.info("✅ Nenhum alerta crítico. Os dados estão adequados para aplicação do método AHP Gaussiano.")

    # 🧮 EXPLICAÇÃO METODOLÓGICA COMPLETA
    with st.expander("🧮 Como funciona o AHP Gaussiano (Metodologia Completa)"):
        st.markdown("""
        ## 📚 Fundamentação Teórica

        O **AHP Gaussiano** (Santos, M. et al.) é uma evolução do AHP tradicional (Saaty, 1980) 
        que substitui a subjetividade das comparações par a par pela análise estatística da 
        **dispersão natural dos dados**.

        ### 🔑 Premissa Central
        > *"Critérios com maior variabilidade entre as alternativas carregam mais informação 
        > discriminativa e, portanto, devem receber maior peso objetivo na decisão."*

        ### 📐 Etapas de Cálculo

        **1. Pré-processamento dos Critérios**
        - Critérios de benefício (maximizar): mantidos como estão
        - Critérios de custo (minimizar): invertidos via `1/x` para alinhamento de direção

        **2. Normalização por Soma**
        ```
        rᵢⱼ = xᵢⱼ / Σᵢ xᵢⱼ
        ```
        Padroniza todas as unidades para escala adimensional [0,1], permitindo comparação direta.

        **3. Estatísticas Descritivas**
        - Média: `μⱼ = Σᵢ rᵢⱼ / n`
        - Desvio-padrão populacional: `σⱼ = √[Σᵢ(rᵢⱼ - μⱼ)² / n]`

        **4. Fator Gaussiano (Coeficiente de Variação)**
        ```
        FGⱼ = σⱼ / μⱼ
        ```
        Mede o poder de discriminação relativo de cada critério. Valores altos = maior capacidade 
        de diferenciar as alternativas.

        **5. Ponderação Objetiva**
        ```
        wⱼ = FGⱼ / Σₖ FGₖ
        ```
        Normaliza os fatores para obter pesos que somam 100%, eliminando subjetividade humana.

        **6. Pontuação Final das Alternativas**
        ```
        Sᵢ = Σⱼ (rᵢⱼ × wⱼ)
        ```
        Combinação linear ponderada que gera o score final para ranking.

        ### ✅ Vantagens sobre o AHP Tradicional
        | AHP Clássico | AHP Gaussiano |
        |-------------|---------------|
        | Pesos definidos por julgamento humano | Pesos calculados objetivamente pelos dados |
        | Risco de inconsistência nas comparações | Consistência matemática garantida |
        | Requer matriz de comparação par a par | Usa diretamente a matriz de decisão |
        | Sensível a viés do decisor | Imparcial e replicável |

        *Referência: Método proposto para decisão multicritério objetiva em contextos acadêmicos e industriais.*
        """)

else:
    st.info("👆 Clique em **Calcular AHP Gaussiano** para executar o método e visualizar os resultados completos.")

# =====================================================
# RODAPÉ
# =====================================================
st.markdown("---")
st.caption(
    "🎓 Aplicação desenvolvida para fins acadêmicos | Método AHP Gaussiano (Santos et al.) | "
    "Implementação manual validada conforme literatura científica"
)