# app.py
import math
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Simulador Algébrico de Reposição",
    page_icon="📦",
    layout="wide"
)


# =========================
# Funções auxiliares
# =========================

def fmt_moeda(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def matriz_latex_coluna(nome, vetor):
    valores = r" \\ ".join([str(v).replace(".", ",") for v in vetor])
    return rf"""
    {nome} =
    \begin{{bmatrix}}
    {valores}
    \end{{bmatrix}}
    """


def matriz_latex(nome, matriz):
    linhas = []
    for linha in matriz:
        linhas.append(" & ".join([str(v).replace(".", ",") for v in linha]))
    conteudo = r" \\ ".join(linhas)
    return rf"""
    {nome} =
    \begin{{bmatrix}}
    {conteudo}
    \end{{bmatrix}}
    """


def calcular_modelo(produtos, demanda, picking, pulmao, norma, tempo, valor):
    D = np.array(demanda, dtype=float)
    P = np.array(picking, dtype=float)
    E = np.array(pulmao, dtype=float)
    N = np.array(norma, dtype=float)
    T = np.array(tempo, dtype=float)
    V = np.array(valor, dtype=float)

    # R = D - P
    R = D - P

    # R+ = max(D-P, 0)
    R_pos = np.maximum(R, 0)

    # Q = min(R+, E)
    Q = np.minimum(R_pos, E)

    # Corte estimado: o que ainda falta depois da reposição possível
    corte = np.maximum(R_pos - Q, 0)

    # Movimentações / pallets
    M = np.array([
        math.ceil(q / n) if q > 0 and n > 0 else 0
        for q, n in zip(Q, N)
    ], dtype=float)

    # Tempo total
    TT = M * T

    # Valor financeiro em risco
    F = R_pos * V

    # Valor possível de reposição
    valor_reposto = Q * V

    # Valor de corte estimado
    valor_corte = corte * V

    # Score de prioridade
    alpha = 1
    beta = 0.1
    gamma = 1

    score_base = (alpha * R_pos) + (beta * F) - (gamma * T)

    # Penalização para produto sem pulmão e com necessidade
    score = np.where((R_pos > 0) & (E == 0), score_base + 999, score_base)

    df = pd.DataFrame({
        "Produto": produtos,
        "Demanda": D.astype(int),
        "Picking": P.astype(int),
        "Pulmão": E.astype(int),
        "Norma": N.astype(int),
        "Tempo": T.astype(int),
        "Valor Unitário": V,
        "Necessidade R+": R_pos.astype(int),
        "Reposição Possível Q": Q.astype(int),
        "Corte Estimado": corte.astype(int),
        "Movimentações M": M.astype(int),
        "Tempo Total TT": TT.astype(int),
        "Valor em Risco": F,
        "Valor Reposto": valor_reposto,
        "Valor Corte": valor_corte,
        "Score Prioridade": score,
    })

    df["Risco de Corte"] = np.where(df["Corte Estimado"] > 0, "Sim", "Não")

    df_prioridade = df.sort_values(
        by=["Score Prioridade", "Valor Corte", "Necessidade R+"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return {
        "D": D,
        "P": P,
        "E": E,
        "N": N,
        "T": T,
        "V": V,
        "R": R,
        "R_pos": R_pos,
        "Q": Q,
        "M": M,
        "TT": TT,
        "F": F,
        "corte": corte,
        "df": df,
        "df_prioridade": df_prioridade,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


# =========================
# Interface
# =========================

st.title("📦 Simulador Algébrico para Redução de Corte no Picking")
st.caption(
    "Modelo baseado em vetores, matrizes e score de prioridade para reposição preventiva em centro de distribuição."
)

produtos = ["A", "B", "C", "D"]

with st.sidebar:
    st.header("⚙️ Parâmetros da Simulação")

    st.markdown("### Quantidade vendida / demanda")
    demanda = []
    demanda.append(st.number_input("Produto A", min_value=0, value=120, step=1))
    demanda.append(st.number_input("Produto B", min_value=0, value=50, step=1))
    demanda.append(st.number_input("Produto C", min_value=0, value=45, step=1))
    demanda.append(st.number_input("Produto D", min_value=0, value=63, step=1))

    st.markdown("---")
    gerar_random = st.button("🎲 Gerar Picking e Pulmão Aleatórios", use_container_width=True)

    if "picking" not in st.session_state or gerar_random:
        st.session_state.picking = [
            random.randint(0, max(10, int(d * 0.8))) for d in demanda
        ]

    if "pulmao" not in st.session_state or gerar_random:
        st.session_state.pulmao = [
            random.randint(0, max(20, int(d * 2))) for d in demanda
        ]

    st.markdown("### Picking gerado")
    picking = []
    for i, p in enumerate(produtos):
        picking.append(
            st.number_input(
                f"Picking Produto {p}",
                min_value=0,
                value=int(st.session_state.picking[i]),
                step=1
            )
        )

    st.markdown("### Pulmão gerado")
    pulmao = []
    for i, p in enumerate(produtos):
        pulmao.append(
            st.number_input(
                f"Pulmão Produto {p}",
                min_value=0,
                value=int(st.session_state.pulmao[i]),
                step=1
            )
        )

    st.markdown("---")
    st.markdown("### Dados fixos do modelo")

    norma = [50, 15, 12, 40]
    tempo = [10, 5, 6, 20]
    valor = [2.54, 31.40, 12.45, 5.69]

    st.write("Norma:", norma)
    st.write("Tempo:", tempo)
    st.write("Valor:", valor)


resultado = calcular_modelo(
    produtos=produtos,
    demanda=demanda,
    picking=picking,
    pulmao=pulmao,
    norma=norma,
    tempo=tempo,
    valor=valor,
)

df = resultado["df"]
df_prioridade = resultado["df_prioridade"]


# =========================
# KPIs
# =========================

total_demanda = int(df["Demanda"].sum())
total_picking = int(df["Picking"].sum())
total_repor = int(df["Necessidade R+"].sum())
total_corte = int(df["Corte Estimado"].sum())
valor_corte = df["Valor Corte"].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Demanda Total", total_demanda)
col2.metric("Estoque Picking", total_picking)
col3.metric("Necessidade Reposição", total_repor)
col4.metric("Corte Estimado", total_corte)
col5.metric("Valor Corte", fmt_moeda(valor_corte))


# =========================
# Abas
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dados",
    "🧮 Vetores e Matrizes",
    "📦 Reposição",
    "🚨 Prioridade",
    "📈 Gráficos"
])


with tab1:
    st.subheader("Dados Operacionais da Carga")

    df_view = df.copy()
    for col in ["Valor Unitário", "Valor em Risco", "Valor Reposto", "Valor Corte", "Score Prioridade"]:
        df_view[col] = df_view[col].round(2)

    st.dataframe(df_view, use_container_width=True)

    st.markdown("### Matriz geral do problema")

    matriz_geral = []
    for i in range(len(produtos)):
        matriz_geral.append([
            int(resultado["D"][i]),
            int(resultado["P"][i]),
            int(resultado["E"][i]),
            int(resultado["N"][i]),
            int(resultado["T"][i]),
            round(float(resultado["V"][i]), 2),
        ])

    st.latex(matriz_latex("A", matriz_geral))

    st.markdown("""
    Onde cada coluna representa:

    \[
    A = [Demanda \\quad Picking \\quad Pulmão \\quad Norma \\quad Tempo \\quad Valor]
    \]
    """)


with tab2:
    st.subheader("Vetores utilizados no modelo")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Vetor de Demanda")
        st.latex(matriz_latex_coluna("D", resultado["D"].astype(int)))

        st.markdown("#### Vetor de Picking")
        st.latex(matriz_latex_coluna("P", resultado["P"].astype(int)))

    with c2:
        st.markdown("#### Vetor de Pulmão")
        st.latex(matriz_latex_coluna("E", resultado["E"].astype(int)))

        st.markdown("#### Vetor de Norma")
        st.latex(matriz_latex_coluna("N", resultado["N"].astype(int)))

    with c3:
        st.markdown("#### Vetor de Tempo")
        st.latex(matriz_latex_coluna("T", resultado["T"].astype(int)))

        st.markdown("#### Vetor de Valor")
        st.latex(matriz_latex_coluna("V", [round(v, 2) for v in resultado["V"]]))

    st.markdown("---")
    st.subheader("Cálculo da Necessidade de Reposição")

    st.latex(r"R = D - P")
    st.latex(matriz_latex_coluna("R", resultado["R"].astype(int)))

    st.markdown("Como não existe reposição negativa:")

    st.latex(r"R^{+} = \max(D - P, 0)")
    st.latex(matriz_latex_coluna("R^{+}", resultado["R_pos"].astype(int)))

    st.markdown("---")
    st.subheader("Matriz Diagonal de Valores")

    DV = np.diag(resultado["V"])
    DV_formatado = [[round(v, 2) for v in row] for row in DV]
    st.latex(matriz_latex("D_V", DV_formatado))

    st.markdown("A matriz diagonal permite multiplicar cada produto pelo seu próprio valor unitário.")
    st.latex(r"F = R^{+} \circ V")


with tab3:
    st.subheader("Reposição possível e movimentações")

    st.markdown("A quantidade possível de reposição é dada por:")

    st.latex(r"Q = \min(R^{+}, E)")
    st.latex(matriz_latex_coluna("Q", resultado["Q"].astype(int)))

    st.markdown("A quantidade de movimentações é calculada pela norma de paletização:")

    st.latex(r"M = \left\lceil \frac{Q}{N} \right\rceil")
    st.latex(matriz_latex_coluna("M", resultado["M"].astype(int)))

    st.markdown("O tempo total é calculado pela multiplicação elemento a elemento:")

    st.latex(r"TT = M \circ T")
    st.latex(matriz_latex_coluna("TT", resultado["TT"].astype(int)))

    st.markdown("O valor financeiro em risco é:")

    st.latex(r"F = R^{+} \circ V")
    st.latex(matriz_latex_coluna("F", [round(v, 2) for v in resultado["F"]]))

    st.markdown("### Tabela de reposição")

    reposicao = df[[
        "Produto",
        "Necessidade R+",
        "Pulmão",
        "Reposição Possível Q",
        "Corte Estimado",
        "Risco de Corte",
        "Movimentações M",
        "Tempo Total TT",
    ]]

    st.dataframe(reposicao, use_container_width=True)


with tab4:
    st.subheader("Score de prioridade para reposição")

    alpha = resultado["alpha"]
    beta = resultado["beta"]
    gamma = resultado["gamma"]

    st.markdown("A função de priorização utilizada é:")

    st.latex(r"S_i = \alpha \cdot R_i^{+} + \beta \cdot F_i - \gamma \cdot T_i")

    st.markdown(f"""
    Onde:

    - \(S_i\) = score de prioridade do produto \(i\)
    - \(R_i^+\) = necessidade de reposição
    - \(F_i\) = valor financeiro em risco
    - \(T_i\) = tempo de reposição
    - \(\alpha = {alpha}\), \(\beta = {beta}\), \(\gamma = {gamma}\)
    """)

    prioridade_view = df_prioridade[[
        "Produto",
        "Necessidade R+",
        "Pulmão",
        "Reposição Possível Q",
        "Corte Estimado",
        "Valor em Risco",
        "Valor Corte",
        "Tempo",
        "Score Prioridade",
        "Risco de Corte",
    ]].copy()

    prioridade_view["Valor em Risco"] = prioridade_view["Valor em Risco"].round(2)
    prioridade_view["Valor Corte"] = prioridade_view["Valor Corte"].round(2)
    prioridade_view["Score Prioridade"] = prioridade_view["Score Prioridade"].round(2)

    st.dataframe(prioridade_view, use_container_width=True)

    st.markdown("### Interpretação automática")

    produto_top = df_prioridade.iloc[0]

    st.success(
        f"O produto com maior prioridade é o Produto {produto_top['Produto']}, "
        f"com score {produto_top['Score Prioridade']:.2f}."
    )

    criticos = df[df["Risco de Corte"] == "Sim"]

    if not criticos.empty:
        st.error(
            "Existem produtos com risco de corte porque a necessidade de reposição "
            "não pode ser totalmente atendida pelo estoque do pulmão."
        )
        st.dataframe(
            criticos[[
                "Produto",
                "Necessidade R+",
                "Pulmão",
                "Reposição Possível Q",
                "Corte Estimado",
                "Valor Corte",
            ]],
            use_container_width=True
        )
    else:
        st.info("Não há risco de corte considerando o estoque atual do pulmão.")


with tab5:
    st.subheader("Visualização gráfica")

    grafico_df = df.copy()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### Demanda x Picking x Pulmão")
        grafico1 = grafico_df[["Produto", "Demanda", "Picking", "Pulmão"]].set_index("Produto")
        st.bar_chart(grafico1)

    with col_g2:
        st.markdown("### Necessidade x Reposição Possível")
        grafico2 = grafico_df[["Produto", "Necessidade R+", "Reposição Possível Q", "Corte Estimado"]].set_index("Produto")
        st.bar_chart(grafico2)

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.markdown("### Score de Prioridade")
        score_df = grafico_df[["Produto", "Score Prioridade"]].set_index("Produto")
        st.bar_chart(score_df)

    with col_g4:
        st.markdown("### Valor financeiro em risco")
        risco_df = grafico_df[["Produto", "Valor em Risco", "Valor Corte"]].set_index("Produto")
        st.bar_chart(risco_df)


st.markdown("---")
st.caption(
    "Modelo acadêmico demonstrativo: utiliza álgebra linear, vetores, matrizes, matriz diagonal, "
    "produto elemento a elemento e score de prioridade para apoiar a decisão logística."
)