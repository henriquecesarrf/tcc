import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Análise Financeira Completa", layout="wide")
st.title("Análise Financeira de Projetos")

# --- Funções auxiliares ---


def calcular_vpl(fluxos_completos, tma):
    """
    Calcula o VPL.
    fluxos_completos = [FC0, FC1, FC2, ...] onde FC0 normalmente é o investimento inicial (negativo)
    tma é decimal (ex: 0.1 para 10%)
    """
    return sum([fc / (1 + tma) ** t for t, fc in enumerate(fluxos_completos)])


def calcular_tir(fluxos, tol=1e-6, max_iter=200):
    """
    Calcula a TIR (em %) usando busca por bissecção robusta.
    Retorna None se não for possível calcular (p.ex. todos fluxos do mesmo sinal
    ou não houver mudança de sinal numa faixa razoável).
    fluxos = [FC0, FC1, FC2, ...]
    """

    # requisito básico: deve existir ao menos um fluxo negativo e ao menos um positivo
    if not (min(fluxos) < 0 < max(fluxos)):
        return None

    def vpl(r):
        # r em decimal (ex: 0.1)
        # cuidado: r > -1
        return sum(fc / (1 + r) ** t for t, fc in enumerate(fluxos))

    # definimos uma faixa inicial ampla e tentamos encontrar mudança de sinal
    left = -0.9999  # evitar -1
    right = 10.0  # 1000% como limite inicial
    v_left = vpl(left)
    v_right = vpl(right)

    # Expande direita até encontrar mudança de sinal ou atingir limite
    expand_attempts = 0
    while v_left * v_right > 0 and expand_attempts < 60:
        right *= 2
        v_right = vpl(right)
        expand_attempts += 1

    # Se ainda não houver mudança de sinal, tenta contrair left (mais perto de -1)
    if v_left * v_right > 0:
        # tenta reduzir left (mais próximo de -0.9999 já é extremo) — se falhar, retorna None
        return None

    # Bissecção
    for _ in range(max_iter):
        mid = (left + right) / 2
        v_mid = vpl(mid)
        if abs(v_mid) < tol:
            return mid * 100  # em %
        # Decide lado
        if v_left * v_mid <= 0:
            right = mid
            v_right = v_mid
        else:
            left = mid
            v_left = v_mid

    # se não convergiu com a tolerância, retorna melhor aproximação
    return ((left + right) / 2) * 100


def calcular_payback_descontado(fluxos_completos, tma):
    """
    Calcula o Payback Descontado.
    fluxos_completos = [FC0, FC1, FC2, ...]
    tma em decimal (ex: 0.1 para 10%)
    Retorna número de anos (float) ou None se não recuperar.
    """
    acumulado = 0.0
    for t, fc in enumerate(fluxos_completos):
        # fluxo descontado para o período t
        fluxo_desc = fc / ((1 + tma) ** t)
        acumulado += fluxo_desc

        # Verifica se recuperou o investimento
        if acumulado >= 0:
            if t == 0:
                return 0.0
            acumulado_anterior = acumulado - fluxo_desc
            # Se o fluxo descontado atual é zero ou negativo, não conseguimos interpolar
            if fluxo_desc <= 0:
                return None
            restante = abs(acumulado_anterior)
            fracao_ano = restante / fluxo_desc
            # O payback ocorre entre (t-1) e t -> (t-1) + fracao_ano
            return (t - 1) + fracao_ano

    return None


# --- Entradas do usuário ---
with st.expander("📋 Dados do Projeto", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        investimento_inicial = st.number_input(
            "Investimento inicial (R$)",
            value=-100.0,
            format="%.2f",
        )
        tma_base = st.slider(
            "TMA base (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
        ) / 100

    with col2:
        num_periodos = st.number_input(
            "Número de períodos", min_value=1, max_value=50, value=3
        )

        st.markdown("**Fluxos de Caixa por Período**")
        fluxos = []
        for periodo in range(num_periodos):
            fluxo = st.number_input(
                f"Período {periodo+1} (R$)",
                value=60.0,
                format="%.2f",
                key=f"fluxo_{periodo}",
            )
            fluxos.append(fluxo)

# --- Cálculos principais ---
fluxos_com_investimento = [investimento_inicial] + fluxos
vpl_base = calcular_vpl(fluxos_com_investimento, tma_base)

# Só calcula TIR quando há investimento inicial negativo e variação de sinais
tir = None
if investimento_inicial < 0:
    tir = calcular_tir(fluxos_com_investimento)

payback = calcular_payback_descontado(fluxos_com_investimento, tma_base)

# --- Resultados ---
st.divider()
st.header("📈 Resultados Financeiros")

col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    st.metric("VPL", f"R$ {vpl_base:,.2f}")
with col_res2:
    if tir is None:
        st.metric(
            "TIR",
            "N/A",
            help="TIR não calculável - verifique se o investimento inicial é negativo e há variação nos fluxos",
        )
    else:
        st.metric("TIR", f"{tir:.2f}%")
with col_res3:
    payback_txt = f"{payback:.2f}" if payback is not None else "Não recuperado"
    st.metric("Payback Descontado", payback_txt)

# Interpretação
if vpl_base > 0:
    st.success("✅ O projeto é VIÁVEL pelo método do VPL")
else:
    st.error("❌ O projeto é INVIÁVEL pelo método do VPL")

if tir is not None:
    if tir > tma_base * 100:
        st.success(f"✅ TIR ({tir:.2f}%) superior à TMA ({tma_base*100:.2f}%)")
    else:
        st.error(f"❌ TIR ({tir:.2f}%) inferior à TMA ({tma_base*100:.2f}%)")

# --- Análise de Sensibilidade ---
st.divider()
st.header("🔍 Análise de Sensibilidade")

tab1, tab2 = st.tabs(["Variação da TMA", "Variação dos Fluxos"])

with tab1:
    st.subheader("Sensibilidade do VPL à TMA")
    tma_min = st.slider(
        "TMA mínima (%)", 0.0, 30.0, 5.0, step=0.5, key="tma_min"
    ) / 100
    tma_max = st.slider(
        "TMA máxima (%)", 0.0, 30.0, 15.0, step=0.5, key="tma_max"
    ) / 100

    if tma_max <= tma_min:
        st.warning("A TMA máxima deve ser maior que a TMA mínima.")
    else:
        tma_range = np.linspace(tma_min, tma_max, 40)
        vpls_tma = [calcular_vpl(fluxos_com_investimento, tma) for tma in tma_range]

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(tma_range * 100, vpls_tma, marker="o", linestyle="-")
        ax1.axhline(0, color="r", linestyle="--")
        ax1.set_xlabel("TMA (%)")
        ax1.set_ylabel("VPL (R$)")
        ax1.set_title("Variação do VPL com a TMA")
        ax1.grid(True)
        st.pyplot(fig1)

with tab2:
    st.subheader("Sensibilidade do VPL aos Fluxos")
    variacao = st.slider("Variação dos fluxos (%)", -50, 50, 0)

    fluxos_var = [fluxos_com_investimento[0]] + [
        f * (1 + variacao / 100) for f in fluxos
    ]
    vpl_var = calcular_vpl(fluxos_var, tma_base)

    st.metric("Novo VPL", f"R$ {vpl_var:,.2f}", delta=f"{variacao}% nos fluxos")

    variacoes = np.linspace(-0.5, 0.5, 40)
    vpls_fluxo = [
        calcular_vpl([fluxos_com_investimento[0]] + [f * (1 + v) for f in fluxos], tma_base)
        for v in variacoes
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(variacoes * 100, vpls_fluxo, marker="s", linestyle="-")
    ax2.axhline(0, color="r", linestyle="--")
    ax2.set_xlabel("Variação dos Fluxos (%)")
    ax2.set_ylabel("VPL (R$)")
    ax2.set_title("Sensibilidade do VPL aos Fluxos de Caixa")
    ax2.grid(True)
    st.pyplot(fig2)

# --- Cenários do Projeto ---
st.divider()
st.header("🧭 Análise por Cenários")

with st.expander("Configuração dos Cenários", expanded=True):
    colp, cola = st.columns(2)

    with colp:
        pess_var = st.slider(
            "Variação do cenário Pessimista (%)",
            min_value=-80, max_value=0, value=-30, step=1
        )

    with cola:
        otm_var = st.slider(
            "Variação do cenário Otimista (%)",
            min_value=0, max_value=200, value=30, step=1
        )

# --- Construção dos cenários ---
cenarios = {
    "Pessimista": [fluxos_com_investimento[0]] + [f * (1 + pess_var/100) for f in fluxos],
    "Base": fluxos_com_investimento,
    "Otimista": [fluxos_com_investimento[0]] + [f * (1 + otm_var/100) for f in fluxos],
}

resultados = {}
for nome, flx in cenarios.items():
    vpl = calcular_vpl(flx, tma_base)
    tir_val = calcular_tir(flx)
    pay = calcular_payback_descontado(flx, tma_base)

    resultados[nome] = {
        "VPL": vpl,
        "TIR": tir_val,
        "Payback": pay
    }

# --- Tabela comparativa ---
st.subheader("📊 Comparativo dos Cenários")

st.write(
    """
    A tabela abaixo mostra os principais indicadores para os três cenários:
    - **Pessimista:** fluxos reduzidos  
    - **Base:** valores originais  
    - **Otimista:** fluxos aumentados  
    """
)

import pandas as pd

df_cenarios = pd.DataFrame({
    "Cenário": resultados.keys(),
    "VPL (R$)": [resultados[c]["VPL"] for c in resultados],
    "TIR (%)": [resultados[c]["TIR"] for c in resultados],
    "Payback (anos)": [resultados[c]["Payback"] for c in resultados],
})

df_cenarios["TIR (%)"] = df_cenarios["TIR (%)"].apply(
    lambda x: f"{x:.2f}%" if x is not None else "N/A"
)
df_cenarios["Payback (anos)"] = df_cenarios["Payback (anos)"].apply(
    lambda x: f"{x:.2f}" if x is not None else "N/A"
)

st.dataframe(df_cenarios, use_container_width=True)

# --- Gráfico comparativo ---
st.subheader("📈 VPL por Cenário")

figc, axc = plt.subplots(figsize=(8,5))
axc.bar(df_cenarios["Cenário"], df_cenarios["VPL (R$)"])
axc.axhline(0, color="r", linestyle="--")
axc.set_ylabel("VPL (R$)")
axc.set_title("Comparação de VPL entre Cenários")
axc.grid(axis="y")

st.pyplot(figc)

# --- Simulação de Monte Carlo ---
st.divider()
st.header("🎲 Simulação de Monte Carlo")

with st.expander("Configuração da Simulação", expanded=True):
    colmc1, colmc2 = st.columns(2)

    with colmc1:
        iteracoes = st.number_input(
            "Número de Iterações",
            min_value=100,
            max_value=50000,
            value=5000,
            step=500
        )

    with colmc2:
        volatilidade = st.slider(
            "Volatilidade dos Fluxos (%)",
            min_value=1, max_value=200,
            value=20, step=1
        ) / 100

    cenario_mc = st.selectbox(
        "Cenário Usado na Simulação",
        options=["Pessimista", "Base", "Otimista"],
        index=1
    )

# Seleciona os fluxos do cenário escolhido
fluxo_base_mc = cenarios[cenario_mc]

# Separa FC0 e fluxos positivos
fc0 = fluxo_base_mc[0]
fluxos_pos = fluxo_base_mc[1:]

# --- Execução da Simulação ---
vpls_mc = []

for _ in range(iteracoes):
    fluxos_simulados = [fc0]

    # Simula cada ano usando distribuição normal
    for f in fluxos_pos:
        fluxo_sort = np.random.normal(loc=f, scale=abs(f) * volatilidade)
        fluxos_simulados.append(fluxo_sort)

    vpl_sim = calcular_vpl(fluxos_simulados, tma_base)
    vpls_mc.append(vpl_sim)

vpls_mc = np.array(vpls_mc)

# --- Estatísticas ---
media = vpls_mc.mean()
mediana = np.median(vpls_mc)
p5 = np.percentile(vpls_mc, 5)
p95 = np.percentile(vpls_mc, 95)
prob_vpl_pos = (vpls_mc > 0).mean() * 100

st.subheader("📊 Resultados da Simulação")

colstats1, colstats2, colstats3 = st.columns(3)
with colstats1:
    st.metric("VPL Médio", f"R$ {media:,.2f}")
with colstats2:
    st.metric("Mediana do VPL", f"R$ {mediana:,.2f}")
with colstats3:
    st.metric("Probabilidade VPL > 0", f"{prob_vpl_pos:.2f}%")

colstats4, colstats5 = st.columns(2)
with colstats4:
    st.metric("Percentil 5%", f"R$ {p5:,.2f}")
with colstats5:
    st.metric("Percentil 95%", f"R$ {p95:,.2f}")

# --- Gráfico ---
st.subheader("📉 Distribuição dos VPLs Simulados")

fig_mc, ax_mc = plt.subplots(figsize=(10,5))
ax_mc.hist(vpls_mc, bins=40)
ax_mc.axvline(0, color="r", linestyle="--", label="VPL = 0")
ax_mc.set_title("Histograma dos VPLs (Monte Carlo)")
ax_mc.set_xlabel("VPL (R$)")
ax_mc.set_ylabel("Frequência")
ax_mc.grid(True)
ax_mc.legend()

st.pyplot(fig_mc)

# --- Fundamentação Teórica ---
with st.expander("📚 Como funciona esta análise?"):
    st.markdown(
        """
### 📌 Objetivo do Aplicativo

Este aplicativo tem como objetivo **avaliar a viabilidade financeira de um projeto ou investimento**, 
considerando o valor do dinheiro ao longo do tempo.  
Ele ajuda a responder perguntas como:

- Vale a pena investir neste projeto?
- Em quanto tempo o investimento se paga?
- O retorno esperado é maior do que o mínimo desejado?

---

### 🧾 O que significam as informações inseridas?

**Investimento Inicial**  
É o valor aplicado no início do projeto (normalmente negativo), como compra de equipamentos, obras ou capital inicial.

**Fluxos de Caixa Anuais**  
São os valores que o projeto gera a cada ano, como receitas líquidas ou economias obtidas.

**TMA – Taxa Mínima de Atratividade**  
Representa o **retorno mínimo esperado** pelo investidor.  
Funciona como uma taxa de comparação: se o projeto render menos que a TMA, ele não é atrativo.

---

### 📊 Indicadores calculados pelo aplicativo

**VPL (Valor Presente Líquido)**  
Mostra quanto o projeto gera de valor hoje, já descontando a TMA.

- VPL **positivo** → projeto financeiramente viável  
- VPL **negativo** → projeto financeiramente inviável

**TIR (Taxa Interna de Retorno)**  
É a taxa de retorno que o próprio projeto oferece.

- Se a TIR for **maior que a TMA**, o projeto é atrativo  
- Se for **menor**, o investimento não compensa

**Payback Descontado**  
Indica **em quantos anos o investimento inicial é recuperado**, considerando o valor do dinheiro no tempo.

---

### 🔍 Análises adicionais

**Análise de Sensibilidade**  
Mostra como o VPL muda quando:
- a taxa de desconto (TMA) varia
- os fluxos de caixa aumentam ou diminuem

**Análise por Cenários**  
Avalia o projeto em três situações:
- **Pessimista:** resultados piores que o esperado  
- **Base:** cenário mais provável  
- **Otimista:** resultados melhores que o esperado  

Isso ajuda a entender os riscos do investimento.

---

📘 **Conclusão:**  
Este aplicativo não prevê o futuro, mas fornece uma base sólida para **tomada de decisão financeira consciente**.
        """
    )
