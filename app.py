"""Aplicação Streamlit — Previsão de churn (CheckPoint05).

Usa exatamente a mesma limpeza (src/preparacao.py), a mesma função de inferência
(src/inferencia.py) e o mesmo pipeline final (models/modelo_final.joblib)
avaliados no notebook. Os gráficos de desempenho são calculados sobre o mesmo
D_teste do notebook (split estratificado 80/20, random_state=42), sem retreino.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

from src.inferencia import carregar_artefato, prever
from src.preparacao import (
    ALVO, VARIAVEIS_CATEGORICAS, VARIAVEIS_MODELO, VARIAVEIS_NUMERICAS,
    carregar_base, codificar_alvo, limpar_dados,
)

RAIZ = Path(__file__).resolve().parent
CAMINHO_CASOS = RAIZ / "models" / "casos_paridade.csv"
CAMINHO_BASE = RAIZ / "data" / "Telco-Customer-Churn.csv"
RANDOM_STATE = 42

# mesma identidade visual do notebook
COR_FICA, COR_CANCELA = "#3b528b", "#5ec962"
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

st.set_page_config(page_title="Previsão de Churn — Telco", page_icon="📉", layout="wide")


@st.cache_resource
def obter_artefato():
    return carregar_artefato(RAIZ / "models" / "modelo_final.joblib")


@st.cache_data
def obter_casos():
    return pd.read_csv(CAMINHO_CASOS) if CAMINHO_CASOS.exists() else pd.DataFrame()


@st.cache_data
def obter_base():
    """Base tratada com a mesma limpeza do notebook."""
    return limpar_dados(carregar_base(str(CAMINHO_BASE)))


@st.cache_data
def obter_avaliacao_teste():
    """Reproduz o D_teste do notebook e calcula as probabilidades do modelo final."""
    df = obter_base()
    X, y = df[VARIAVEIS_MODELO].copy(), codificar_alvo(df[ALVO])
    _, X_teste, _, y_teste = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    proba = obter_artefato()["pipeline"].predict_proba(X_teste)[:, 1]
    return y_teste.to_numpy(), proba


@st.cache_data
def obter_importancias():
    """Importância (total_gain) agregada por variável original, como no notebook (6.3)."""
    pipeline = obter_artefato()["pipeline"]
    nomes = pipeline.named_steps["preprocessamento"].get_feature_names_out()
    ganho = pipeline.named_steps["modelo"].get_booster().get_score(importance_type="total_gain")
    valores = np.array([ganho.get(f"f{i}", ganho.get(nome, 0.0)) for i, nome in enumerate(nomes)])
    imp = pd.Series(valores / valores.sum(), index=nomes)
    original = {nome: next((v for v in sorted(VARIAVEIS_MODELO, key=len, reverse=True)
                            if nome == v or nome.startswith(v + "_")), nome) for nome in nomes}
    return imp.groupby(original).sum().sort_values()


artefato = obter_artefato()
casos = obter_casos()
LIMIAR = artefato["limiar"]

# valores padrão do formulário (formato bruto, igual ao CSV original)
PADRAO = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 960.0,
}
for chave, valor in PADRAO.items():
    st.session_state.setdefault(chave, valor)


def carregar_caso():
    """Preenche o formulário com um caso do teste de paridade."""
    linha = casos[casos["caso"] == st.session_state["caso_escolhido"]].iloc[0]
    for chave in PADRAO:
        valor = linha[chave]
        if chave == "SeniorCitizen":
            valor = int(valor)
        elif chave in ("tenure",):
            valor = int(valor)
        elif chave in ("MonthlyCharges", "TotalCharges"):
            valor = float(valor) if str(valor).strip() not in ("", "nan") else 0.0
        elif chave in ("MultipleLines",) and valor == "No phone service":
            valor = "No"
        elif chave in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                       "StreamingTV", "StreamingMovies") and valor == "No internet service":
            valor = "No"
        st.session_state[chave] = valor


# ------------------------------------------------------------------ cabeçalho
st.title("📉 Previsão de cancelamento de clientes (churn)")
st.markdown(
    "Informe o perfil do cliente para estimar a **probabilidade de cancelamento**. "
    f"Modelo: **{artefato['configuracao']}** · limiar de decisão **{LIMIAR:.2f}** "
    f"(definido na validação cruzada) · AUC no teste **{artefato['metricas_teste']['ROC-AUC']:.3f}**."
)

with st.sidebar:
    st.header("Teste de paridade")
    st.caption("Carrega no formulário um dos casos do teste de consistência do notebook.")
    if not casos.empty:
        st.selectbox("Caso", casos["caso"], key="caso_escolhido")
        st.button("Carregar caso no formulário", on_click=carregar_caso, width="stretch")
        escolhido = casos[casos["caso"] == st.session_state["caso_escolhido"]].iloc[0]
        st.metric("Probabilidade no notebook", f"{escolhido['prob_churn_notebook']:.2%}")
        st.caption(f"Cliente {escolhido['customerID']}")
    st.divider()
    st.header("Sobre o modelo")
    st.json({k: v for k, v in artefato["parametros"].items()
             if k in ("n_estimators", "max_depth", "learning_rate", "subsample")})
    st.caption("Grupo: Pedro Rodrigues Almeida · Vitor Carvalho Alexandre · "
               "Alexandre Martins Lucas · Gabriel Barbosa da Silva — FIAP 2026")

aba_previsao, aba_dados, aba_modelo = st.tabs(
    ["🔮 Previsão", "📊 Exploração dos dados", "🎯 Desempenho do modelo"]
)

# ================================================================== aba 1: previsão
with aba_previsao:
    sim_nao = ["No", "Yes"]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Contrato e cobrança")
        st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"], key="Contract")
        st.number_input("Meses como cliente (tenure)", 0, 72, step=1, key="tenure")
        st.number_input("Mensalidade (US$)", 0.0, 200.0, step=0.05, format="%.2f", key="MonthlyCharges")
        st.number_input("Total já faturado (US$)", 0.0, 10000.0, step=0.05, format="%.2f", key="TotalCharges")
        st.selectbox("Forma de pagamento", ["Electronic check", "Mailed check",
                                            "Bank transfer (automatic)", "Credit card (automatic)"],
                     key="PaymentMethod")
        st.selectbox("Fatura digital", sim_nao, key="PaperlessBilling")

    with col2:
        st.subheader("Serviços")
        st.selectbox("Telefonia", sim_nao, key="PhoneService")
        st.selectbox("Múltiplas linhas", sim_nao, key="MultipleLines")
        st.selectbox("Internet", ["DSL", "Fiber optic", "No"], key="InternetService")
        st.selectbox("Segurança online", sim_nao, key="OnlineSecurity")
        st.selectbox("Backup online", sim_nao, key="OnlineBackup")
        st.selectbox("Proteção de dispositivo", sim_nao, key="DeviceProtection")

    with col3:
        st.subheader("Suporte, streaming e perfil")
        st.selectbox("Suporte técnico", sim_nao, key="TechSupport")
        st.selectbox("Streaming de TV", sim_nao, key="StreamingTV")
        st.selectbox("Streaming de filmes", sim_nao, key="StreamingMovies")
        st.selectbox("Gênero", ["Female", "Male"], key="gender")
        st.selectbox("Idoso (65+)", [0, 1], format_func=lambda v: "Sim" if v else "Não", key="SeniorCitizen")
        st.selectbox("Possui parceiro(a)", sim_nao, key="Partner")
        st.selectbox("Possui dependentes", sim_nao, key="Dependents")

    # coerência: sem internet => serviços de internet = "No"; sem telefone => múltiplas linhas = "No"
    avisos = []
    if st.session_state["InternetService"] == "No" and any(
            st.session_state[c] == "Yes" for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                                   "TechSupport", "StreamingTV", "StreamingMovies"]):
        avisos.append("Cliente sem internet não pode ter serviços de internet — eles serão tratados como 'No'.")
    if st.session_state["PhoneService"] == "No" and st.session_state["MultipleLines"] == "Yes":
        avisos.append("Cliente sem telefonia não pode ter múltiplas linhas — será tratado como 'No'.")

    entrada = pd.DataFrame([{k: st.session_state[k] for k in PADRAO}])
    if entrada.at[0, "InternetService"] == "No":
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            entrada.at[0, c] = "No"
    if entrada.at[0, "PhoneService"] == "No":
        entrada.at[0, "MultipleLines"] = "No"

    # -------------------------------------------------------------- previsão
    st.divider()
    resultado = prever(entrada, artefato).iloc[0]
    prob = float(resultado["prob_churn"])
    vai_cancelar = bool(resultado["churn_previsto"])

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Probabilidade de churn", f"{prob:.2%}")
    c2.metric("Taxa média de churn (treino)", f"{artefato['taxa_churn_treino']:.2%}")
    with c3:
        if vai_cancelar:
            st.error(f"⚠️ **Alto risco de cancelamento** — probabilidade ≥ limiar de {LIMIAR:.2f}. "
                     "Cliente recomendado para ação de retenção.")
        else:
            st.success(f"✅ **Baixo risco de cancelamento** — probabilidade abaixo do limiar de {LIMIAR:.2f}.")
    st.progress(min(prob, 1.0))
    for aviso in avisos:
        st.warning(aviso)

    # posição do cliente na distribuição de risco do conjunto de teste
    y_teste, proba_teste = obter_avaliacao_teste()
    percentil = (proba_teste < prob).mean()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(proba_teste[y_teste == 0], bins=bins, alpha=0.6, color=COR_FICA, label="ficou (real)")
    ax.hist(proba_teste[y_teste == 1], bins=bins, alpha=0.6, color=COR_CANCELA, label="cancelou (real)")
    ax.axvline(LIMIAR, color="gray", ls="--", lw=1.2, label=f"limiar = {LIMIAR:.2f}")
    ax.axvline(prob, color="#d62728", lw=2.5, label=f"este cliente = {prob:.1%}")
    ax.set_xlabel("probabilidade de churn prevista")
    ax.set_ylabel("clientes")
    ax.set_title("Onde este cliente se posiciona entre os clientes do conjunto de teste")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Este cliente tem risco maior que **{percentil:.0%}** dos clientes do conjunto de teste.")

    with st.expander("Entradas enviadas ao modelo"):
        st.dataframe(entrada, width="stretch", hide_index=True)

# ================================================================== aba 2: exploração
with aba_dados:
    df = obter_base()
    st.markdown(
        f"Base tratada: **{df.shape[0]:,} clientes** · taxa de churn **{(df[ALVO] == 'Yes').mean():.1%}**. "
        "Mesma limpeza aplicada no notebook (`src/preparacao.py`)."
    )

    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots(figsize=(6, 4))
        contagem = df[ALVO].value_counts().reindex(["No", "Yes"])
        barras = ax.bar(["Ficou (No)", "Cancelou (Yes)"], contagem.values, color=[COR_FICA, COR_CANCELA])
        for barra, n in zip(barras, contagem.values):
            ax.annotate(f"{n:,}\n({n / len(df):.1%})", (barra.get_x() + barra.get_width() / 2, n),
                        ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, contagem.max() * 1.18)
        ax.set_ylabel("clientes")
        ax.set_title("Distribuição da variável-alvo")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Classes desbalanceadas (~27% de churn): por isso a métrica principal é ROC-AUC, "
                   "e não acurácia.")

    with e2:
        variavel = st.selectbox(
            "Taxa de churn por categoria",
            ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
             "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"],
        )
        taxa = (df.assign(churn=df[ALVO].eq("Yes")).groupby(variavel)["churn"]
                .agg(["mean", "size"]).sort_values("mean"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(taxa.index.astype(str), taxa["mean"], color=sns.color_palette("viridis", len(taxa)))
        ax.axvline((df[ALVO] == "Yes").mean(), color="gray", ls="--", lw=1, label="média geral")
        for i, (m, n) in enumerate(zip(taxa["mean"], taxa["size"])):
            ax.text(m + 0.005, i, f"{m:.1%} (n={n:,})", va="center", fontsize=9)
        ax.set_xlim(0, taxa["mean"].max() * 1.35)
        ax.set_xlabel("taxa de churn")
        ax.set_title(f"Taxa de churn por {variavel}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col in zip(axes, VARIAVEIS_NUMERICAS):
        sns.histplot(data=df, x=col, hue=ALVO, hue_order=["No", "Yes"], bins=30, stat="density",
                     common_norm=False, element="step", palette=[COR_FICA, COR_CANCELA], ax=ax)
        ax.set_title(f"{col} por Churn")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Clientes que cancelam concentram-se nos primeiros meses (tenure baixo) e em mensalidades "
               "mais altas; consequentemente, têm TotalCharges menor.")

# ================================================================== aba 3: desempenho
with aba_modelo:
    y_teste, proba_teste = obter_avaliacao_teste()
    m = artefato["metricas_teste"]
    st.markdown(
        f"Avaliação do pipeline final no **conjunto de teste** ({len(y_teste):,} clientes, "
        "nunca usados no treino nem no tuning), reproduzindo o split do notebook."
    )
    k = st.columns(6)
    k[0].metric("ROC-AUC", f"{roc_auc_score(y_teste, proba_teste):.3f}",
                f"{roc_auc_score(y_teste, proba_teste) - artefato['metricas_cv']['roc_auc_media']:+.3f} vs. CV")
    k[1].metric("PR-AUC", f"{average_precision_score(y_teste, proba_teste):.3f}")
    k[2].metric("Recall", f"{m['recall']:.3f}")
    k[3].metric("Precision", f"{m['precision']:.3f}")
    k[4].metric("F1", f"{m['F1']:.3f}")
    k[5].metric("Acurácia", f"{next(v for c, v in m.items() if c.lower().startswith('acur')):.3f}")

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        fpr, tpr, _ = roc_curve(y_teste, proba_teste)
        ax.plot(fpr, tpr, color=COR_FICA, lw=2, label=f"modelo (AUC = {roc_auc_score(y_teste, proba_teste):.3f})")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="aleatório")
        ax.set_xlabel("taxa de falsos positivos")
        ax.set_ylabel("taxa de verdadeiros positivos (recall)")
        ax.set_title("Curva ROC — conjunto de teste")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with d2:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        prec, rec, _ = precision_recall_curve(y_teste, proba_teste)
        ax.plot(rec, prec, color=COR_CANCELA, lw=2,
                label=f"modelo (AP = {average_precision_score(y_teste, proba_teste):.3f})")
        ax.axhline(y_teste.mean(), color="gray", ls="--", lw=1, label=f"prevalência = {y_teste.mean():.1%}")
        ax.scatter([m["recall"]], [m["precision"]], color="#d62728", zorder=3, label=f"limiar {LIMIAR:.2f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("Curva Precision–Recall — conjunto de teste")
        ax.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    d3, d4 = st.columns(2)
    with d3:
        limiar_cm = st.slider("Limiar de decisão para a matriz de confusão", 0.05, 0.95, float(LIMIAR), 0.01)
        cm = confusion_matrix(y_teste, (proba_teste >= limiar_cm).astype(int))
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        ConfusionMatrixDisplay(cm, display_labels=["Ficou", "Cancelou"]).plot(
            ax=ax, cmap="viridis", colorbar=False, values_format="d")
        ax.grid(False)
        ax.set_title(f"Matriz de confusão (limiar = {limiar_cm:.2f})")
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        vn, fp, fn, vp = cm.ravel()
        st.caption(f"Com este limiar o modelo identifica **{vp / (vp + fn):.1%}** dos clientes que cancelam "
                   f"(recall), com **{vp / max(vp + fp, 1):.1%}** de acerto entre os sinalizados (precision). "
                   f"O limiar {LIMIAR:.2f} foi escolhido na validação cruzada.")

    with d4:
        imp = obter_importancias()
        fig, ax = plt.subplots(figsize=(6, 5.6))
        ax.barh(imp.index, imp.values, color=sns.color_palette("viridis", len(imp)))
        ax.set_xlabel("importância normalizada — ganho total (total_gain)")
        ax.set_title("Importância das variáveis (agregada por variável original)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Tipo de contrato, internet por fibra e tempo como cliente dominam a decisão do modelo.")
