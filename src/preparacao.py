"""Preparação de dados compartilhada entre o notebook e a aplicação Streamlit.

Este módulo concentra as regras de data wrangling **determinísticas** (que não
aprendem parâmetros da amostra) e a definição do pipeline de pré-processamento.
Notebook e aplicação importam as mesmas funções, garantindo que a interface use
exatamente a mesma preparação avaliada no CheckPoint05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

URL_BASE = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

ALVO = "Churn"
IDENTIFICADOR = "customerID"

VARIAVEIS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
VARIAVEIS_CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
VARIAVEIS_MODELO = VARIAVEIS_NUMERICAS + VARIAVEIS_CATEGORICAS

# Serviços que dependem de internet: "No internet service" é redundante com
# InternetService == "No" e é recodificado para "No".
SERVICOS_INTERNET = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def carregar_base(caminho: str | None = None) -> pd.DataFrame:
    """Carrega a base local; se não existir, baixa da fonte pública da IBM."""
    if caminho is not None:
        try:
            return pd.read_csv(caminho)
        except FileNotFoundError:
            pass
    return pd.read_csv(URL_BASE)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as regras determinísticas de limpeza (sem estatísticas da amostra).

    1. ``TotalCharges`` é convertida para numérica; textos em branco viram NaN.
       Clientes com ``tenure == 0`` ainda não foram faturados, então o valor
       correto pelo domínio é 0 (regra de negócio, não imputação estatística).
    2. Categorias "No internet service" / "No phone service" são recodificadas
       para "No"; a informação permanece em ``InternetService``/``PhoneService``.
    3. ``SeniorCitizen`` (0/1) é convertida para "No"/"Yes", padronizando o
       formato das variáveis binárias.
    4. Espaços extras em textos são removidos.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
        if "tenure" in df.columns:
            sem_faturamento = df["TotalCharges"].isna() & (df["tenure"] == 0)
            df.loc[sem_faturamento, "TotalCharges"] = 0.0

    for col in SERVICOS_INTERNET:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].astype(str).map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
        )

    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # garante dtype object para as categóricas (compatível com pandas 2 e 3)
    for col in VARIAVEIS_CATEGORICAS + [ALVO, IDENTIFICADOR]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    return df


def codificar_alvo(serie: pd.Series) -> pd.Series:
    """Converte o alvo textual em 0/1 (1 = cliente cancelou)."""
    return serie.map({"No": 0, "Yes": 1}).astype(int)


def criar_preprocessador() -> ColumnTransformer:
    """Pré-processamento comum aos três algoritmos.

    - Numéricas: imputação pela mediana (salvaguarda; ajustada só no treino)
      e sem escalonamento, pois modelos de árvore são invariantes a escala.
    - Categóricas: imputação pela moda + One-Hot Encoding; binárias viram uma
      única coluna; categorias desconhecidas em produção são ignoradas.
    """
    numericas = Pipeline([("imputador", SimpleImputer(strategy="median"))])
    categoricas = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numericas, VARIAVEIS_NUMERICAS),
            ("cat", categoricas, VARIAVEIS_CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    )


def criar_pipeline(modelo) -> Pipeline:
    """Encadeia pré-processamento e estimador em um único objeto."""
    return Pipeline([("preprocessamento", criar_preprocessador()), ("modelo", modelo)])
