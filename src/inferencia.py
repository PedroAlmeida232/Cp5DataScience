"""Funções de inferência usadas pelo notebook (teste de paridade) e pelo Streamlit."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.preparacao import VARIAVEIS_MODELO, limpar_dados

CAMINHO_ARTEFATO = Path(__file__).resolve().parents[1] / "models" / "modelo_final.joblib"


def carregar_artefato(caminho: str | Path = CAMINHO_ARTEFATO) -> dict:
    """Carrega o dicionário salvo no Exercício 7 (pipeline, limiar e metadados)."""
    return joblib.load(caminho)


def prever(entradas: pd.DataFrame, artefato: dict) -> pd.DataFrame:
    """Aplica a mesma limpeza do notebook e o pipeline final às entradas.

    Retorna a probabilidade estimada de churn e a classe prevista segundo o
    limiar de decisão definido na validação cruzada (Exercício 6).
    """
    dados = limpar_dados(entradas)[VARIAVEIS_MODELO]
    proba = artefato["pipeline"].predict_proba(dados)[:, 1]
    limiar = artefato["limiar"]
    return pd.DataFrame(
        {
            "prob_churn": proba,
            "churn_previsto": (proba >= limiar).astype(int),
        },
        index=entradas.index,
    )
