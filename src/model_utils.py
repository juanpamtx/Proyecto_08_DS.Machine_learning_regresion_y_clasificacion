"""
Funciones auxiliares para la evaluación y visualización
de los modelos de Machine Learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluar_regresion(y_real, y_pred):
    """
    Calcula las principales métricas de evaluación
    para un modelo de regresión.

    Parameters
    ----------
    y_real : array-like
        Valores reales de la variable objetivo.

    y_pred : array-like
        Valores predichos por el modelo.

    Returns
    -------
    pandas.DataFrame
        Tabla con las métricas MAE, RMSE y R2.
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    metricas = pd.DataFrame({
        "metrica": ["MAE", "RMSE", "R2"],
        "valor": [mae, rmse, r2]
    })

    return metricas


def graficar_reales_predichos(y_real, y_pred):
    """
    Representa los valores reales frente a los valores
    predichos por un modelo de regresión.
    """
    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        x=y_real,
        y=y_pred,
        alpha=0.7
    )

    limite_minimo = min(np.min(y_real), np.min(y_pred))
    limite_maximo = max(np.max(y_real), np.max(y_pred))

    plt.plot(
        [limite_minimo, limite_maximo],
        [limite_minimo, limite_maximo],
        color="red",
        linestyle="--",
        label="Predicción perfecta"
    )

    plt.title("Valores reales frente a valores predichos")
    plt.xlabel("Nota final real")
    plt.ylabel("Nota final predicha")
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_residuos(y_real, y_pred):
    """
    Representa los residuos del modelo respecto
    a los valores predichos.
    """
    residuos = np.asarray(y_real) - np.asarray(y_pred)

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        x=y_pred,
        y=residuos,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title("Distribución de los residuos")
    plt.xlabel("Nota final predicha")
    plt.ylabel("Residuo")
    plt.tight_layout()
    plt.show()