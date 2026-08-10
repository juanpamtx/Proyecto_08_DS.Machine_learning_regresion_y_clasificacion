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
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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


def evaluar_clasificacion(y_real, y_pred):
    """
    Calcula las principales métricas de evaluación
    para un modelo de clasificación binaria.

    Se incluyen métricas globales y métricas específicas
    para la clase minoritaria 0, correspondiente a los
    estudiantes no aprobados.

    Parameters
    ----------
    y_real : array-like
        Valores reales de la variable objetivo.

    y_pred : array-like
        Clases predichas por el modelo.

    Returns
    -------
    pandas.DataFrame
        Tabla con las métricas de clasificación.
    """
    accuracy = accuracy_score(y_real, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_real, y_pred)

    precision_no_aprobado = precision_score(
        y_real,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall_no_aprobado = recall_score(
        y_real,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1_no_aprobado = f1_score(
        y_real,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    metricas = pd.DataFrame({
        "metrica": [
            "Accuracy",
            "Balanced accuracy",
            "Precision no aprobado",
            "Recall no aprobado",
            "F1 no aprobado"
        ],
        "valor": [
            accuracy,
            balanced_accuracy,
            precision_no_aprobado,
            recall_no_aprobado,
            f1_no_aprobado
        ]
    })

    return metricas


def graficar_matriz_confusion(y_real, y_pred, titulo):
    """
    Representa la matriz de confusión de un modelo
    de clasificación binaria.
    """
    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No aprobado", "Aprobado"],
        yticklabels=["No aprobado", "Aprobado"]
    )

    plt.title(titulo)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()


def obtener_informe_clasificacion(y_real, y_pred):
    """
    Genera el informe de clasificación en formato DataFrame.
    """
    informe = classification_report(
        y_real,
        y_pred,
        labels=[0, 1],
        target_names=["No aprobado", "Aprobado"],
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(informe).T
  