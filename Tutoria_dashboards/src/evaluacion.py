"""
Qué estamos construyendo
------------------------
El CEREBRO del tablero. No pinta widgets. No escribe archivos.

Tres gestos (los términos nuevos de ESTA clase):

    1. Predicción en LOTE
       Una fila  = lo que hacía el mostrador (Streamlit / front-back).
       Un lote   = .predict(X) con N filas de un golpe. El tablero
                   necesita N para poder HACER ESTADÍSTICA.

    2. Métricas (un número que resume N errores)
       MAE    error medio en euros
       RMSE   como el MAE, pero castiga más los fallos gordos
       R2     "¿mejoro a predecir siempre la media?"
       sesgo  "¿el modelo se pasa o se queda corto, de media?"

    3. Diagnóstico por SEGMENTOS
       El MAE global esconde la verdad. Puede ir bien en clientes
       pequeños y fatal en los grandes (Q4). Eso es una decisión
       de negocio, no un detalle estadístico.

Pregunta que un único predict() NO responde:
    "¿el modelo sirve, o solo acertó en el cliente 8?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.features_modelo import COLUMNA_TARGET, COLUMNAS_FEATURES

# Semilla = número que fija el "azar". Misma semilla → mismo barajado.
# 42 y 0.2 son EXACTAMENTE los de scripts/entrenar_modelos.py.
# Si cambias uno, el holdout deja de ser el de meta.json.
SEMILLA_OFICIAL = 42
TEST_SIZE_OFICIAL = 0.2


def particion_oficial(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Qué construimos: el mismo CORTE que usó el entrenamiento.

    Vocabulario
    -----------
    Entrenamiento (train)
        Filas con las que el modelo APRENDIÓ (fit). ~80 %.
        Aquí el modelo "ya vio" la respuesta. Puntuarlas es hacerse
        trampas: es como examinar a un alumno con las preguntas
        que memorizó anoche.

    Holdout / test / conjunto de prueba
        Filas que el fit() NO vio. ~20 %.
        Es el examen de verdad. meta.json se midió AQUÍ.
        Un tablero serio mira el holdout, no el train.

    train_test_split(tabla, test_size=0.2, random_state=42)
        Baraja las FILAS (no las columnas) y corta.
        test_size=0.2 → 20 % holdout, 80 % train.
        random_state=42 → el barajado es reproducible: todos en
        clase obtenemos las MISMAS 2.000 filas.

    Devuelve (tren, holdout). El tablero usa el segundo.
    """
    # train_test_split(df, test_size=, random_state=)
    #   Baraja las FILAS y corta en dos DataFrames.
    #   test_size=0.2 → el segundo es el 20 % (holdout).
    #   random_state=42 → mismo barajado siempre (reproducible).
    #   DEVUELVE (train, test). No toca las columnas.
    tren, holdout = train_test_split(
        tabla,
        test_size=TEST_SIZE_OFICIAL,
        random_state=SEMILLA_OFICIAL,
    )
    return tren, holdout


def predecir_lote(paquete: dict, tabla: pd.DataFrame) -> pd.Series:
    """
    Qué construimos: N predicciones de un golpe (no un cliente).

    paquete["modelo"] es el objeto sklearn. Tiene .predict(X).
    X tiene que ser un DataFrame con las 4 columnas, EN ESTE ORDEN.

    loc[:, COLUMNAS_FEATURES]
        "todas las filas (:), solo esas columnas".
        El cliente_id y el target se quedan fuera: el modelo
        no los vio al entrenar y no debe verlos ahora.

    .predict(x) → array numpy de N números (uno por fila).
    En la clase Streamlit hacíamos [0] porque N valía 1.
    Aquí N vale 2.000 o 9.996: devolvemos una Series entera.

    Series = columna de pandas con índice. El índice lo copiamos
    del DataFrame de entrada para poder alinear después
    y_real (fila 17) con y_pred (fila 17).
    """
    # loc[filas, columnas]: selecciona. ":" = todas las filas.
    # COLUMNAS_FEATURES = las 4 X. cliente_id y el target se quedan fuera.
    x = tabla.loc[:, COLUMNAS_FEATURES]
    # .predict(X) = método de sklearn. X = DataFrame N×4.
    # DEVUELVE un array numpy de N números (uno por fila). Sin [0]: N > 1.
    valores = paquete["modelo"].predict(x)
    # pd.Series(datos, index=, name=): array → columna con índice.
    # index=tabla.index alinea y_pred[17] con y_real[17].
    # .round(2) = redondea cada número a 2 decimales (céntimos).
    return pd.Series(valores, index=tabla.index, name="y_pred").round(2)


def metricas(y_real: pd.Series, y_pred: pd.Series) -> dict:
    """
    Qué construimos: 4 números que caben en un KPI del tablero.

    Residual / residuo
        predicho − real   (a veces se define al revés; aquí es pred − real)
        +300  → el modelo se PASÓ (dijo 1.300, gastó 1.000)
        −300  → el modelo se QUEDÓ CORTO

    MAE — Mean Absolute Error (error absoluto medio)
        media de |pred − real|
        Unidad: euros. "De media, fallamos por 1.017 €".
        Es la métrica de meta.json. La más fácil de explicar a negocio.

    RMSE — Root Mean Squared Error (raíz del error cuadrático medio)
        como el MAE, pero ANTES de restar se ELEVAN AL CUADRADO.
        Un fallo de 10.000 € pesa muchísimo más que cien fallos de 100 €.
        Si RMSE >> MAE, hay COLAS: unos pocos clientes destrozan la media.

    R2 — coeficiente de determinación
        1.0  = predicciones = realidad (perfecto, no existe en la calle)
        0.0  = el modelo no mejora a decir SIEMPRE la media de y
        < 0  = peor que la media (el modelo estorba)
        El nuestro ronda 0.05: apenas mejoramos a predecir la media.
        Eso NO se esconde. El dashboard existe para verlo.

    Sesgo (bias)
        media(pred − real)
        > 0  el modelo es OPTIMISTA (predice de más, de media)
        < 0  el modelo es PESIMISTA (se queda corto, de media)
        Un sesgo de −50 € con MAE de 1.017 €: no va sistemáticamente
        alto o bajo; va mal en las DOS direcciones.

    to_numpy()
        Series → array plano. Las funciones de sklearn hablan numpy,
        no pandas. dtype=float evita enteros que redondeen mal.
    """
    # to_numpy(dtype=): Series de pandas → array plano de numpy.
    # sklearn.metrics habla numpy, no pandas. dtype=float evita enteros.
    real = y_real.to_numpy(dtype=float)
    pred = y_pred.to_numpy(dtype=float)
    residuo = pred - real
    return {
        # len() = cuántos elementos. int() = por si numpy devuelve np.int64.
        "n": int(len(real)),
        # mean_absolute_error(y, ŷ) = media de |ŷ − y|. Euros. 2 decimales.
        "mae": round(float(mean_absolute_error(real, pred)), 2),
        # root_mean_squared_error(y, ŷ) = √ media((ŷ − y)²). Castiga colas.
        "rmse": round(float(root_mean_squared_error(real, pred)), 2),
        # r2_score(y, ŷ) = 1 − (SSres / SStot). 0 = no mejoras a la media.
        "r2": round(float(r2_score(real, pred)), 3),
        # np.mean(array) = promedio. Aquí del residuo = sesgo.
        "sesgo": round(float(np.mean(residuo)), 2),
    }


def construir_monitoreo(
    tabla: pd.DataFrame,
    paquete_sklearn: dict,
    paquete_mlp: dict,
) -> pd.DataFrame:
    """
    Qué construimos: UNA tabla "lista para el tablero y para el CSV".

    Los dos motores predicen las MISMAS filas. Si no, comparar
    MAE_sklearn con MAE_mlp sería hacer trampas (distintos clientes).

    Columnas que salen
    ------------------
    cliente_id + 4 features     el expediente (para filtrar / entender)
    y_real                      lo que ocurrió (el target, renombrado)
    pred_sklearn, pred_mlp      lo que dijo cada cerebro
    error_sklearn, error_mlp    pred − real  (lleva SIGNO: dirección)
    abs_sklearn, abs_mlp        |error|      (sin signo: MAGNITUD)

    ¿Por qué error Y abs?
        error sirve para el sesgo y el histograma (izquierda/derecha).
        abs sirve para "los 10 peores" y para el MAE por tramo:
        da igual pasarse 8.000 que quedarse corto 8.000: fallaste 8.000.

    reset_index(drop=True)
        el holdout heredó índices 7, 40, 112... del CSV original.
        Para el CSV descargable queremos 0..n-1 limpio.
        drop=True = no conviertas el índice viejo en una columna.
    """
    pred_sk = predecir_lote(paquete_sklearn, tabla)
    pred_mlp = predecir_lote(paquete_mlp, tabla)
    y_real = tabla[COLUMNA_TARGET]

    # loc + lista de columnas: recorta. *COLUMNAS_FEATURES = desempaqueta la lista.
    # .copy() = DataFrame NUEVO. Sin copy, pandas avisa (SettingWithCopy).
    salida = tabla.loc[:, ["cliente_id", *COLUMNAS_FEATURES, COLUMNA_TARGET]].copy()
    # rename(columns={viejo: nuevo}): cambia el nombre. No toca los valores.
    salida = salida.rename(columns={COLUMNA_TARGET: "y_real"})
    # to_numpy(): Series → array, para asignar por posición (mismos N).
    salida["pred_sklearn"] = pred_sk.to_numpy()
    salida["pred_mlp"] = pred_mlp.to_numpy()
    # .round(2) otra vez: céntimos. error = pred − real (lleva SIGNO).
    salida["error_sklearn"] = (salida["pred_sklearn"] - salida["y_real"]).round(2)
    salida["error_mlp"] = (salida["pred_mlp"] - salida["y_real"]).round(2)
    # .abs() = valor absoluto de cada celda. |error| = magnitud (sin signo).
    salida["abs_sklearn"] = salida["error_sklearn"].abs()
    salida["abs_mlp"] = salida["error_mlp"].abs()
    # reset_index(drop=True): índice 7,40,112… → 0,1,2…
    # drop=True = NO conviertas el índice viejo en una columna extra.
    return salida.reset_index(drop=True)


def resumir_metricas(monitoreo: pd.DataFrame) -> dict:
    """
    Qué construimos: las métricas de los DOS motores, lado a lado.

    Misma tabla → misma n, mismo y_real. La única diferencia es
    la columna pred_*. Así "quién gana" es comparable.

    Devuelve:
        {"sklearn": {n, mae, rmse, r2, sesgo},
         "mlp":     {n, mae, rmse, r2, sesgo}}
    """
    return {
        "sklearn": metricas(monitoreo["y_real"], monitoreo["pred_sklearn"]),
        "mlp": metricas(monitoreo["y_real"], monitoreo["pred_mlp"]),
    }


def peores_clientes(monitoreo: pd.DataFrame, motor: str = "sklearn", n: int = 10) -> pd.DataFrame:
    """
    Qué construimos: los n clientes con mayor |error|.

    sort_values(col, ascending=False) = de mayor a menor.
    head(n) = quédate con los n primeros.

    Pregunta de clase (no la respondas tú: lánzala):
        ¿Qué tienen en común? ¿y_real = 0 (no compraron en 90 días)?
        ¿gasto_historico extremo? Ahí el modelo no interpola: EXTRAPOLA.

    motor="sklearn" o "mlp" elige QUÉ columna de error ordenamos.
    El parámetro se llama motor y no "modelo" para no chocar con
    el objeto sklearn que vive dentro del paquete.
    """
    col = "abs_sklearn" if motor == "sklearn" else "abs_mlp"
    cols = [
        "cliente_id",
        "gasto_historico",
        "y_real",
        f"pred_{motor}",
        f"error_{motor}",
        col,
    ]
    # loc[:, cols]     = quédate solo esas columnas.
    # sort_values(col, ascending=False) = ordena esa columna de MAYOR a menor.
    # head(n)          = las n primeras filas del resultado (los n peores).
    return monitoreo.loc[:, cols].sort_values(col, ascending=False).head(n)


def error_por_tramo(
    monitoreo: pd.DataFrame,
    motor: str = "sklearn",
    col_corte: str = "gasto_historico",
) -> pd.DataFrame:
    """
    Qué construimos: el MAE partido en 4 grupos de clientes.

    Cuartil
        Partes la variable en 4 trozos con el MISMO número de filas
        (no el mismo ancho en euros).
        Q1 = el 25 % que menos gastó en el pasado
        Q4 = el 25 % que más gastó en el pasado

    pd.qcut(serie, q=4, labels=...)
        q = quantile cut. "Corta por cuantiles".
        Distinto de pd.cut, que corta por euros fijos (0-1000, 1000-2000…).
        qcut garantiza ~500 clientes por barra si el holdout tiene 2.000.

    groupby("tramo")[col_abs].mean()
        dentro de cada cuartil, media del |error| = MAE de ese tramo.

    observed=False
        pandas 2.x: si un tramo quedara vacío, igual aparece en la tabla.
        En clase no debería pasar (qcut rellena los 4).

    Lectura de negocio
        Si Q4.mae >> Q1.mae el modelo NO vale igual para clientes VIP.
        Un único MAE de 1.017 € te lo escondía.
    """
    col_abs = "abs_sklearn" if motor == "sklearn" else "abs_mlp"
    # pd.qcut(serie, q=, labels=): corta por CUANTILES (mismo nº de filas).
    #   q=4 → 4 trozos (cuartiles). labels= nombre de cada trozo.
    #   Distinto de pd.cut (cortes en euros fijos). DEVUELVE una Series categórica.
    tramo = pd.qcut(
        monitoreo[col_corte],
        q=4,
        labels=["Q1 bajo", "Q2", "Q3", "Q4 alto"],
    )
    resumen = (
        # assign(col=serie): añade una columna temporal ("tramo") y sigue.
        monitoreo.assign(tramo=tramo)
        # groupby("tramo")[col]: parte el DataFrame en 4 grupos; quédate |error|.
        # observed=False: si un tramo vacío, igual aparece (pandas 2.x).
        .groupby("tramo", observed=False)[col_abs]
        # mean() = promedio dentro de cada grupo = MAE de ese cuartil.
        .mean()
        .round(2)
        # rename("mae"): el nombre de la Series (la columna) pasa a ser "mae".
        .rename("mae")
        # to_frame(): Series → DataFrame de 1 columna (st.bar_chart lo prefiere).
        .to_frame()
    )
    # groupby(...).size() = cuántas filas hay en cada grupo (no suma valores).
    resumen["n"] = monitoreo.groupby(tramo, observed=False).size()
    return resumen


def histograma_residuales(
    monitoreo: pd.DataFrame,
    motor: str = "sklearn",
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Qué construimos: "¿cuántos clientes caen en cada rango de error?"

    Histograma
        Partes el eje (aquí: pred − real) en cajones (bins) y CUENTAS
        cuántos puntos caen en cada cajón. No es un gráfico de barras
        de categorías (lunes/martes): es la FORMA de una variable continua.

    Cómo se lee
        Campana centrada en 0     residuales sanos (fallamos arriba y abajo)
        Campana desplazada a +    sesgo positivo (nos pasamos)
        Una cola larguísima       pocos fallos ENORMES (RMSE >> MAE)

    

    """
    col = "error_sklearn" if motor == "sklearn" else "error_mlp"
    # value_counts(bins=n): parte el eje en n cajones y CUENTA filas por cajón.
    #   (sin bins= contaría valores exactos: inútil en un continuo).
    # sort_index() = ordena los cajones de menor a mayor (de −grandes a +grandes).
    conteo = monitoreo[col].value_counts(bins=n_bins).sort_index()
    # Cada índice es un Interval. .left / .right = los dos extremos en euros.
    etiquetas = [f"{iv.left:.0f} a {iv.right:.0f}" for iv in conteo.index]
    # DataFrame({col: array}, index=): 1 columna "clientes", filas = rangos.
    # to_numpy() otra vez: valores del conteo, sin el IntervalIndex.
    return pd.DataFrame({"clientes": conteo.to_numpy()}, index=etiquetas)