"""
Qué estamos construyendo
------------------------
La SALIDA del tablero. No pinta widgets. No llama a Streamlit.
No entrena. No predice.

El cerebro ya hizo su trabajo en evaluacion.py:
    construir_monitoreo  → una tabla (filas = clientes del examen)
    resumir_metricas     → dos dicts, sklearn y mlp, con
                           n, mae, rmse, r2, sesgo

Aquí convertimos eso en dos archivos que un analista se puede llevar:

    1. csv_bytes
       La tabla cruda. Para abrirla en Excel o en otro notebook.
       Una fila = un cliente del holdout, con y_real y las dos
       predicciones. Sin resumir.

    2. informe_markdown / informe_bytes
       El acta. Quién gana (MAE más bajo), si el tablero REPITE
       los números de meta.json, y el glosario para no leer
       R2 = 0.05 como si fuera un éxito.

meta.json es la ficha del entrenamiento. Guardó MAE y R2 del
holdout (semilla 42, 20 %). No guardó RMSE ni sesgo: esos solo
salen del tablero. n_filas de la ficha es el CSV ENTERO (~9.996),
no las ~2.000 filas del examen. Comparar "n" con "n_filas" sería
mezclar el examen con la clase completa.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.evaluacion import resumir_metricas


def csv_bytes(monitoreo: pd.DataFrame) -> bytes:
    """
    Qué construimos: el CSV de monitoreo, ya en bytes.

    El botón de descarga no quiere un DataFrame (vive en RAM).
    Quiere bytes: la misma tabla escrita como texto.

    index=False
        El índice 0, 1, 2… no es un dato. El identificador es
        cliente_id, que ya va en una columna. Si lo dejamos,
        Excel abre una columna "Unnamed: 0" que confunde.
    """
    # to_csv(index=False)
    #   RECIBE: el DataFrame monitoreo (una fila por cliente).
    #   index=False: no escribas la columna del índice.
    #   DEVUELVE: str. Cabecera + filas, separadas por comas y saltos de línea.
    texto = monitoreo.to_csv(index=False)
    # encode("utf-8")
    #   RECIBE: ese str (texto Unicode: cabecera y filas).
    #   DEVUELVE: bytes. Lo que se guarda en disco o se descarga.
    return texto.encode("utf-8")


def informe_markdown(monitoreo: pd.DataFrame, meta: dict) -> str:
    """
    Qué construimos: el acta en Markdown (texto, todavía no archivo).

    Pasos
    -----
    1. resumir_metricas(monitoreo)
       Misma tabla, dos motores. Devuelve
       {"sklearn": {n, mae, rmse, r2, sesgo}, "mlp": {...}}.
    2. Ganador = el MAE más bajo. MAE es euros de error medio:
       el número que negocio entiende y el que guardó meta.json.
    3. Comparar MAE y R2 del tablero contra meta.json.
       Si no cuadran, el corte no es el holdout oficial o el
       joblib no es el de la ficha.
    4. Glosario. El R2 ~ 0.05 se escribe con sus decimales.
       No se redondea a "el modelo va bien".
    """
    cifras = resumir_metricas(monitoreo)
    sklearn = cifras["sklearn"]
    mlp = cifras["mlp"]

    # datetime.now()
    #   RECIBE: nada. Lee el reloj del ordenador.
    #   DEVUELVE: un datetime con la fecha y la hora de este instante.
    ahora = datetime.now()
    # strftime(formato)
    #   RECIBE: ese datetime y el molde "%Y-%m-%d %H:%M".
    #   DEVUELVE: str legible, por ejemplo "2026-10-29 20:52".
    cuando = ahora.strftime("%Y-%m-%d %H:%M")

    # dict.get(clave, default)
    #   RECIBE: meta (el dict de meta.json) y la clave.
    #   Si la clave no está, NO lanza KeyError: DEVUELVE el default ("—").
    version = meta.get("version", "—")
    # dict.get(clave, default)
    #   RECIBE: meta y "fecha_corte". DEVUELVE el día en que se congelaron
    #   las features, o "—" si la ficha no lo trae.
    fecha_corte = meta.get("fecha_corte", "—")
    # dict.get(clave, default)
    #   RECIBE: meta y "horizonte_dias". DEVUELVE el horizonte del target
    #   (90) o "—".
    horizonte = meta.get("horizonte_dias", "—")
    # dict.get(clave, default)
    #   RECIBE: meta y "n_filas". DEVUELVE cuántas filas tenía el CSV
    #   completo al entrenar, no el tamaño del holdout.
    n_ficha = meta.get("n_filas", "—")

    if sklearn["mae"] < mlp["mae"]:
        ganador = "sklearn"
    elif mlp["mae"] < sklearn["mae"]:
        ganador = "mlp"
    else:
        ganador = "empate"

    if ganador == "empate":
        texto_ganador = (
            f"Empate. Los dos motores tienen el mismo MAE "
            f"({sklearn['mae']:.2f} €). Por esta regla no hay ganador."
        )
    else:
        perdedor = "mlp" if ganador == "sklearn" else "sklearn"
        texto_ganador = (
            f"Gana **{ganador}**. Su MAE es {cifras[ganador]['mae']:.2f} €, "
            f"más bajo que el de {perdedor} ({cifras[perdedor]['mae']:.2f} €). "
            "Más bajo = menos euros de error medio. No miramos el R2 para "
            "elegir: la ficha y el negocio hablan de MAE."
        )

    filas_comparacion = []
    r2_ficha = {}
    for motor in ("sklearn", "mlp"):
        tablero = cifras[motor]
        # dict.get(clave, default)
        #   RECIBE: meta y el nombre del motor ("sklearn" o "mlp").
        #   DEVUELVE: el dict {mae, r2, segundos} de ese motor, o {} si falta.
        ficha = meta.get(motor, {})
        if not isinstance(ficha, dict):
            ficha = {}
        # dict.get(clave)
        #   RECIBE: la ficha de ese motor y la clave "mae".
        #   DEVUELVE: el MAE de meta.json, o None si no está.
        mae_meta = ficha.get("mae")
        # dict.get(clave)
        #   RECIBE: la ficha de ese motor y la clave "r2".
        #   DEVUELVE: el R2 de meta.json, o None si no está.
        r2_meta = ficha.get("r2")
        r2_ficha[motor] = r2_meta
        filas_comparacion.append(
            "| {motor} | {mae_tab:.2f} | {mae_meta} | {delta_mae} | {marca_mae} "
            "| {r2_tab:.3f} | {r2_meta} | {delta_r2} | {marca_r2} |".format(
                motor=motor,
                mae_tab=tablero["mae"],
                mae_meta=_fmt(mae_meta, 2),
                delta_mae=_delta(tablero["mae"], mae_meta, 2),
                marca_mae=_cuadra(tablero["mae"], mae_meta, 0.01),
                r2_tab=tablero["r2"],
                r2_meta=_fmt(r2_meta, 3),
                delta_r2=_delta(tablero["r2"], r2_meta, 3),
                marca_r2=_cuadra(tablero["r2"], r2_meta, 0.001),
            )
        )

    lineas = [
        "# Informe de monitoreo del gasto",
        "",
        f"Generado: {cuando}",
        "",
        "## Ficha (meta.json)",
        "",
        f"- Versión del artefacto: {version}",
        f"- Fecha de corte de las features: {fecha_corte}",
        f"- Horizonte del target: {horizonte} días",
        f"- Filas del CSV completo (n_filas): {n_ficha}",
        "",
        "Esas filas NO son el examen. El MAE y el R2 de la ficha se midieron",
        "en el holdout: 20 % de las filas, semilla 42. El tablero, si está",
        f"en modo honesto, resume {sklearn['n']} filas. Comparar ese n con",
        "n_filas mezcla el examen con la clase entera.",
        "",
        "## Métricas del tablero",
        "",
        "Las calcula `resumir_metricas`: los dos motores ven las mismas filas",
        "y el mismo `y_real`. Solo cambia la columna predicha.",
        "",
        "| Motor | n | MAE (€) | RMSE (€) | R2 | Sesgo (€) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| sklearn | {sklearn['n']} | {sklearn['mae']:.2f} | {sklearn['rmse']:.2f} "
            f"| {sklearn['r2']:.3f} | {sklearn['sesgo']:.2f} |"
        ),
        (
            f"| mlp | {mlp['n']} | {mlp['mae']:.2f} | {mlp['rmse']:.2f} "
            f"| {mlp['r2']:.3f} | {mlp['sesgo']:.2f} |"
        ),
        "",
        "## Quién gana",
        "",
        texto_ganador,
        "",
        "Lectura rápida del resto, sin esconderlo:",
        (
            f"- sklearn: RMSE {sklearn['rmse']:.2f} € frente a MAE {sklearn['mae']:.2f} €. "
            f"Sesgo {sklearn['sesgo']:.2f} € ({_lectura_sesgo(sklearn['sesgo'])})."
        ),
        (
            f"- mlp: RMSE {mlp['rmse']:.2f} € frente a MAE {mlp['mae']:.2f} €. "
            f"Sesgo {mlp['sesgo']:.2f} € ({_lectura_sesgo(mlp['sesgo'])})."
        ),
        "Si el RMSE queda claramente por encima del MAE, hay colas: pocos clientes",
        "con un fallo enorme pesan más que muchos fallos pequeños.",
        "",
        "## Tablero contra meta.json",
        "",
        "Solo MAE y R2 tienen pareja en la ficha. RMSE y sesgo no se guardaron",
        "al entrenar: no hay contra qué cuadrarlos.",
        "",
        "| Motor | MAE tablero | MAE ficha | Δ MAE | MAE | R2 tablero | R2 ficha | Δ R2 | R2 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
        *filas_comparacion,
        "",
        "Δ = tablero − ficha. «cuadra» = la diferencia cabe en el redondeo",
        "(1 céntimo en el MAE, 0.001 en el R2). Si no cuadra, o el holdout",
        "no es el de la semilla 42 o el joblib no es el de esta ficha.",
        "",
        "## El R2, dicho entero",
        "",
        (
            f"R2 del tablero: sklearn = {sklearn['r2']:.3f}, "
            f"mlp = {mlp['r2']:.3f}."
        ),
        (
            f"R2 de meta.json: sklearn = {_fmt(r2_ficha['sklearn'], 3)}, "
            f"mlp = {_fmt(r2_ficha['mlp'], 3)}. En esta clase **rondan 0.05**."
        ),
        "Si el tablero cuadra con la ficha, el 0.05 se ve también arriba,",
        "con tres decimales. No se redondea a «el modelo funciona».",
        "",
        "Un R2 de 0.0 significa: el modelo no mejora a predecir siempre la media",
        "de `y` (el gasto medio del holdout). 1.0 sería acertar cada cliente.",
        "0.05 es apenas mejor que decir la media. No es un éxito escondido",
        "detrás de un MAE en euros: es el resultado, y el informe lo escribe",
        "con tres decimales.",
        "",
        "## Glosario",
        "",
        "### MAE — error absoluto medio",
        "",
        "Media de |predicho − real|. Unidad: euros.",
        "«De media, fallamos por tantos euros». Es la métrica de meta.json",
        "y la regla del ganador de este informe. No dice si el fallo fue",
        "hacia arriba o hacia abajo: el valor absoluto borra el signo.",
        "",
        "### RMSE — raíz del error cuadrático medio",
        "",
        "Como el MAE, pero cada error se eleva al cuadrado antes de promediar",
        "y al final se saca la raíz. Un fallo de 10.000 € pesa mucho más que",
        "cien fallos de 100 €. Si RMSE >> MAE, unos pocos clientes destrozan",
        "la media. También va en euros.",
        "",
        "### R2 — coeficiente de determinación",
        "",
        "¿Mejoramos a predecir siempre la media?",
        "",
        "- 1.0 = cada predicción coincide con la realidad (no pasa en la calle).",
        "- 0.0 = el modelo no aporta nada frente a decir siempre la media.",
        "- menor que 0 = peor que la media: el modelo estorba.",
        "",
        "El de esta clase ronda 0.05. Se dice. No se esconde.",
        "",
        "### Sesgo",
        "",
        "Media de (predicho − real), ahora CON signo.",
        "",
        "- Mayor que 0: optimista. De media, el modelo se pasa.",
        "- Menor que 0: pesimista. De media, se queda corto.",
        "- Cerca de 0 con un MAE grande: no se va siempre para el mismo lado;",
        "  se equivoca en las dos direcciones.",
        "",
    ]
    return "\n".join(lineas)


def informe_bytes(monitoreo: pd.DataFrame, meta: dict) -> bytes:
    """
    Qué construimos: el mismo acta, en bytes.

    informe_markdown devuelve str. Para descargarlo o escribirlo
    hace falta la misma conversión que el CSV: texto → bytes UTF-8.
    Así las tildes del glosario no se rompen al guardar el archivo.
    """
    texto = informe_markdown(monitoreo, meta)
    # encode("utf-8")
    #   RECIBE: el Markdown (str), con tildes y el símbolo €.
    #   DEVUELVE: bytes codificados en UTF-8.
    return texto.encode("utf-8")


def _fmt(valor, decimales: int) -> str:
    """Número de la ficha → texto. None (clave ausente) → raya."""
    if valor is None:
        return "—"
    return f"{float(valor):.{decimales}f}"


def _delta(tablero: float, ficha, decimales: int) -> str:
    """tablero − ficha, con signo. Sin ficha no hay resta."""
    if ficha is None:
        return "—"
    return f"{tablero - float(ficha):+.{decimales}f}"


def _cuadra(tablero: float, ficha, tolerancia: float) -> str:
    """¿La diferencia cabe en el redondeo? Sin ficha, no se puede saber."""
    if ficha is None:
        return "sin ficha"
    if abs(tablero - float(ficha)) <= tolerancia:
        return "cuadra"
    return "no cuadra"


def _lectura_sesgo(sesgo: float) -> str:
    """Una frase corta según el signo. El número ya va en la tabla."""
    if sesgo > 0:
        return "optimista: predice de más, de media"
    if sesgo < 0:
        return "pesimista: se queda corto, de media"
    return "sin sesgo medio"
