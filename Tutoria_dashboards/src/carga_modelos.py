"""
Qué estamos construyendo
------------------------
Abrir los ARTEFACTOS que ya existen. Hoy no hay fit(). No se entrena.

Artefacto = archivo que "congela" un trabajo anterior para no repetirlo.
Aquí hay cuatro, todos en modelos/:

    gasto_sklearn.joblib     Random Forest ya entrenado
    gasto_mlp.joblib         red (Pipeline: StandardScaler + MLP)
    features_clientes.csv    1 fila por cliente: las 4 X + la y
    meta.json                ficha técnica: MAE, R2, versión, fecha de corte

joblib = librería que serializa objetos de Python a un archivo binario.
joblib.dump (en el entrenamiento) escribió el dict
    {"modelo", "features", "target", "version", "metricas"}
joblib.load (AQUÍ) lo reconstruye en RAM. No es un CSV: no lo abras
con el bloc de notas.

Diferencia con la clase front-back
----------------------------------
Allí el front NO hacía joblib.load: le hablaba a FastAPI por HTTP.
Aquí SÍ cargamos el cerebro. El tablero es una herramienta INTERNA
(el data scientist / el analista), no un mostrador de cliente.

¿Por qué no pegarle 2.000 POST a la API?
    1. Necesitamos y_real, y la API no lo devuelve.
    2. Un lote en local tarda décimas; 2.000 HTTP tardan minutos.
    3. Monitorear no es "servir una predicción": es auditar el modelo.
"""

from __future__ import annotations

import json
from pathlib import Path

# joblib.load: bytes del disco → objeto Python (el dict-paquete).
import joblib
import pandas as pd

from src.features_modelo import COLUMNA_TARGET, COLUMNAS_FEATURES

# Path(__file__) = este archivo (dashboards/src/carga_modelos.py).
# .parents[1]    = un nivel arriba = dashboards/  (la raíz de ESTA clase).
RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "modelos"


class ErrorModeloNoEncontrado(RuntimeError):
    """
    Excepción NUESTRA (no la genérica FileNotFoundError).

    RuntimeError = "el programa no puede seguir, pero no es un bug de
    sintaxis": falta un archivo que debería estar. En clase el mensaje
    dice DÓNDE copiarlo, no solo "No such file".
    """


def _exigir(ruta: Path) -> Path:
    """
    Portero. Si el archivo no existe, paramos con un texto útil.

    Path.exists() = True/False. No lanza nada: por eso preguntamos
    nosotros y lanzamos ErrorModeloNoEncontrado.
    El _ del nombre = "función interna, el notebook no la llama".
    """
    if not ruta.exists():
        raise ErrorModeloNoEncontrado(
            f"No está {ruta.name}. Debe vivir en dashboards/modelos/ "
            "(copia de la clase FastAPI)."
        )
    return ruta


def cargar_paquete(nombre_archivo: str) -> dict:
    """
    Qué construimos: "ábreme el cerebro y comprueba que es el de esta clase".

    joblib.load(ruta) → dict:
        modelo     objeto sklearn (RandomForest o Pipeline). Tiene .predict()
        features   lista de 4 nombres. Debe == COLUMNAS_FEATURES
        target     "gasto_siguiente_periodo"
        version    "1.0.0" (versión del ARTEFACTO, no de sklearn)
        metricas   {"mae", "r2", "segundos"} medidas en el holdout

    La comprobación de columnas evita un fallo silencioso: si alguien
    reentrenó con otras features, predict() a veces no peta y predice
    números que no significan nada. Mejor un RuntimeError ahora.
    """
    ruta = _exigir(CARPETA / nombre_archivo)
    paquete = joblib.load(ruta)
    if paquete.get("features") != COLUMNAS_FEATURES:
        raise RuntimeError(
            f"{nombre_archivo} se entrenó con otras columnas: {paquete.get('features')}"
        )
    return paquete


def cargar_tabla_evaluacion() -> pd.DataFrame:
    """
    Qué construimos: la TABLA con la que vamos a juzgar al modelo.

    CSV congelado = snapshot. No se regenera en cada clase.
    Columnas: cliente_id + 4 features + gasto_siguiente_periodo (el y_real).

    Índice por defecto de pandas: 0, 1, 2, ... n-1.
    NO hacemos set_index("cliente_id").

    ¿Por qué importa el índice?
        train_test_split (en el entrenamiento) barajó estas FILAS con
        semilla 42. Si cambias el índice a cliente_id, el 20 % que sale
        ya NO es el mismo 20 % de meta.json. El MAE "no cuadra" y parece
        que el código está mal, cuando lo que está mal es el índice.
    """
    ruta = _exigir(CARPETA / "features_clientes.csv")
    tabla = pd.read_csv(ruta)
    faltan = [c for c in COLUMNAS_FEATURES + [COLUMNA_TARGET] if c not in tabla.columns]
    if faltan:
        raise RuntimeError(f"{ruta.name} no tiene {faltan}")
    return tabla


def cargar_meta() -> dict:
    """
    Qué construimos: la FICHA TÉCNICA del entrenamiento (texto JSON).

    JSON = JavaScript Object Notation. Es texto, se abre en el editor.
    json.loads(str) → dict de Python.

    Claves que usamos hoy:
        version, fecha_corte, horizonte_dias, features, target
        sklearn: {mae, r2, segundos}
        mlp:     {mae, r2, segundos}
        n_filas

    Esos MAE/R2 se midieron en el 20 % de TEST (holdout), no en todo
    el CSV. El tablero, si está en modo honesto, debe REPETIR esos
    números. Si no los repite, o partimos mal o cargamos otro joblib.
    """
    ruta = _exigir(CARPETA / "meta.json")
    return json.loads(ruta.read_text(encoding="utf-8"))
