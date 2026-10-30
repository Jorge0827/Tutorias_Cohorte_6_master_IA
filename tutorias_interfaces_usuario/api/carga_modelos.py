"""
Qué estamos construyendo: abrir el cerebro en el BACK.
El front NUNCA llama a estas funciones.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features_modelo import COLUMNAS_FEATURES

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "modelos"


class ErrorModeloNoEncontrado(RuntimeError):
    """Falta el .joblib. Mejor un error claro al arrancar que un 500 en cada POST."""


def _exigir(ruta: Path) -> Path:
    """Si el archivo no está, error entendible."""
    if not ruta.exists():
        raise ErrorModeloNoEncontrado(f"No está {ruta.name} en modelos/.")
    return ruta


def cargar_paquete(nombre_archivo: str) -> dict:
    """
    joblib.load reconstruye el dict {modelo, features, version, metricas}.
    Si las columnas no coinciden con COLUMNAS_FEATURES, no predijamos a ciegas.
    """
    paquete = joblib.load(_exigir(CARPETA / nombre_archivo))
    if paquete.get("features") != COLUMNAS_FEATURES:
        raise RuntimeError(f"Columnas distintas: {paquete.get('features')}")
    return paquete


def cargar_lookup() -> pd.DataFrame:
    """CSV → tabla indexada por cliente_id (para GET /predecir/cliente/{id})."""
    return pd.read_csv(_exigir(CARPETA / "features_clientes.csv")).set_index("cliente_id")


def cargar_meta() -> dict:
    """meta.json → dict (versión, MAE)."""
    return json.loads(_exigir(CARPETA / "meta.json").read_text(encoding="utf-8"))


def features_de_cliente(lookup: pd.DataFrame, cliente_id: int) -> dict:
    """Saca las 4 cifras del PASADO. No incluye el target."""
    fila = lookup.loc[cliente_id]
    return {
        "num_compras": int(fila["num_compras"]),
        "gasto_historico": float(fila["gasto_historico"]),
        "ticket_medio": float(fila["ticket_medio"]),
        "cantidad_total": int(fila["cantidad_total"]),
    }
