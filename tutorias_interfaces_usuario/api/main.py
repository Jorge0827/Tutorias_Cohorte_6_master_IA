"""
Qué estamos construyendo (esto YA está: se SEÑALA, no se reescribe hoy)
------------------------------------------------------------------------
El BACKEND. FastAPI carga el .joblib y responde JSON.

El front NO importa este archivo. Le habla por HTTP:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.carga_modelos import (
    ErrorModeloNoEncontrado,
    cargar_lookup,
    cargar_meta,
    cargar_paquete,
    features_de_cliente,
)
from api.esquemas import EntradaPrediccion, SalidaPrediccion
from src.features_modelo import COLUMNAS_FEATURES

# ESTADO: los modelos en RAM. Se rellena al arrancar, no en cada POST.
ESTADO: dict = {"sklearn": None, "mlp": None, "lookup": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta UNA vez al encender Uvicorn (antes del yield)
    y al apagar (después).
    Aquí va joblib.load. El front nunca llama a esto.
    """
    ESTADO["sklearn"] = cargar_paquete("gasto_sklearn.joblib")
    ESTADO["mlp"] = cargar_paquete("gasto_mlp.joblib")
    ESTADO["lookup"] = cargar_lookup()
    ESTADO["meta"] = cargar_meta()
    yield
    ESTADO.clear()


# FastAPI(...) crea la app. @app.get / @app.post registran rutas.
app = FastAPI(
    title="API de predicción de gasto",
    description="Backend de la clase front-end / back-end.",
    version="1.0.0",
    lifespan=lifespan,
)


def _asegurar() -> None:
    """Si TestClient no corrió el lifespan, carga ahora."""
    if ESTADO.get("sklearn") is not None:
        return
    ESTADO["sklearn"] = cargar_paquete("gasto_sklearn.joblib")
    ESTADO["mlp"] = cargar_paquete("gasto_mlp.joblib")
    ESTADO["lookup"] = cargar_lookup()
    ESTADO["meta"] = cargar_meta()


@app.get("/", tags=["salud"])
def raiz() -> dict:
    """GET /  → 'estoy vivo'. No predice."""
    return {"servicio": "prediccion_gasto", "docs": "/docs", "salud": "/salud"}


@app.get("/salud", tags=["salud"])
def salud() -> dict:
    """GET /salud → lo que pide pedir_salud() del front."""
    _asegurar()
    return {
        "estado": "ok",
        "sklearn": ESTADO["sklearn"]["version"],
        "mlp": ESTADO["mlp"]["version"],
        "clientes_en_lookup": int(len(ESTADO["lookup"])),
    }


def _predecir(paquete: dict, entrada: EntradaPrediccion, etiqueta: str) -> SalidaPrediccion:
    """
    Cocina común: 1 fila (orden COLUMNAS_FEATURES) → modelo.predict(x)[0].
    Pydantic YA validó entrada. Si no, no llegamos aquí.
    """
    inicio = time.perf_counter()
    x = pd.DataFrame(
        [
            {
                "num_compras": entrada.num_compras,
                "gasto_historico": entrada.gasto_historico,
                "ticket_medio": entrada.ticket_medio,
                "cantidad_total": entrada.cantidad_total,
            }
        ],
        columns=COLUMNAS_FEATURES,
    )
    valor = float(paquete["modelo"].predict(x)[0])
    return SalidaPrediccion(
        cliente_id=entrada.cliente_id,
        valor_predicho=round(valor, 2),
        modelo=etiqueta,
        version=paquete["version"],
        tiempo_ms=int((time.perf_counter() - inicio) * 1000),
    )


@app.post("/predecir/sklearn", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_sklearn(entrada: EntradaPrediccion) -> SalidaPrediccion:
    """POST: el front mandó las 4 cifras. Cerebro = bosque."""
    _asegurar()
    return _predecir(ESTADO["sklearn"], entrada, "gasto_sklearn")


@app.post("/predecir/red", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_red(entrada: EntradaPrediccion) -> SalidaPrediccion:
    """POST: mismo JSON. Cerebro = MLP."""
    _asegurar()
    return _predecir(ESTADO["mlp"], entrada, "gasto_mlp")


@app.get("/predecir/cliente/{cliente_id}", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_por_id(cliente_id: int, motor: Literal["sklearn", "red"] = "sklearn") -> SalidaPrediccion:
    """GET: solo el id. Si no está → HTTPException 404 (el front lo pinta)."""
    _asegurar()
    if cliente_id not in ESTADO["lookup"].index:
        raise HTTPException(404, f"El cliente {cliente_id} no está en el lookup.")
    nums = features_de_cliente(ESTADO["lookup"], cliente_id)
    entrada = EntradaPrediccion(cliente_id=cliente_id, **nums)
    paquete = ESTADO["sklearn"] if motor == "sklearn" else ESTADO["mlp"]
    etiqueta = "gasto_sklearn" if motor == "sklearn" else "gasto_mlp"
    return _predecir(paquete, entrada, etiqueta)


@app.exception_handler(ErrorModeloNoEncontrado)
async def _sin_modelo(_, error: ErrorModeloNoEncontrado) -> JSONResponse:
    """Si faltan los .joblib, 503 claro en vez de un 500 opaco."""
    return JSONResponse(status_code=503, content={"detail": str(error)})
