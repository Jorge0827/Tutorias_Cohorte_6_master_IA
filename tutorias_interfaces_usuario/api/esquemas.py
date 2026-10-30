"""
Qué estamos construyendo: el CONTRATO JSON del back (Pydantic).

El front manda un dict. FastAPI lo convierte en EntradaPrediccion.
Field(..., ge=0) = obligatorio y ≥ 0.
Si gasto_historico es -10 → 422 y _predecir() NO se ejecuta.
"""

from pydantic import BaseModel, Field


class EntradaPrediccion(BaseModel):
    """Body del POST. Los nombres = las 4 columnas del modelo."""

    cliente_id: int = Field(..., ge=1, examples=[8])
    num_compras: int = Field(..., ge=0)
    gasto_historico: float = Field(..., ge=0)
    ticket_medio: float = Field(..., ge=0)
    cantidad_total: int = Field(..., ge=0)


class SalidaPrediccion(BaseModel):
    """Lo que el front lee en resultado['datos'] cuando estado=exito."""

    cliente_id: int
    valor_predicho: float
    modelo: str
    version: str
    tiempo_ms: int
