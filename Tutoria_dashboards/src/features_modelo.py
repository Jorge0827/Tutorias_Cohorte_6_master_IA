"""
Qué estamos construyendo
------------------------
El CONTRATO de columnas. No carga datos. No predice. Solo nombra.

Esta carpeta es aislada: no importamos el src/ del padre. Por eso
esta lista vive AQUÍ otra vez (sí, está duplicada a propósito).

Dos familias de columnas, y no se mezclan:

    X  (features / variables explicativas)
       Lo que el modelo TIENE DERECHO a ver: el pasado del cliente
       ANTES del 2025-06-30.
           num_compras, gasto_historico, ticket_medio, cantidad_total

    y  (target / variable objetivo)
       Lo que queremos ACERTAR: cuánto gastó en los 90 días SIGUIENTES.
           gasto_siguiente_periodo

Analogía: X es el expediente que le das al tasador; y es lo que acabó
pagando el cliente tres meses después. Si le enseñas y al entrenar
como si fuera X, estás haciendo trampa (data leakage / fuga de datos).

Hoy NO reconstruimos X/y desde ventas.csv. Eso ya se hizo en FastAPI.
Leemos el CSV congelado modelos/features_clientes.csv. Estas listas
solo sirven para no equivocar el ORDEN: joblib.predict espera las
mismas 4 columnas, en este orden, que vio en el fit().
"""

# Orden FIJO. Si cambias el orden, predict() no petará: predice BASURA
# (usa ticket_medio donde esperaba num_compras, etc.).
COLUMNAS_FEATURES = [
    "num_compras",
    "gasto_historico",
    "ticket_medio",
    "cantidad_total",
]

# El target SÍ está en el CSV. En la clase de predicción (mostrador)
# NO se lo pasábamos al modelo: el modelo no puede "ver el futuro".
# En el DASHBOARD lo necesitamos: es el "qué ocurrió de verdad" (y_real)
# contra el que medimos el error.
COLUMNA_TARGET = "gasto_siguiente_periodo"
