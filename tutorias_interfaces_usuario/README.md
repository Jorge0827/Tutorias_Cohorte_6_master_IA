# Clase: conexión front-end y back-end

Carpeta **aislada**. El front (Streamlit) **no carga** el `.joblib`.
Manda JSON al back (FastAPI) y pinta tres estados: **carga, error, éxito**.

## Dos terminales

```powershell
conda activate py_env
cd ...\persisitencia\backend_frontend
pip install -r requirements.txt

# 1) BACK
uvicorn api.main:app --reload --port 8000

# 2) FRONT
streamlit run frontend/app.py
```

Se **escriben** `frontend/cliente_api.py` y `frontend/app.py` en el editor.
El notebook solo **prueba** después: `notebooks/01_conexion_frontend_backend.ipynb`

| Carpeta | Rol |
|---|---|
| `api/` | Backend. `joblib.load` + rutas HTTP |
| `frontend/` | Interfaz. Solo `httpx` |
| `modelos/` | Cerebro (copia). Solo lo toca el back |
| `docs/GUIA_DOCENTE.md` | Minutado |
