# Instalación — Front + Back

```powershell
conda activate py_env
cd C:\Programacion\DataScienceIA\Docente_Dev_Senior_code\files\persisitencia\backend_frontend
pip install -r requirements.txt
```

Los `.joblib` ya están en `modelos/`.

Comprobar:

```powershell
python -c "import fastapi, streamlit, httpx; print('ok')"
```
