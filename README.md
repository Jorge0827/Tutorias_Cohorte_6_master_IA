## Tutorías Data Science e IA - Cohorte 6

Este repositorio contiene el material de las tutorías de la Cohorte 6 del programa de **Data Science e Inteligencia Artificial**.

### Estructura del repositorio

- **`tutoria1` / `Tutoria_1`**: Introducción, análisis exploratorio de datos (EDA) con el dataset de Titanic.
- **`Tutoria_2`**: Primeros ejercicios en Python (por ejemplo, `tutoria2.py`, `sumar.py`).
- **`Tutoria_3`**: Continuación de análisis con notebooks y dataset de Titanic.
- **`EDA.ipynb`**: Notebook principal de análisis exploratorio (ubicado en la raíz del proyecto o en las carpetas de tutoría, según el entorno local).
- **`titanic_tutoria_3.csv`**: Dataset de ejemplo utilizado en las tutorías.

La organización puede variar ligeramente entre tu entorno local (`Tutoria_1`, `Tutoria_2`, `Tutoria_3`) y la estructura en GitHub (`tutoria1`, notebooks en la raíz), pero el contenido corresponde a las mismas sesiones.

### Requisitos

Para ejecutar los notebooks y scripts se recomienda:

- Python 3.9 o superior.
- Tener instalado `pip` y un entorno virtual (opcional pero recomendado).

Paquetes típicos a utilizar en las tutorías:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `jupyter`

### Cómo ejecutar los notebooks

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/Jorge0827/tutorias-data-science-ia-cohorte6.git
   cd tutorias-data-science-ia-cohorte6
   ```

2. (Opcional) Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # En Windows
   .venv\Scripts\activate
   # En macOS / Linux
   source .venv/bin/activate
   ```

3. Instalar dependencias básicas:

   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

4. Lanzar Jupyter:

   ```bash
   jupyter notebook
   ```

5. Abrir el notebook `EDA.ipynb` o los notebooks dentro de cada carpeta de tutoría.

### Objetivo de las tutorías

El objetivo principal es aprender, mediante práctica guiada:

- Fundamentos de Python aplicados a ciencia de datos.
- Manejo de datos con `pandas`.
- Visualización exploratoria con `matplotlib` y `seaborn`.
- Preparación de datos para modelos de Machine Learning.

### Contribuciones

Este repositorio está pensado como material de apoyo para las sesiones, pero puedes:

- Añadir nuevos ejercicios o notebooks.
- Corregir errores tipográficos o de código.
- Proponer mejoras en el material.

Para ello, crea una rama, realiza tus cambios y abre un *pull request* en GitHub.

