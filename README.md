# INNOBEST Huánuco 2026 — Contaminación Ambiental

Aplicación web en Python/Flask que utiliza el modelo entrenado en Google Colab para clasificar imágenes en:

1. Incendio forestal
2. Humo / contaminación
3. Minería ilegal
4. Basura / residuos
5. Río contaminado
6. Ambiente sano

Además calcula indicadores visuales de humo, fuego y residuos, muestra un nivel de contaminación y guarda el historial en MongoDB Atlas mediante GridFS para la imagen anotada.

## 1. Estructura

```text
INNOBEST_Contaminacion_Huanuco/
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
├── .env.example
├── .gitignore
├── README.md
├── model/
│   └── modelo_contaminacion_hco2026.pth   <-- colocar aquí
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── history.html
└── static/
    └── style.css
```

## 2. Modelo

El proyecto necesita el archivo `modelo_contaminacion_hco2026.pth` que se descargó desde Google Colab después del entrenamiento.

No se debe inventar ni reemplazar el modelo: hay que copiar el archivo real entrenado ayer dentro de `model/`.

## 3. Ejecutar en Visual Studio Code

Abrir la carpeta en VS Code y ejecutar:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Crear `.env` a partir de `.env.example` y colocar la cadena de conexión de MongoDB Atlas.

Luego:

```powershell
python app.py
```

Abrir:

```text
http://127.0.0.1:5000
```

## 4. MongoDB Atlas

La aplicación usa:

```text
Base de datos: INNOBEST
Colección: predicciones
```

La colección se crea automáticamente al guardar el primer análisis.

La imagen anotada se almacena mediante GridFS en MongoDB.

## 5. Variables para Render

En Render → Environment Variables:

```text
MONGODB_URI = tu_cadena_mongodb_atlas
SECRET_KEY = una_clave_segura
```

El proyecto usa el puerto `PORT` proporcionado por Render.

## 6. GitHub

Desde la carpeta del proyecto:

```powershell
git init
git add .
git commit -m "INNOBEST contaminacion ambiental"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPOSITORIO.git
git push -u origin main
```

IMPORTANTE: no subir `.env` ni contraseñas de MongoDB.

## 7. Render

En Render crear un Web Service conectado al repositorio de GitHub.

Build Command:

```text
pip install -r requirements.txt
```

Start Command:

```text
gunicorn app:app
```

Agregar `MONGODB_URI` en Environment Variables.

## 8. Evidencia para la exposición

Mostrar:

- aplicación funcionando en Render;
- carga de una imagen;
- clasificación;
- confianza;
- nivel de riesgo;
- porcentaje estimado;
- recomendación;
- historial;
- MongoDB Atlas con los documentos guardados;
- modelo real entrenado en Google Colab.

## 9. Nota sobre el porcentaje

El porcentaje de contaminación es un indicador visual basado en máscaras de color para humo, fuego y residuos. No debe presentarse como una medición científica de calidad del aire. La clasificación principal proviene del modelo entrenado.
