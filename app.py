import os
import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from dotenv import load_dotenv
from PIL import Image, ImageOps
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from pymongo import MongoClient, DESCENDING
from gridfs import GridFSBucket
from werkzeug.utils import secure_filename
from torchvision import models, transforms


# =========================================================
# 1. CONFIGURACI?N GENERAL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cambia-esta-clave-en-env")

MODEL_PATH = BASE_DIR / "model" / "modelo_contaminacion_hco2026.pth"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # MÃ¡ximo: 10 MB

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# Las categorÃ­as deben respetar el orden usado al entrenar el modelo.
CLASSES = [
    "incendio_forestal",
    "humo_contaminacion",
    "mineria_ilegal",
    "basura_contaminacion",
    "rio_contaminado",
    "ambiente_sano",
]

CLASS_NAMES = {
    "incendio_forestal": "Incendio forestal",
    "humo_contaminacion": "Humo y contaminaciÃ³n",
    "mineria_ilegal": "MinerÃ­a ilegal",
    "basura_contaminacion": "Basura y contaminaciÃ³n",
    "rio_contaminado": "RÃ­o contaminado",
    "ambiente_sano": "Ambiente sano",
}

CLASS_DESCRIPTIONS = {
    "incendio_forestal": "Posibles seÃ±ales visuales de incendios forestales.",
    "humo_contaminacion": "Posibles seÃ±ales visuales de humo o contaminaciÃ³n del aire.",
    "mineria_ilegal": "Posibles seÃ±ales visuales de actividad minera.",
    "basura_contaminacion": "Posibles residuos acumulados o contaminaciÃ³n por basura.",
    "rio_contaminado": "Posibles seÃ±ales visuales de contaminaciÃ³n del agua.",
    "ambiente_sano": "La imagen presenta caracterÃ­sticas visuales asociadas a un ambiente sano.",
}


# =========================================================
# 2. VARIABLES GLOBALES
# =========================================================

mongo_client = None
db = None
fs = None

model = None
model_classes = CLASSES.copy()
model_error = "El modelo entrenado todavÃ­a no estÃ¡ disponible."
model_arch = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. CONEXI?N A MONGODB ATLAS
# =========================================================

def connect_mongodb():
    """Conecta con MongoDB Atlas usando las variables de .env."""
    global mongo_client, db, fs

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "INNOBEST")

    if not uri:
        print("MongoDB: falta configurar MONGODB_URI en .env")
        return False

    try:
        if mongo_client is not None:
            mongo_client.close()

        mongo_client = MongoClient(
            uri,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )

        # Verifica que la conexiÃ³n y las credenciales funcionen.
        mongo_client.admin.command("ping")

        # Utiliza el nombre configurado en .env.
        db = mongo_client[db_name]
        fs = GridFSBucket(db, bucket_name="imagenes_analizadas")

        print(f"MongoDB conectado. Base de datos: {db_name}")
        return True

    except Exception as exc:
        print(f"MongoDB: no se pudo conectar ({type(exc).__name__}).")
        mongo_client = None
        db = None
        fs = None
        return False


def mongodb_ok():
    """Comprueba si existe una conexiÃ³n configurada."""
    return mongo_client is not None and db is not None and fs is not None


def count_predictions():
    """Cuenta los anÃ¡lisis guardados."""
    if not mongodb_ok():
        return 0

    try:
        return db.predicciones.count_documents({})
    except Exception:
        return 0


def get_recent_predictions(limit=10):
    """Obtiene los Ãºltimos anÃ¡lisis registrados."""
    if not mongodb_ok():
        return []

    try:
        return list(
            db.predicciones.find(
                {},
                {
                    "_id": 1,
                    "filename": 1,
                    "categoria": 1,
                    "categoria_nombre": 1,
                    "confianza": 1,
                    "fecha": 1,
                },
            )
            .sort("fecha", DESCENDING)
            .limit(limit)
        )
    except Exception:
        return []


# =========================================================
# 4. CONSTRUCCI?N DE LA ARQUITECTURA DEL MODELO
# =========================================================

def build_model(architecture, num_classes):
    """Construye una arquitectura compatible con el checkpoint."""
    architecture = str(architecture).lower().strip()

    if architecture in {"efficientnet_b0", "efficientnet-b0"}:
        net = models.efficientnet_b0(weights=None)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture in {"resnet18", "resnet-18"}:
        net = models.resnet18(weights=None)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)

    elif architecture in {"mobilenet_v3_small", "mobilenet-v3-small"}:
        net = models.mobilenet_v3_small(weights=None)
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Arquitectura no compatible: {architecture}. "
            "Arquitecturas admitidas: efficientnet_b0, resnet18 y mobilenet_v3_small."
        )

    return net


# =========================================================
# 5. CARGA DEL MODELO ENTRENADO
# =========================================================

def load_model():
    """Carga el archivo .pth si existe y tiene el formato esperado."""
    global model, model_classes, model_error, model_arch

    model = None
    model_arch = None

    if not MODEL_PATH.is_file():
        model_error = (
            "TodavÃ­a no se encuentra el modelo entrenado. "
            "Coloca el archivo modelo_contaminacion_hco2026.pth "
            "dentro de la carpeta model."
        )
        print(model_error)
        return False

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("El checkpoint debe ser un diccionario.")

        # Se espera un checkpoint guardado con estos campos.
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "El archivo .pth no contiene la clave model_state_dict."
            )

        architecture = checkpoint.get("arch", "efficientnet_b0")
        saved_classes = checkpoint.get("classes", CLASSES)

        if not isinstance(saved_classes, (list, tuple)) or not saved_classes:
            raise ValueError("La lista de clases del modelo no es vÃ¡lida.")

        if len(saved_classes) != len(set(saved_classes)):
            raise ValueError("El checkpoint contiene clases duplicadas.")

        if len(saved_classes) != len(CLASSES):
            raise ValueError(
                f"Se esperaban {len(CLASSES)} clases, "
                f"pero el modelo declara {len(saved_classes)}."
            )

        net = build_model(architecture, len(saved_classes))
        net.load_state_dict(checkpoint["model_state_dict"], strict=True)
        net.to(device)
        net.eval()

        model = net
        model_classes = list(saved_classes)
        model_arch = architecture
        model_error = ""

        print(f"Modelo cargado correctamente: {architecture}")
        print(f"Dispositivo de inferencia: {device}")
        return True

    except Exception as exc:
        model = None
        model_error = (
            "No se pudo cargar el modelo. Verifica que el archivo .pth "
            "corresponda a la arquitectura y al formato esperado."
        )
        print(f"Error al cargar el modelo: {type(exc).__name__}: {exc}")
        return False


# =========================================================
# 6. PREPROCESAMIENTO DE IMÃGENES
# =========================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def allowed_file(filename):
    """Comprueba la extensiÃ³n permitida."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(image_bytes):
    """Abre la imagen, corrige su orientaciÃ³n y la prepara para PyTorch."""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = image_transform(image).unsqueeze(0)
    return image, tensor.to(device)


# =========================================================
# 7. CLASIFICACI?N DE IMÃGENES
# =========================================================

def predict_image(image_bytes):
    """Clasifica una imagen y devuelve las probabilidades de cada clase."""
    if model is None:
        raise RuntimeError("El modelo entrenado no está disponible.")

    image, tensor = prepare_image(image_bytes)

    with torch.inference_mode():
        outputs = model(tensor)

        if outputs.ndim != 2 or outputs.shape[1] != len(model_classes):
            raise ValueError(
                "La salida del modelo no coincide con la cantidad de clases."
            )

        probabilities_tensor = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities_tensor, dim=0)

    category = model_classes[int(predicted_index.item())]

    probabilities = []
    for index, probability in enumerate(probabilities_tensor.tolist()):
        class_name = model_classes[index]
        probabilities.append({
            "categoria": class_name,
            "categoria_nombre": CLASS_NAMES.get(
                class_name,
                class_name.replace("_", " ").title(),
            ),
            "porcentaje": round(float(probability) * 100, 2),
        })

    probabilities.sort(key=lambda item: item["porcentaje"], reverse=True)

    return {
        "image": image,
        "categoria": category,
        "categoria_nombre": CLASS_NAMES.get(
            category,
            category.replace("_", " ").title(),
        ),
        "confianza": round(float(confidence.item()) * 100, 2),
        "descripcion": CLASS_DESCRIPTIONS.get(
            category,
            "Resultado de la clasificación visual.",
        ),
        "probabilidades": probabilities,
    }

# =========================================================
# 8. ALMACENAMIENTO DE IMÃGENES EN GRIDFS
# =========================================================

def save_image_to_gridfs(image_bytes, filename, content_type):
    """Guarda la imagen original en MongoDB GridFS."""
    if not mongodb_ok():
        return None

    try:
        return fs.upload_from_stream(
            filename,
            io.BytesIO(image_bytes),
            metadata={
                "contentType": content_type or "application/octet-stream",
                "fecha": datetime.now(timezone.utc),
            },
        )
    except Exception as exc:
        print(f"ERROR DETALLADO AL GUARDAR IMAGEN: {exc!r}")
        return None


def get_image_from_gridfs(file_id):
    """Recupera una imagen de GridFS."""
    if not mongodb_ok():
        return None, None

    try:
        file_info = db["imagenes_analizadas.files"].find_one(
            {"_id": file_id}
        )

        if not file_info:
            return None, None

        output = io.BytesIO()
        fs.download_to_stream(file_id, output)
        output.seek(0)

        content_type = (
            file_info.get("metadata", {}).get("contentType")
            or "application/octet-stream"
        )

        return output, content_type

    except Exception:
        return None, None


# =========================================================
# 9. PÃGINA PRINCIPAL
# =========================================================

@app.route("/", methods=["GET"])
def index():
    """Muestra el panel principal de INNOBEST."""
    recent = get_recent_predictions()

    # Agrega nombres legibles para el HTML.
    for item in recent:
        category = item.get("categoria", "")
        item["categoria_nombre"] = item.get(
            "categoria_nombre",
            CLASS_NAMES.get(category, category.replace("_", " ").title()),
        )

    return render_template(
        "index.html",
        model_ok=model is not None,
        model_error=model_error,
        mongo_ok=mongodb_ok(),
        count=count_predictions(),
        recent=recent,
        classes=CLASS_NAMES,
    )


# =========================================================
# 10. PROCESAMIENTO DE IMÃGENES
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Recibe una imagen, la clasifica y guarda el resultado."""
    if model is None:
        load_model()

    if model is None:
        flash(
            "La clasificaciÃ³n todavÃ­a no estÃ¡ disponible: falta cargar "
            "un modelo entrenado vÃ¡lido.",
            "error",
        )
        return redirect(url_for("index"))

    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        flash("Selecciona una imagen antes de continuar.", "error")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash(
            "Formato no permitido. Utiliza JPG, JPEG, PNG o WEBP.",
            "error",
        )
        return redirect(url_for("index"))

    image_bytes = uploaded_file.read()

    if not image_bytes:
        flash("El archivo seleccionado estÃ¡ vacÃ­o.", "error")
        return redirect(url_for("index"))

    if len(image_bytes) > MAX_CONTENT_LENGTH:
        flash("La imagen supera el lÃ­mite permitido de 10 MB.", "error")
        return redirect(url_for("index"))

    try:
        result = predict_image(image_bytes)
    except Exception as exc:
        print(f"Error durante la predicciÃ³n: {type(exc).__name__}: {exc}")
        flash(
            "No se pudo analizar la imagen. Comprueba que sea una imagen vÃ¡lida.",
            "error",
        )
        return redirect(url_for("index"))

    original_name = secure_filename(uploaded_file.filename) or "imagen"
    content_type = uploaded_file.mimetype or "application/octet-stream"

    # Los resultados se muestran aunque MongoDB estÃ© temporalmente desconectado.
    saved_file_id = save_image_to_gridfs(
        image_bytes,
        original_name,
        content_type,
    )

    if mongodb_ok():
        try:
            document = {
                "filename": original_name,
                "categoria": result["categoria"],
                "categoria_nombre": result["categoria_nombre"],
                "confianza": result["confianza"],
                "descripcion": result["descripcion"],
                "fecha": datetime.now(timezone.utc),
                "image_id": saved_file_id,
                "modelo": model_arch,
            }

            db.predicciones.insert_one(document)

        except Exception as exc:
            print(f"No se pudo guardar la predicciÃ³n: {type(exc).__name__}")
            flash(
                "La imagen fue analizada, pero no se pudo guardar el resultado.",
                "warning",
            )
    else:
        flash(
            "La imagen fue analizada, pero MongoDB no estÃ¡ conectado; "
            "el resultado no se guardÃ³.",
            "warning",
        )

    # Muestra el resultado usando la plantilla principal.
    # La imagen analizada se envÃ­a en formato JPEG.
    image_buffer = io.BytesIO()
    result["image"].save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    return render_template(
        "index.html",
        model_ok=model is not None,
        model_error=model_error,
        mongo_ok=mongodb_ok(),
        count=count_predictions(),
        recent=get_recent_predictions(),
        classes=CLASS_NAMES,
        result=result,
        resultado=result,
        imagen_analizada=image_buffer.getvalue(),
    )


# =========================================================
# 11. CONSULTA DE ESTADO
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    """Devuelve el estado tÃ©cnico de la aplicaciÃ³n."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mongodb_connected": mongodb_ok(),
        "database": os.getenv("MONGODB_DB", "INNOBEST"),
        "device": str(device),
    }


# =========================================================
# 12. VISUALIZACI?N DE IMÃGENES GUARDADAS
# =========================================================

@app.route("/imagen/<file_id>", methods=["GET"])
def show_image(file_id):
    """Muestra una imagen guardada en GridFS."""
    from bson import ObjectId

    try:
        object_id = ObjectId(file_id)
    except Exception:
        return "Identificador de imagen invÃ¡lido.", 400

    image_stream, content_type = get_image_from_gridfs(object_id)

    if image_stream is None:
        return "Imagen no encontrada o base de datos no disponible.", 404

    return send_file(
        image_stream,
        mimetype=content_type,
        download_name="imagen_analizada",
    )


# =========================================================
# 13. MANEJO DE ERRORES
# =========================================================

@app.errorhandler(413)
def file_too_large(_error):
    flash("El archivo supera el lÃ­mite permitido de 10 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_error):
    return "La pÃ¡gina solicitada no existe.", 404


# =========================================================
# 14. INICIALIZACI?N
# =========================================================



# =========================================================
# 15. CAMARA AMBIENTAL Y ANALISIS DE IMAGENES
# =========================================================

@app.route("/camara", methods=["GET"])
def camera_page():
    """Muestra la interfaz de la camara ambiental."""
    return render_template("camara.html")


@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    """Analiza una imagen capturada por la camara y guarda el resultado."""
    if model is None:
        load_model()

    if model is None:
        return {"error": "El modelo de IA no esta disponible."}, 503

    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        return {"error": "No se recibio ninguna imagen."}, 400

    if not allowed_file(uploaded_file.filename):
        return {"error": "Formato no permitido. Usa JPG, JPEG, PNG o WEBP."}, 400

    image_bytes = uploaded_file.read()

    if not image_bytes:
        return {"error": "La imagen esta vacia."}, 400

    if len(image_bytes) > MAX_CONTENT_LENGTH:
        return {"error": "La imagen supera el limite de 10 MB."}, 413

    try:
        result = predict_image(image_bytes)
    except Exception as exc:
        print(f"Error en analisis de camara: {type(exc).__name__}: {exc}")
        return {"error": "No se pudo analizar la imagen. Comprueba el archivo."}, 400

    original_name = secure_filename(uploaded_file.filename) or "captura.jpg"
    content_type = uploaded_file.mimetype or "image/jpeg"

    saved_file_id = save_image_to_gridfs(
        image_bytes,
        original_name,
        content_type,
    )

    saved_prediction = False

    if mongodb_ok():
        try:
            document = {
                "filename": original_name,
                "categoria": result["categoria"],
                "categoria_nombre": result["categoria_nombre"],
                "confianza": result["confianza"],
                "descripcion": result["descripcion"],
                "fecha": datetime.now(timezone.utc),
                "image_id": saved_file_id,
                "modelo": model_arch,
                "origen": "camara",
            }
            db.predicciones.insert_one(document)
            saved_prediction = True
        except Exception as exc:
            print(f"No se pudo guardar el resultado de camara: {type(exc).__name__}")

    image_url = None
    if saved_file_id is not None:
        image_url = url_for("show_image", file_id=str(saved_file_id))

    return {
        "categoria": result["categoria"],
        "categoria_nombre": result["categoria_nombre"],
        "confianza": result["confianza"],
        "descripcion": result["descripcion"],
        "probabilidades": result.get("probabilidades", []),
        "image_url": image_url,
        "imagen_guardada": saved_file_id is not None,
        "resultado_guardado": saved_prediction,
    }, 200



@app.route("/api/camara/historial", methods=["GET"])
def camera_history():
    """Devuelve los análisis recientes para la galería."""
    if not mongodb_ok():
        return {"error": "MongoDB no está conectado.", "items": []}, 503

    try:
        items = []
        cursor = db.predicciones.find(
            {},
            {
                "filename": 1,
                "categoria": 1,
                "categoria_nombre": 1,
                "confianza": 1,
                "descripcion": 1,
                "fecha": 1,
                "image_id": 1,
            },
        ).sort("fecha", DESCENDING).limit(40)

        for doc in cursor:
            image_id = doc.get("image_id")
            fecha = doc.get("fecha")

            items.append({
                "id": str(doc.get("_id", "")),
                "filename": doc.get("filename", "imagen"),
                "categoria": doc.get("categoria", ""),
                "categoria_nombre": doc.get(
                    "categoria_nombre", "Sin categoría"
                ),
                "confianza": doc.get("confianza", 0),
                "descripcion": doc.get("descripcion", ""),
                "fecha": fecha.strftime("%d/%m/%Y %H:%M")
                    if fecha else "Fecha no disponible",
                "image_url": url_for(
                    "show_image", file_id=str(image_id)
                ) if image_id is not None else None,
            })

        return {"items": items}

    except Exception as exc:
        print(
            f"Error al cargar galería: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "error": "No se pudo consultar el historial.",
            "items": [],
        }, 500


# =========================================================
# 16. INICIALIZACIÓN PARA RENDER Y GUNICORN
# =========================================================

# Estas funciones deben ejecutarse cuando Gunicorn importa app.py.
# También permiten iniciar la aplicación localmente.

connect_mongodb()
load_model()



# =========================================================
# 16. INICIALIZACIÓN PARA RENDER Y GUNICORN
# =========================================================

# Estas funciones deben ejecutarse cuando Gunicorn importa app.py.
# También permiten iniciar la aplicación localmente.

connect_mongodb()
load_model()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
