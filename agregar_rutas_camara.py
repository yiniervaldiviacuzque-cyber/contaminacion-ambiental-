from pathlib import Path

path = Path("app.py")
source = path.read_text(encoding="utf-8")

marker = 'if __name__ == "__main__":'
route_marker = '@app.route("/predict_camera"'

if route_marker in source:
    print("La ruta /predict_camera ya existe. No se hicieron cambios.")
    raise SystemExit(0)

if marker not in source:
    print("ERROR: no se encontró el bloque de inicio de app.py. No se hicieron cambios.")
    raise SystemExit(1)

routes = '''

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
        "image_url": image_url,
        "imagen_guardada": saved_file_id is not None,
        "resultado_guardado": saved_prediction,
    }, 200


'''

updated = source.replace(marker, routes + marker, 1)
path.write_text(updated, encoding="utf-8")
print("Rutas de camara agregadas correctamente a app.py.")
