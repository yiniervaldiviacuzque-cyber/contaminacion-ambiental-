from pathlib import Path

path = Path("app.py")
source = path.read_text(encoding="utf-8")

if '/api/camara/historial' in source:
    print("La ruta del historial ya existe en app.py.")
else:
    marker = 'if __name__ == "__main__":'

    if marker not in source:
        raise SystemExit(
            "No se encontró el bloque principal de app.py. "
            "No se modificó el archivo."
        )

    route = '''
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


'''

    path.write_text(
        source.replace(marker, route + marker, 1),
        encoding="utf-8"
    )
    print("Ruta /api/camara/historial agregada correctamente.")
