from pathlib import Path
import re

path = Path("app.py")
source = path.read_text(encoding="utf-8-sig")

pattern = r"def predict_image\(image_bytes\):.*?(?=\n# ={10,}\n# 8\.)"

replacement = '''def predict_image(image_bytes):
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
'''

updated, count = re.subn(pattern, replacement, source, count=1, flags=re.S)

if count != 1:
    raise SystemExit(
        "No se encontró el bloque completo de predict_image. "
        "No se modificó app.py; revisa el encabezado de la sección 8."
    )

old = '''        "descripcion": result["descripcion"],
        "image_url": image_url,
        "imagen_guardada": saved_file_id is not None,
        "resultado_guardado": saved_prediction,
'''

new = '''        "descripcion": result["descripcion"],
        "probabilidades": result.get("probabilidades", []),
        "image_url": image_url,
        "imagen_guardada": saved_file_id is not None,
        "resultado_guardado": saved_prediction,
'''

if updated.count(old) != 1:
    raise SystemExit(
        "No se encontró exactamente una respuesta de predict_camera. "
        "No se guardaron los cambios."
    )

updated = updated.replace(old, new, 1)
path.write_text(updated, encoding="utf-8")
print("OK: predict_image devuelve las probabilidades y predict_camera las envía.")
