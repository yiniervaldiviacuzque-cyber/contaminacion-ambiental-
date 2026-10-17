const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");
const fileInput = document.getElementById("file");
const resultBox = document.getElementById("result");
const diagnostic = document.getElementById("diagnostic");
const probabilitiesBox = document.getElementById("probabilities");
const gallery = document.getElementById("gallery");

const startButton = document.getElementById("start");
const captureButton = document.getElementById("capture");
const stopButton = document.getElementById("stop");
const analyzeButton = document.getElementById("analyze");
const clearButton = document.getElementById("clear");
const refreshButton = document.getElementById("refresh");

let cameraStream = null;
let selectedBlob = null;
let previewUrl = null;

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function setStatus(text) {
  resultBox.textContent = text;
}

function selectImage(blob) {
  if (!blob || !blob.type.startsWith("image/")) {
    setStatus("Selecciona un archivo de imagen válido.");
    return;
  }

  if (blob.size > 10 * 1024 * 1024) {
    setStatus("La imagen supera el límite de 10 MB.");
    return;
  }

  selectedBlob = blob;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);

  preview.src = previewUrl;
  preview.hidden = false;
  video.hidden = true;
  analyzeButton.disabled = false;

  diagnostic.textContent = "Imagen lista para analizar.";
  probabilitiesBox.replaceChildren();
  setStatus("Imagen preparada. Pulsa «Analizar con IA».");
}

async function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  video.srcObject = null;
  video.hidden = true;
  captureButton.disabled = true;
  stopButton.disabled = true;
}

startButton.addEventListener("click", async () => {
  try {
    await stopCamera();

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("La cámara requiere un navegador compatible y una conexión segura.");
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });

    video.srcObject = cameraStream;
    video.hidden = false;
    preview.hidden = true;

    await video.play();

    captureButton.disabled = false;
    stopButton.disabled = false;
    setStatus("Cámara activa. Encuadra la escena y toma la fotografía.");
  } catch (error) {
    setStatus("No se pudo activar la cámara. Comprueba los permisos y la conexión. " + error.message);
  }
});

captureButton.addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) {
    setStatus("Espera a que la cámara muestre la imagen.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    if (blob) {
      selectImage(blob);
      setStatus("Fotografía capturada. Puedes analizarla ahora.");
    } else {
      setStatus("No se pudo capturar la fotografía.");
    }
  }, "image/jpeg", 0.92);
});

stopButton.addEventListener("click", async () => {
  await stopCamera();
  setStatus("Cámara apagada.");
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) selectImage(file);
});

clearButton.addEventListener("click", () => {
  selectedBlob = null;
  fileInput.value = "";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  preview.removeAttribute("src");
  preview.hidden = true;
  analyzeButton.disabled = true;
  diagnostic.textContent = "El resultado del análisis aparecerá aquí.";
  probabilitiesBox.replaceChildren();
  setStatus("Selección limpiada. Puedes tomar otra fotografía.");
});

function showProbabilities(items) {
  probabilitiesBox.replaceChildren();

  if (!Array.isArray(items) || items.length === 0) {
    probabilitiesBox.textContent = "El servidor no devolvió las probabilidades por categoría.";
    return;
  }

  items.forEach(item => {
    const value = Math.max(0, Math.min(100, Number(item.porcentaje) || 0));
    const section = document.createElement("div");
    section.className = "prob";

    const heading = document.createElement("div");
    heading.className = "prob-head";

    const name = document.createElement("span");
    name.textContent = item.categoria_nombre || item.categoria || "Categoría";

    const percentage = document.createElement("strong");
    percentage.textContent = value.toFixed(2) + "%";

    heading.append(name, percentage);

    const track = document.createElement("div");
    track.className = "track";

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = value + "%";

    track.appendChild(bar);
    section.append(heading, track);
    probabilitiesBox.appendChild(section);
  });
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedBlob) {
    setStatus("Primero toma una fotografía o selecciona una imagen.");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedBlob, "captura.jpg");

  analyzeButton.disabled = true;
  setStatus("Analizando la imagen con el modelo de IA...");

  try {
    const response = await fetch("/predict_camera", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "El servidor no pudo analizar la imagen.");
    }

    diagnostic.replaceChildren();

    const category = document.createElement("div");
    category.className = "result-title";
    category.textContent = data.categoria_nombre || "Categoría no disponible";

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    confidence.textContent = (Number(data.confianza) || 0).toFixed(2) + "%";

    const caption = document.createElement("p");
    caption.className = "muted";
    caption.textContent = "Confianza estimada del modelo";

    const description = document.createElement("p");
    description.textContent = data.descripcion || "No hay descripción disponible.";

    diagnostic.append(category, confidence, caption, description);
    showProbabilities(data.probabilidades);

    const imageStatus = data.imagen_guardada
      ? "Fotografía guardada en MongoDB/GridFS."
      : "La fotografía no se pudo guardar en GridFS.";

    const resultStatus = data.resultado_guardado
      ? "Resultado guardado en MongoDB."
      : "El resultado no se confirmó como guardado en MongoDB.";

    setStatus("Análisis completado.\n" + imageStatus + "\n" + resultStatus);

    if (data.image_url) {
      const link = document.createElement("a");
      link.href = data.image_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Abrir fotografía guardada";
      resultBox.appendChild(document.createTextNode("\n"));
      resultBox.appendChild(link);
    }

    await loadGallery();
  } catch (error) {
    setStatus("Error durante el análisis: " + error.message);
  } finally {
    analyzeButton.disabled = !selectedBlob;
  }
});

async function loadGallery() {
  gallery.innerHTML = '<div class="empty">Consultando el historial...</div>';
  refreshButton.disabled = true;

  try {
    const response = await fetch("/api/camara/historial", {
      cache: "no-store"
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "No se pudo consultar MongoDB.");
    }

    if (!Array.isArray(data.items) || data.items.length === 0) {
      gallery.innerHTML = '<div class="empty">No hay fotografías guardadas todavía. Analiza una imagen para comenzar la galería.</div>';
      return;
    }

    gallery.innerHTML = data.items.map(item => {
      const image = item.image_url
        ? `<a href="${escapeHTML(item.image_url)}" target="_blank" rel="noopener"><img src="${escapeHTML(item.image_url)}" alt="Fotografía analizada" loading="lazy"></a>`
        : '<div class="empty">Fotografía no disponible</div>';

      return `<article class="item">
        ${image}
        <div class="item-content">
          <span class="badge">${escapeHTML(item.categoria_nombre || "Sin categoría")}</span>
          <p><strong>Confianza:</strong> ${escapeHTML(item.confianza)}%</p>
          <p><strong>Fecha:</strong> ${escapeHTML(item.fecha || "No disponible")}</p>
          <p><strong>Archivo:</strong> ${escapeHTML(item.filename || "imagen")}</p>
          <p class="muted">${escapeHTML(item.descripcion || "")}</p>
        </div>
      </article>`;
    }).join("");
  } catch (error) {
    gallery.innerHTML = "";
    const errorBox = document.createElement("div");
    errorBox.className = "empty";
    errorBox.textContent = "No se pudo cargar la galería: " + error.message;
    gallery.appendChild(errorBox);
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadGallery);

window.addEventListener("beforeunload", () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

loadGallery();
