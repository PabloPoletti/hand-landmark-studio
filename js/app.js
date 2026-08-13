import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";
import { CONNECTIONS, NAMES, colorFor } from "./schema.js";
import { HandStudio3D } from "./hand3d.js";
import { predictWilor, wilorToHands } from "./wilor.js";

const MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";

const statusEl = document.getElementById("engine-status");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const results = document.getElementById("results");
const toolbar = document.getElementById("toolbar");
const switcher = document.getElementById("hand-switcher");
const note = document.getElementById("detect-note");
const emptyMsg = document.getElementById("empty-msg");
const copyBtn = document.getElementById("copy-json");
const wilorBtn = document.getElementById("wilor-btn");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

let landmarker = null;
let studio = null;
let lastImage = null;
let lastFile = null;
let lastHands = [];
let activeIndex = 0;

function setStatus(text, ok = false) {
  statusEl.textContent = text;
  statusEl.style.borderColor = ok ? "#4ade80" : "";
}

async function createLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "IMAGE",
    numHands: 2,
    minHandDetectionConfidence: 0.4,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
}

async function initModel() {
  const vision = await FilesetResolver.forVisionTasks(WASM);
  try {
    landmarker = await createLandmarker(vision, "GPU");
  } catch {
    landmarker = await createLandmarker(vision, "CPU");
  }
  studio = new HandStudio3D();
  setStatus("Modelo listo · pegá una foto", true);
}

function drawOverlay(image, landmarks) {
  const maxW = 900;
  const scale = Math.min(1, maxW / image.width);
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const px = (lm) => ({ x: lm.x * canvas.width, y: lm.y * canvas.height });
  ctx.lineWidth = Math.max(2, canvas.width / 280);
  ctx.lineCap = "round";
  CONNECTIONS.forEach(([a, b]) => {
    const pa = px(landmarks[a]);
    const pb = px(landmarks[b]);
    ctx.strokeStyle = colorFor(b);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });

  landmarks.forEach((lm, i) => {
    const p = px(lm);
    const r = Math.max(4, canvas.width / 90);
    ctx.fillStyle = colorFor(i);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#111";
    ctx.font = `700 ${Math.max(10, canvas.width / 52)}px Segoe UI, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i), p.x, p.y);
  });
}

function handednessOf(hand) {
  return hand.handedness?.[0]?.categoryName || hand.handednesses?.[0]?.categoryName || "Unknown";
}

function renderActive() {
  const hand = lastHands[activeIndex];
  if (!hand || !lastImage) return;
  drawOverlay(lastImage, hand.landmarks);
  studio.setHand(hand.worldLandmarks, handednessOf(hand));
  const label = handednessOf(hand) === "Left" ? "izquierda" : "derecha";
  note.textContent = `Mano ${activeIndex + 1} · ${label} · 21 landmarks`;
  copyBtn.hidden = false;
  wilorBtn.hidden = false;
}

function showHands(image, detection) {
  lastImage = image;
  lastHands = (detection.landmarks || []).map((landmarks, i) => ({
    landmarks,
    worldLandmarks: detection.worldLandmarks[i],
    handedness: detection.handedness?.[i] || detection.handednesses?.[i],
  }));

  if (!lastHands.length) {
    results.hidden = true;
    toolbar.hidden = true;
    copyBtn.hidden = true;
    wilorBtn.hidden = true;
    emptyMsg.hidden = false;
    studio?.clear();
    return;
  }

  emptyMsg.hidden = true;
  results.hidden = false;
  toolbar.hidden = false;
  activeIndex = 0;
  switcher.innerHTML = "";
  lastHands.forEach((hand, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (i === 0 ? " active" : "");
    const side = handednessOf(hand) === "Left" ? "Left" : "Right";
    btn.textContent = `Mano ${i + 1} · ${side}`;
    btn.addEventListener("click", () => {
      activeIndex = i;
      [...switcher.children].forEach((c, k) => c.classList.toggle("active", k === i));
      renderActive();
    });
    switcher.appendChild(btn);
  });
  renderActive();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function processFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (!landmarker) {
    setStatus("El modelo todavía está cargando…");
    return;
  }
  setStatus("Detectando mano…");
  lastFile = file;
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    const detection = landmarker.detect(image);
    showHands(image, detection);
    setStatus(`${lastHands.length} mano${lastHands.length === 1 ? "" : "s"} detectada${lastHands.length === 1 ? "" : "s"}`, true);
  } catch (err) {
    console.error(err);
    setStatus("No se pudo leer la imagen");
  } finally {
    URL.revokeObjectURL(url);
  }
}

function onPaste(event) {
  const items = event.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      event.preventDefault();
      processFile(item.getAsFile());
      return;
    }
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) processFile(file);
});
["dragenter", "dragover"].forEach((type) => {
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((type) => {
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) processFile(file);
});
window.addEventListener("paste", onPaste);

wilorBtn.addEventListener("click", async () => {
  if (!lastFile || !lastImage) return;
  wilorBtn.disabled = true;
  try {
    const result = await predictWilor(lastFile, setStatus);
    const hands = wilorToHands(result);
    if (!hands.length) {
      setStatus("WiLoR no encontró manos");
      return;
    }
    showHands(lastImage, {
      landmarks: hands.map((h) => h.landmarks),
      worldLandmarks: hands.map((h) => h.worldLandmarks),
      handedness: hands.map((h) => h.handedness),
    });
    setStatus(`WiLoR · ${hands.length} mano${hands.length === 1 ? "" : "s"}`, true);
  } catch (err) {
    console.error(err);
    setStatus(err.message || "WiLoR no está disponible");
  } finally {
    wilorBtn.disabled = false;
  }
});

copyBtn.addEventListener("click", async () => {
  const hand = lastHands[activeIndex];
  if (!hand) return;
  const payload = [
    {
      hand: activeIndex + 1,
      handedness: handednessOf(hand),
      landmarks: hand.landmarks.map((lm, id) => ({
        id,
        name: NAMES[id],
        x: Number(lm.x.toFixed(5)),
        y: Number(lm.y.toFixed(5)),
        z: Number(lm.z.toFixed(5)),
      })),
    },
  ];
  await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
  copyBtn.textContent = "Copiado";
  setTimeout(() => {
    copyBtn.textContent = "Copiar JSON 21 pts";
  }, 1200);
});

initModel().catch((err) => {
  console.error(err);
  setStatus("No se pudo cargar MediaPipe");
});
