import { CONNECTIONS, NAMES, colorFor } from "./schema.js";
import { predictWilor, wilorToHands } from "./wilor.js";

const LOCAL_BASE = new URL("../vendor/mediapipe/", import.meta.url);
const MODEL = new URL("hand_landmarker.task", LOCAL_BASE).href;
const MEDIAPIPE_SOURCES = [
  {
    label: "local",
    module: new URL("vision_bundle.mjs", LOCAL_BASE).href,
    wasm: new URL("wasm/", LOCAL_BASE).href,
  },
  {
    label: "unpkg",
    module: "https://unpkg.com/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs",
    wasm: "https://unpkg.com/@mediapipe/tasks-vision@0.10.32/wasm",
  },
];

const statusEl = document.getElementById("engine-status");
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
let ready = false;

function setStatus(text, ok = false) {
  statusEl.textContent = text;
  statusEl.style.borderColor = ok ? "#4ade80" : "";
}

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} tardó más de ${ms / 1000}s`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

async function loadMediapipe() {
  let lastError = null;
  for (const source of MEDIAPIPE_SOURCES) {
    try {
      setStatus(`Cargando motor (${source.label})…`);
      const mod = await withTimeout(import(source.module), 15000, source.label);
      const HandLandmarker = mod.HandLandmarker;
      const FilesetResolver = mod.FilesetResolver;
      if (!HandLandmarker || !FilesetResolver) {
        throw new Error("El bundle no exporta HandLandmarker");
      }
      setStatus(`Cargando WASM (${source.label})…`);
      const vision = await withTimeout(
        FilesetResolver.forVisionTasks(source.wasm),
        20000,
        `WASM ${source.label}`,
      );
      return { HandLandmarker, vision };
    } catch (err) {
      lastError = err;
      console.warn("MediaPipe fallback", source.module, err);
    }
  }
  throw lastError || new Error("No se pudo cargar MediaPipe");
}

async function createLandmarker(HandLandmarker, vision, delegate) {
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
  setStatus("Cargando MediaPipe…");
  const { HandLandmarker, vision } = await loadMediapipe();
  setStatus("Preparando el modelo de manos…");
  landmarker = await withTimeout(
    createLandmarker(HandLandmarker, vision, "CPU"),
    25000,
    "Modelo de manos",
  );
  try {
    const three = await import("./hand3d.js");
    studio = new three.HandStudio3D();
  } catch (err) {
    console.warn("Vista 3D no disponible", err);
  }
  ready = true;
  setStatus("Modelo listo · pegá una foto", true);
  const pending = window.__pendingHandFile;
  if (pending) {
    window.__pendingHandFile = null;
    await processFile(pending);
  }
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
  studio?.setHand(hand.worldLandmarks, handednessOf(hand));
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
  if (!file || !String(file.type || "").startsWith("image/")) return;
  lastFile = file;
  if (!ready || !landmarker) {
    window.__pendingHandFile = file;
    setStatus("Imagen lista · esperando modelo…");
    return;
  }
  setStatus("Detectando mano…");
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    const detection = landmarker.detect(image);
    showHands(image, detection);
    setStatus(
      `${lastHands.length} mano${lastHands.length === 1 ? "" : "s"} detectada${lastHands.length === 1 ? "" : "s"}`,
      true,
    );
  } catch (err) {
    console.error(err);
    setStatus("No se pudo leer la imagen");
  } finally {
    URL.revokeObjectURL(url);
  }
}

window.addEventListener("hand-file", (event) => {
  processFile(event.detail);
});

if (window.__pendingHandFile) {
  processFile(window.__pendingHandFile);
}

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
  setStatus(err?.message || "No se pudo cargar MediaPipe");
});
