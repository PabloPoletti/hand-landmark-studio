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
let sources = { mediapipe: null, wilor: null };
let activeSource = "mediapipe";

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
    minHandDetectionConfidence: 0.05,
    minHandPresenceConfidence: 0.05,
    minTrackingConfidence: 0.05,
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

function imageSize(image) {
  return {
    width: image.width || image.naturalWidth || 0,
    height: image.height || image.naturalHeight || 0,
  };
}

function toDetectCanvas(image) {
  const { width, height } = imageSize(image);
  const scale = Math.min(1, 1280 / Math.max(width, height, 1));
  const detectCanvas = document.createElement("canvas");
  detectCanvas.width = Math.max(1, Math.round(width * scale));
  detectCanvas.height = Math.max(1, Math.round(height * scale));
  detectCanvas.getContext("2d", { willReadFrequently: true }).drawImage(
    image,
    0,
    0,
    detectCanvas.width,
    detectCanvas.height,
  );
  return detectCanvas;
}

function fitRect(srcW, srcH, dstW, dstH) {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { x: (dstW - w) / 2, y: (dstH - h) / 2, w, h };
}

function drawPhoto(image) {
  const { width, height } = imageSize(image);
  if (!width || !height) return null;
  const size = 720;
  canvas.width = size;
  canvas.height = size;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, size, size);
  const fit = fitRect(width, height, size, size);
  ctx.drawImage(image, fit.x, fit.y, fit.w, fit.h);
  return fit;
}

function isOccluded(lm) {
  if (!lm) return false;
  if (typeof lm.occluded === "boolean") return lm.occluded;
  if (typeof lm.visibility === "number") return lm.visibility < 0.5;
  return false;
}

function addOcclusion(landmarks, world) {
  const src = world || landmarks;
  const palm = [0, 5, 9, 13, 17];
  if (!src?.[0] || landmarks.some((lm) => typeof lm.occluded === "boolean")) {
    return landmarks;
  }
  const palmZ = palm.reduce((sum, id) => sum + (src[id]?.z ?? 0), 0) / palm.length;
  return landmarks.map((lm, i) => ({
    ...lm,
    occluded: !palm.includes(i) && (src[i]?.z ?? 0) > palmZ + 0.012,
  }));
}

function pentagonPath(x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const px = x + r * Math.cos(angle);
    const py = y + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function drawLandmarkShape(x, y, r, color, isRight, hidden) {
  ctx.save();
  ctx.globalAlpha = hidden ? 0.55 : 1;
  if (isRight) pentagonPath(x, y, r + 1.6);
  else {
    ctx.beginPath();
    ctx.arc(x, y, r + 1.6, 0, Math.PI * 2);
  }
  ctx.fillStyle = "#111";
  ctx.fill();
  if (isRight) pentagonPath(x, y, r);
  else {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
  }
  if (hidden) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash([3, 2]);
    ctx.stroke();
    ctx.setLineDash([]);
  } else if (isRight) {
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

function drawOverlay(image, landmarks, handedness) {
  const fit = drawPhoto(image);
  if (!fit || !landmarks?.length) return;
  const isRight = handedness === "Right";

  const px = (lm) => ({ x: fit.x + lm.x * fit.w, y: fit.y + lm.y * fit.h });
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const stroke = Math.max(4, canvas.width / 140);
  CONNECTIONS.forEach(([a, b]) => {
    const pa = px(landmarks[a]);
    const pb = px(landmarks[b]);
    const hidden = isOccluded(landmarks[a]) || isOccluded(landmarks[b]);
    ctx.globalAlpha = hidden ? 0.45 : 1;
    ctx.setLineDash(hidden ? [8, 5] : []);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = stroke + 3;
    ctx.stroke();
    ctx.strokeStyle = colorFor(b === 0 ? a : b);
    ctx.lineWidth = stroke;
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  const r = Math.max(5.5, canvas.width / 120);
  const font = Math.max(12, canvas.width / 56);
  landmarks.forEach((lm, i) => {
    const p = px(lm);
    const hidden = isOccluded(lm);
    drawLandmarkShape(p.x, p.y, r, colorFor(i), isRight, hidden);
    ctx.font = `800 ${font}px Segoe UI, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    const label = hidden ? `${i}*` : String(i);
    const tx = p.x + r + 3;
    const ty = p.y - r;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    ctx.strokeText(label, tx, ty);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, tx, ty);
  });
}

function handednessOf(hand) {
  return hand.handedness?.[0]?.categoryName || hand.handednesses?.[0]?.categoryName || "Unknown";
}

function renderActive() {
  const hand = lastHands[activeIndex];
  if (!hand || !lastImage) return;
  const landmarks = addOcclusion(hand.landmarks, hand.worldLandmarks);
  hand.landmarks = landmarks;
  drawOverlay(lastImage, landmarks, handednessOf(hand));
  studio?.setHand(hand.worldLandmarks, handednessOf(hand), hand.mesh, landmarks);
  const label = handednessOf(hand) === "Left" ? "izquierda" : "derecha";
  const hidden = landmarks.filter(isOccluded).length;
  note.textContent = `Mano ${activeIndex + 1} · ${label} · 21 landmarks${
    hidden ? ` · ${hidden} ocluidos (*)` : ""
  }`;
  copyBtn.hidden = false;
  wilorBtn.hidden = false;
}

function detectionToHands(detection) {
  return (detection.landmarks || []).map((landmarks, i) => ({
    landmarks,
    worldLandmarks: detection.worldLandmarks[i],
    handedness: detection.handedness?.[i] || detection.handednesses?.[i],
    mesh: detection.mesh?.[i] || null,
  }));
}

function renderSourceChips() {
  const existing = document.getElementById("source-switcher");
  if (!existing) return;
  existing.innerHTML = "";
  [
    ["mediapipe", "Opción A · MediaPipe"],
    ["wilor", "Opción B · WiLoR"],
  ].forEach(([key, label]) => {
    if (!sources[key]?.length) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (activeSource === key ? " active" : "");
    btn.textContent = label;
    btn.addEventListener("click", () => {
      activeSource = key;
      lastHands = sources[key];
      activeIndex = 0;
      renderSourceChips();
      renderHandChips();
      renderActive();
    });
    existing.appendChild(btn);
  });
}

function renderHandChips() {
  switcher.innerHTML = "";
  lastHands.forEach((hand, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (i === activeIndex ? " active" : "");
    const side = handednessOf(hand) === "Left" ? "Left" : "Right";
    btn.textContent = `Mano ${i + 1} · ${side}`;
    btn.addEventListener("click", () => {
      activeIndex = i;
      [...switcher.children].forEach((c, k) => c.classList.toggle("active", k === i));
      renderActive();
    });
    switcher.appendChild(btn);
  });
}

function pinOverlayToMediaPipe(hands) {
  const mp = sources.mediapipe?.[0];
  if (!mp?.landmarks?.length || mp.landmarks.length !== 21) return hands;
  return hands.map((hand) => ({
    ...hand,
    landmarks: mp.landmarks.map((lm, i) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z ?? 0,
      occluded: Boolean(hand.landmarks[i]?.occluded),
    })),
  }));
}

function showHands(image, detection, source = "mediapipe") {
  lastImage = image;
  const { width, height } = imageSize(image);
  let hands = detectionToHands(detection);
  if (source === "wilor") hands = pinOverlayToMediaPipe(hands);
  sources[source] = hands;
  activeSource = source;
  lastHands = hands;

  results.hidden = false;
  drawPhoto(image);

  if (!lastHands.length) {
    toolbar.hidden = true;
    copyBtn.hidden = true;
    wilorBtn.hidden = false;
    emptyMsg.hidden = false;
    emptyMsg.textContent = `No se detectó ninguna mano en ${width}×${height}. Probá una foto más cercana, con la palma o el dorso bien visibles.`;
    studio?.clear();
    return;
  }

  emptyMsg.hidden = true;
  results.hidden = false;
  toolbar.hidden = false;
  activeIndex = 0;
  renderSourceChips();
  renderHandChips();
  renderActive();
}

async function loadImage(file) {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      return createImageBitmap(file);
    }
  }
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
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
  try {
    const image = await loadImage(file);
    sources = { mediapipe: null, wilor: null };
    const detection = landmarker.detect(toDetectCanvas(image));
    showHands(image, detection, "mediapipe");
    const { width, height } = imageSize(image);
    setStatus(
      lastHands.length
        ? `${lastHands.length} mano${lastHands.length === 1 ? "" : "s"} · ${width}×${height}`
        : `0 manos · ${width}×${height}`,
      lastHands.length > 0,
    );
  } catch (err) {
    console.error(err);
    setStatus("No se pudo leer la imagen");
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
    const mpHand = sources.mediapipe?.[0];
    const hint = mpHand ? handednessOf(mpHand) === "Right" : null;
    const result = await predictWilor(lastFile, setStatus, hint);
    const hands = wilorToHands(result);
    if (!hands.length) {
      setStatus("WiLoR no encontró manos");
      return;
    }
    showHands(lastImage, {
      landmarks: hands.map((h) => h.landmarks),
      worldLandmarks: hands.map((h) => h.worldLandmarks),
      handedness: hands.map((h) => h.handedness),
      mesh: hands.map((h) => h.mesh),
    }, "wilor");
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
