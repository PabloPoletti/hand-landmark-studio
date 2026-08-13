import { CONNECTIONS, NAMES, colorFor, boneColor } from "./schema.js?v=34";
import { predictWilor, wilorToHands } from "./wilor.js?v=34";

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
const skeletonBtn = document.getElementById("skeleton-btn");
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
let wilorJob = 0;

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
    const three = await import("./hand3d.js?v=34");
    studio = new three.HandStudio3D();
  } catch (err) {
    console.warn("Vista 3D no disponible", err);
    setStatus("3D no cargó · recargá con Ctrl+F5");
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

function handCropBox(landmarks, pad = 0.32) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  landmarks.forEach((lm) => {
    minX = Math.min(minX, lm.x);
    minY = Math.min(minY, lm.y);
    maxX = Math.max(maxX, lm.x);
    maxY = Math.max(maxY, lm.y);
  });
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const side = Math.max(maxX - minX, maxY - minY, 0.12) * (1 + pad * 2);
  return {
    x: Math.max(0, cx - side / 2),
    y: Math.max(0, cy - side / 2),
    w: Math.min(1 - Math.max(0, cx - side / 2), side),
    h: Math.min(1 - Math.max(0, cy - side / 2), side),
  };
}

function cropHandFile(image, landmarks) {
  const { width, height } = imageSize(image);
  const box = handCropBox(landmarks);
  const sx = Math.floor(box.x * width);
  const sy = Math.floor(box.y * height);
  const sw = Math.max(48, Math.ceil(box.w * width));
  const sh = Math.max(48, Math.ceil(box.h * height));
  const size = Math.max(sw, sh, 256);
  const ox = (size - sw) / 2;
  const oy = (size - sh) / 2;
  const crop = document.createElement("canvas");
  crop.width = size;
  crop.height = size;
  const g = crop.getContext("2d");
  g.fillStyle = "#777";
  g.fillRect(0, 0, size, size);
  g.drawImage(image, sx, sy, sw, sh, ox, oy, sw, sh);
  const meta = { sx, sy, sw, sh, size, ox, oy, width, height };
  return new Promise((resolve) => {
    crop.toBlob(
      (blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        resolve({ file: new File([blob], "hand-crop.png", { type: "image/png" }), meta });
      },
      "image/png",
      0.92,
    );
  });
}

function remapCropToFull(lm, meta) {
  const px = lm.x * meta.size;
  const py = lm.y * meta.size;
  return {
    ...lm,
    x: (meta.sx + (px - meta.ox)) / meta.width,
    y: (meta.sy + (py - meta.oy)) / meta.height,
    mapped: true,
  };
}

function toDetectCanvas(image) {
  const { width, height } = imageSize(image);
  const scale = Math.min(1, 1600 / Math.max(width, height, 1));
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

const VISIBLE_KNUCKLES = [0, 5, 9, 13, 17];
const TUCK_IDS = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20];

function pointInPoly(point, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const hit = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

function addOcclusion(landmarks, world) {
  const src = world || landmarks;
  const palm = VISIBLE_KNUCKLES;
  const palmZ = palm.reduce((sum, id) => sum + (src[id]?.z ?? 0), 0) / palm.length;
  const hull = [0, 1, 5, 9, 13, 17].map((id) => landmarks[id]).filter(Boolean);
  return landmarks.map((lm, i) => {
    if (VISIBLE_KNUCKLES.includes(i)) return { ...lm, occluded: false };
    let occluded = lm.occluded === true;
    if (typeof lm.visibility === "number") occluded = occluded || lm.visibility < 0.45;
    if (typeof lm.presence === "number") occluded = occluded || lm.presence < 0.5;
    if (TUCK_IDS.includes(i) && hull.length >= 4 && pointInPoly(lm, hull)) occluded = true;
    if ((src[i]?.z ?? 0) > palmZ + 0.03) occluded = true;
    return { ...lm, occluded };
  });
}

function markPath(x, y, r, isRight) {
  if (isRight) {
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      const px = x + r * Math.cos(angle);
      const py = y + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    return;
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
}

function drawLandmarkShape(x, y, r, color, isRight, hidden) {
  ctx.save();
  ctx.globalAlpha = hidden ? 0.5 : 1;
  ctx.shadowColor = color;
  ctx.shadowBlur = r * 0.9;
  markPath(x, y, r + 2.1, isRight);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.shadowBlur = 0;
  markPath(x, y, r, isRight);
  if (hidden) {
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2.4, r * 0.28);
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.globalAlpha = hidden ? 0.22 : 0.55;
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(x - r * 0.28, y - r * 0.3, r * 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawBone(pa, pb, color, width, hidden) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = hidden ? 0.45 : 1;
  ctx.setLineDash(hidden ? [7, 5] : []);
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = "rgba(255,255,255,0.42)";
  ctx.lineWidth = width + 1.6;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

function drawOverlay(image, landmarks, handedness) {
  const fit = drawPhoto(image);
  if (!fit || !landmarks?.length) return;
  const isRight = handedness === "Right";
  const px = (lm) => ({ x: fit.x + lm.x * fit.w, y: fit.y + lm.y * fit.h });
  const stroke = Math.max(1.7, canvas.width / 280);
  CONNECTIONS.forEach(([a, b]) => {
    drawBone(
      px(landmarks[a]),
      px(landmarks[b]),
      boneColor(b === 0 ? a : b),
      stroke,
      isOccluded(landmarks[a]) || isOccluded(landmarks[b]),
    );
  });
  const r = Math.max(4.6, canvas.width / 150);
  landmarks.forEach((lm, i) => {
    const p = px(lm);
    drawLandmarkShape(p.x, p.y, r, colorFor(i), isRight, isOccluded(lm));
  });
}

function handednessOf(hand) {
  return hand.handedness?.[0]?.categoryName || hand.handednesses?.[0]?.categoryName || "Unknown";
}

function photoLandmarks(hand) {
  const mp = sources.mediapipe?.[activeIndex] || sources.mediapipe?.[0];
  if (!mp?.landmarks?.length) return hand.landmarks;
  return mp.landmarks.map((lm, i) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z ?? 0,
    visibility: lm.visibility,
    presence: lm.presence,
    occluded: hand.landmarks[i]?.occluded,
  }));
}

function retractTips(landmarks) {
  const tips = [4, 8, 12, 16, 20];
  const prev = [3, 7, 11, 15, 19];
  return landmarks.map((lm, i) => {
    const idx = tips.indexOf(i);
    if (idx < 0) return lm;
    const base = landmarks[prev[idx]];
    if (!base) return lm;
    return {
      ...lm,
      x: base.x + (lm.x - base.x) * 0.9,
      y: base.y + (lm.y - base.y) * 0.9,
    };
  });
}

function hybridOccluded(landmarks, hand) {
  return landmarks.map((lm, i) => {
    const wilor = hand.landmarks[i];
    if (
      lm.occluded &&
      wilor &&
      Number.isFinite(wilor.x) &&
      Number.isFinite(wilor.y)
    ) {
      return { ...lm, x: wilor.x, y: wilor.y, z: wilor.z ?? lm.z };
    }
    return lm;
  });
}

function renderActive() {
  const hand = lastHands[activeIndex];
  if (!hand || !lastImage) return;
  const landmarks = retractTips(
    hybridOccluded(addOcclusion(photoLandmarks(hand), hand.worldLandmarks), hand),
  );
  drawOverlay(lastImage, landmarks, handednessOf(hand));
  const ownWorld = hand.worldLandmarks?.length >= 21;
  const world =
    ownWorld
      ? hand.worldLandmarks
      : sources.mediapipe?.[activeIndex]?.worldLandmarks ||
        sources.mediapipe?.[0]?.worldLandmarks ||
        landmarks;
  const mesh = ownWorld ? hand.mesh : null;
  try {
    studio?.setHand(world, handednessOf(hand), mesh, landmarks);
  } catch (err) {
    console.warn(err);
    studio?.setHand(world, handednessOf(hand), null, landmarks);
  }
  const label = handednessOf(hand) === "Left" ? "izquierda" : "derecha";
  const hidden = landmarks.filter(isOccluded).length;
  note.textContent = `Mano ${activeIndex + 1} · ${label} · 21 landmarks${
    hidden ? ` · ${hidden} ocluidos` : ""
  }`;
  copyBtn.hidden = false;
  wilorBtn.hidden = false;
  if (skeletonBtn) skeletonBtn.hidden = false;
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
    ["mediapipe", "MediaPipe · preview"],
    ["wilor", "WiLoR · mano 3D"],
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

function showHands(image, detection, source = "mediapipe") {
  lastImage = image;
  const { width, height } = imageSize(image);
  const hands = detectionToHands(detection);
  sources[source] = hands;
  activeSource = source;
  lastHands = hands;

  results.hidden = false;
  drawPhoto(image);

  if (!lastHands.length) {
    toolbar.hidden = true;
    copyBtn.hidden = true;
    if (skeletonBtn) skeletonBtn.hidden = true;
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
  wilorJob += 1;
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
        ? `${lastHands.length} mano${lastHands.length === 1 ? "" : "s"} · refinando con WiLoR…`
        : `0 manos · ${width}×${height}`,
      lastHands.length > 0,
    );
    refineWithWilor();
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

async function refineWithWilor() {
  if (!lastFile || !lastImage) return;
  const job = ++wilorJob;
  wilorBtn.disabled = true;
  try {
    const mpHand = sources.mediapipe?.[0];
    const hint = mpHand ? handednessOf(mpHand) === "Right" : null;
    const cropped =
      mpHand?.landmarks?.length === 21
        ? await cropHandFile(lastImage, mpHand.landmarks)
        : null;
    const payload = cropped?.file || lastFile;
    const result = await predictWilor(payload, setStatus, hint);
    if (job !== wilorJob) return;
    let hands = wilorToHands(result);
    if (cropped?.meta) {
      hands = hands.map((hand) => ({
        ...hand,
        landmarks: hand.landmarks.map((lm) => remapCropToFull(lm, cropped.meta)),
      }));
    }
    if (!hands.length) {
      setStatus("WiLoR no encontró manos · se deja MediaPipe");
      return;
    }
    showHands(lastImage, {
      landmarks: hands.map((h) => h.landmarks),
      worldLandmarks: hands.map((h) => h.worldLandmarks),
      handedness: hands.map((h) => h.handedness),
      mesh: hands.map((h) => h.mesh),
    }, "wilor");
    const hasMesh = hands.some((h) => h.mesh?.vertices?.length);
    setStatus(
      hasMesh
        ? `WiLoR · mano 3D MANO · ${hands.length} mano${hands.length === 1 ? "" : "s"}`
        : `WiLoR · ${hands.length} mano${hands.length === 1 ? "" : "s"}`,
      true,
    );
  } catch (err) {
    if (job !== wilorJob) return;
    console.error(err);
    setStatus(err.message || "WiLoR no está disponible · se deja MediaPipe");
  } finally {
    if (job === wilorJob) wilorBtn.disabled = false;
  }
}

wilorBtn.addEventListener("click", () => {
  refineWithWilor();
});

skeletonBtn?.addEventListener("click", () => {
  const on = !studio?.skeletonOnly;
  studio?.setSkeletonMode(on);
  skeletonBtn.classList.toggle("active", on);
  skeletonBtn.textContent = on ? "Ver mano" : "Ver esqueleto";
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
