import { WILOR_SPACE } from "./config.js";

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function parseSseJson(text) {
  let payload = null;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const raw = line.slice(6).trim();
    if (!raw || raw === "[DONE]") continue;
    payload = JSON.parse(raw);
  }
  return payload;
}

function normalizeResult(raw) {
  let result = Array.isArray(raw) ? raw[0] : raw;
  if (typeof result === "string") {
    result = JSON.parse(result);
  }
  return result;
}

async function predictDirect(dataUrl, onStatus, isRight) {
  onStatus?.("Mandando la foto a WiLoR (Hugging Face)…");
  const response = await fetch(`${WILOR_SPACE}/wilor`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, is_right: isRight }),
  });
  if (response.status === 404 || response.status === 410) return null;
  if (!response.ok) {
    throw new Error(`WiLoR no respondió (${response.status}). El Space puede estar despertando.`);
  }
  return response.json();
}

async function predictGradio(dataUrl, onStatus) {
  const endpoints = [
    `${WILOR_SPACE}/call/predict`,
    `${WILOR_SPACE}/gradio_api/call/predict`,
  ];
  let started = null;
  let url = "";
  for (const endpoint of endpoints) {
    const start = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [dataUrl] }),
    });
    if (start.ok) {
      started = await start.json();
      url = endpoint;
      break;
    }
  }
  if (!started?.event_id) return null;
  onStatus?.("Esperando WiLoR… la primera vez puede tardar 1–2 min");
  const stream = await fetch(`${url}/${started.event_id}`);
  if (!stream.ok) throw new Error(`WiLoR falló al devolver el resultado (${stream.status}).`);
  return normalizeResult(parseSseJson(await stream.text()));
}

export async function predictWilor(file, onStatus, isRight = null) {
  const dataUrl = await fileToDataUrl(file);
  let result = await predictDirect(dataUrl, onStatus, isRight);
  if (!result) result = await predictGradio(dataUrl, onStatus);
  if (!result) {
    throw new Error("El Space de WiLoR no está publicado o se está reiniciando.");
  }
  if (result.error) throw new Error(result.error);
  if (!result.hands?.length) throw new Error("WiLoR no encontró manos en la foto.");
  return result;
}

export function wilorToHands(result) {
  return (result.hands || []).map((hand) => ({
    landmarks: hand.landmarks.map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z ?? 0,
      occluded: Boolean(lm.occluded),
    })),
    worldLandmarks: hand.landmarks.map((lm) => ({
      x: Number(lm.X ?? lm.x ?? 0),
      y: Number(lm.Y ?? lm.y ?? 0),
      z: Number(lm.Z ?? lm.z ?? 0),
    })),
    handedness: [{ categoryName: hand.is_right ? "Right" : "Left" }],
    mesh:
      hand.vertices && (result.faces || hand.faces)
        ? { vertices: hand.vertices, faces: result.faces || hand.faces }
        : null,
  }));
}
