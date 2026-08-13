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

export async function predictWilor(file, onStatus) {
  onStatus?.("Mandando la foto a WiLoR (Hugging Face)…");
  const dataUrl = await fileToDataUrl(file);
  const start = await fetch(`${WILOR_SPACE}/gradio_api/call/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: [dataUrl] }),
  });

  if (start.status === 404 || start.status === 410) {
    throw new Error(
      "El Space todavía no está publicado. Creá una cuenta gratis en huggingface.co y avisame para subirlo.",
    );
  }
  if (!start.ok) {
    throw new Error(`WiLoR no respondió (${start.status}). El Space puede estar despertando.`);
  }

  const started = await start.json();
  const eventId = started.event_id;
  if (!eventId) throw new Error("El Space no devolvió event_id.");

  onStatus?.("Esperando WiLoR… la primera vez puede tardar 1–2 min");
  const stream = await fetch(`${WILOR_SPACE}/gradio_api/call/predict/${eventId}`);
  if (!stream.ok) throw new Error(`WiLoR falló al devolver el resultado (${stream.status}).`);
  const payload = parseSseJson(await stream.text());
  let result = Array.isArray(payload) ? payload[0] : payload;
  if (typeof result === "string") {
    try {
      result = JSON.parse(result);
    } catch {
      throw new Error(result || "WiLoR devolvió un texto inválido.");
    }
  }
  if (!result || result.error) {
    throw new Error(result?.error || "WiLoR no devolvió landmarks.");
  }
  return result;
}

export function wilorToHands(result) {
  return (result.hands || []).map((hand) => ({
    landmarks: hand.landmarks.map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z ?? 0,
    })),
    worldLandmarks: hand.landmarks.map((lm) => ({
      x: lm.X,
      y: lm.Y,
      z: lm.Z,
    })),
    handedness: [{ categoryName: hand.is_right ? "Right" : "Left" }],
  }));
}
