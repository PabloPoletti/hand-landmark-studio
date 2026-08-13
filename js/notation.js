import { FINGER, colorFor } from "./schema.js";

export function isLeftHand(handedness) {
  return String(handedness || "").toLowerCase().startsWith("left");
}

export function markerSVG(id, color, left, occluded) {
  const shape = left
    ? `<circle cx="10" cy="10" r="7.2"/>`
    : `<polygon points="10,1.4 18.6,8 15.2,18.4 4.8,18.4 1.4,8"/>`;
  const slash = occluded
    ? `<line x1="4" y1="4" x2="16" y2="16" stroke="${color}" stroke-width="2.3" stroke-linecap="round"/>`
    : "";
  return `<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
    <g fill="${occluded ? "none" : color}" stroke="${occluded ? color : "#fff"}" stroke-width="${occluded ? 2.3 : 1.5}">
      ${shape}
    </g>
    ${slash}
  </svg><span>${id}</span>`;
}

export function setMarker(el, id, color, left, occluded) {
  el.className = `lm ${left ? "left" : "right"}${occluded ? " occluded" : ""}`;
  el.style.color = color;
  el.innerHTML = markerSVG(id, color, left, occluded);
}

export function createMarker(id, color, left, occluded) {
  const el = document.createElement("div");
  setMarker(el, id, color, left, occluded);
  return el;
}

export function drawShape(ctx, x, y, r, left) {
  ctx.beginPath();
  if (left) {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    return;
  }
  for (let i = 0; i < 5; i++) {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const px = x + r * Math.cos(a);
    const py = y + r * Math.sin(a);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

export function drawLandmark2D(ctx, x, y, r, id, left, occluded) {
  const color = colorFor(id);
  ctx.save();
  if (occluded) {
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2.2, r * 0.45);
    drawShape(ctx, x, y, r, left);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - r * 0.72, y - r * 0.72);
    ctx.lineTo(x + r * 0.72, y + r * 0.72);
    ctx.stroke();
  } else {
    ctx.fillStyle = "#fff";
    drawShape(ctx, x, y, r + 1.5, left);
    ctx.fill();
    ctx.fillStyle = color;
    drawShape(ctx, x, y, r, left);
    ctx.fill();
  }
  ctx.restore();
}

export const LEGEND = [
  { key: "wrist", label: "Wrist", ids: "0" },
  { key: "thumb", label: "Thumb", ids: "1–4" },
  { key: "index", label: "Index", ids: "5–8" },
  { key: "middle", label: "Middle", ids: "9–12" },
  { key: "ring", label: "Ring", ids: "13–16" },
  { key: "pinky", label: "Pinky", ids: "17–20" },
].map((row) => ({ ...row, color: FINGER[row.key].color }));
