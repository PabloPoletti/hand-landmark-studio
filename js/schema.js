export const NAMES = [
  "Wrist",
  "Thumb CMC",
  "Thumb MCP",
  "Thumb IP",
  "Thumb tip",
  "Index MCP",
  "Index PIP",
  "Index DIP",
  "Index tip",
  "Middle MCP",
  "Middle PIP",
  "Middle DIP",
  "Middle tip",
  "Ring MCP",
  "Ring PIP",
  "Ring DIP",
  "Ring tip",
  "Pinky MCP",
  "Pinky PIP",
  "Pinky DIP",
  "Pinky tip",
];

export const FINGER = {
  wrist: { ids: [0], color: "#9aa0a6" },
  thumb: { ids: [1, 2, 3, 4], color: "#f0abfc" },
  index: { ids: [5, 6, 7, 8], color: "#fb923c" },
  middle: { ids: [9, 10, 11, 12], color: "#4ade80" },
  ring: { ids: [13, 14, 15, 16], color: "#38bdf8" },
  pinky: { ids: [17, 18, 19, 20], color: "#a78bfa" },
};

export const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

export const FINGER_CHAINS = [
  { key: "thumb", ids: [1, 2, 3, 4], radii: [0.11, 0.1, 0.085, 0.068] },
  { key: "index", ids: [5, 6, 7, 8], radii: [0.09, 0.075, 0.062, 0.05] },
  { key: "middle", ids: [9, 10, 11, 12], radii: [0.092, 0.078, 0.064, 0.052] },
  { key: "ring", ids: [13, 14, 15, 16], radii: [0.086, 0.072, 0.06, 0.048] },
  { key: "pinky", ids: [17, 18, 19, 20], radii: [0.078, 0.065, 0.054, 0.044] },
];

export function colorFor(id) {
  if (id === 0) return FINGER.wrist.color;
  if (id <= 4) return FINGER.thumb.color;
  if (id <= 8) return FINGER.index.color;
  if (id <= 12) return FINGER.middle.color;
  if (id <= 16) return FINGER.ring.color;
  return FINGER.pinky.color;
}
