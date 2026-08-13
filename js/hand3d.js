import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CONNECTIONS, FINGER_CHAINS, colorFor, boneColor } from "./schema.js?v=33";

const SKIN_SCALE = 1.72;
const SURFACE_LIFT = 0.034;
const PALM_LIFT = 0.055;
const PALM_EDGES = new Set(["0-1", "0-5", "0-9", "0-13", "0-17", "5-9", "9-13", "13-17"]);

const Y_UP = new THREE.Vector3(0, 1, 0);

function toVec(p) {
  if (Array.isArray(p)) return new THREE.Vector3(Number(p[0]), -Number(p[1]), -Number(p[2]));
  return new THREE.Vector3(Number(p.x ?? p.X), -Number(p.y ?? p.Y), -Number(p.z ?? p.Z));
}

function isFiniteVec(v) {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

function usableWorld(world) {
  return Array.isArray(world) && world.length >= 21 && world.every((p) => isFiniteVec(toVec(p)));
}

function usableMesh(mesh) {
  if (!mesh?.vertices?.length || !mesh?.faces?.length) return false;
  const n = mesh.vertices.length;
  return asTriangles(mesh.faces).every(
    ([a, b, c]) =>
      Number.isInteger(a) &&
      Number.isInteger(b) &&
      Number.isInteger(c) &&
      a >= 0 &&
      b >= 0 &&
      c >= 0 &&
      a < n &&
      b < n &&
      c < n,
  );
}

function alignmentFromLandmarks(world, handedness) {
  const pts = world.map(toVec);
  const wrist = pts[0].clone();
  pts.forEach((p) => p.sub(wrist));

  const mid = pts[9].clone();
  const index = pts[5].clone();
  const pinky = pts[17].clone();
  const y = mid.clone().normalize();
  let z = new THREE.Vector3().crossVectors(index, pinky);
  if (z.lengthSq() < 1e-8) z = new THREE.Vector3(0, 0, 1);
  z.normalize();
  if (handedness === "Left") z.negate();
  const x = new THREE.Vector3().crossVectors(y, z).normalize();
  z.crossVectors(x, y).normalize();
  const basis = new THREE.Matrix4().makeBasis(x, y, z).invert();
  pts.forEach((p) => p.applyMatrix4(basis));

  let max = 0;
  pts.forEach((p) => {
    max = Math.max(max, p.length());
  });
  const scale = max > 0 ? 1.05 / max : 1;
  pts.forEach((p) => p.multiplyScalar(scale));
  const center = new THREE.Vector3();
  pts.forEach((p) => center.add(p));
  center.multiplyScalar(1 / pts.length);
  pts.forEach((p) => p.sub(center));

  return {
    pts,
    apply(raw) {
      const v = toVec(raw).sub(wrist);
      v.applyMatrix4(basis);
      v.multiplyScalar(scale);
      v.sub(center);
      return v;
    },
  };
}

function bone(a, b, r0, r1, material) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-5) return null;
  const top = r1 ?? r0;
  const geo = new THREE.CylinderGeometry(top, r0, len, 28, 1, false);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(Y_UP, dir.clone().normalize());
  return mesh;
}

function overlayMaterial(color, opacity = 1) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
}

function palmNormal(pts) {
  const n = new THREE.Vector3().crossVectors(
    pts[5].clone().sub(pts[0]),
    pts[17].clone().sub(pts[0]),
  );
  if (n.lengthSq() < 1e-8) return new THREE.Vector3(0, 0, 1);
  n.normalize();
  if (n.z < 0) n.negate();
  return n;
}

function radiusForId(id) {
  if (id === 0) return 0.08;
  for (const finger of FINGER_CHAINS) {
    const idx = finger.ids.indexOf(id);
    if (idx >= 0) return finger.radii[idx] * SKIN_SCALE;
  }
  return 0.045;
}

function isPalmEdge(a, b) {
  return PALM_EDGES.has(a < b ? `${a}-${b}` : `${b}-${a}`);
}

function surfacePoints(pts) {
  const normal = palmNormal(pts);
  return pts.map((p, id) => {
    const knuckle = id === 0 || id === 5 || id === 9 || id === 13 || id === 17;
    const lift = (knuckle ? PALM_LIFT : SURFACE_LIFT) + radiusForId(id) * 0.35;
    return p.clone().addScaledVector(normal, lift);
  });
}

function addBoneSegment(overlay, a, b, color, radius, hidden) {
  const mesh = bone(a, b, radius, radius, overlayMaterial(color, hidden ? 0.55 : 0.95));
  if (mesh) {
    mesh.renderOrder = 21;
    overlay.add(mesh);
  }
}

function addSkeleton(group, pts, landmarks = [], handedness = "Unknown") {
  const isRight = handedness === "Right";
  const normal = palmNormal(pts);
  const outer = surfacePoints(pts);
  const overlay = new THREE.Group();
  overlay.renderOrder = 20;

  CONNECTIONS.forEach(([a, b]) => {
    const hidden = landmarks[a]?.occluded || landmarks[b]?.occluded;
    const color = boneColor(b === 0 ? a : b);
    const radius = hidden ? 0.0032 : 0.0046;
    if (isPalmEdge(a, b)) {
      const mid = outer[a].clone().add(outer[b]).multiplyScalar(0.5).addScaledVector(normal, 0.042);
      addBoneSegment(overlay, outer[a], mid, color, radius, hidden);
      addBoneSegment(overlay, mid, outer[b], color, radius, hidden);
    } else {
      addBoneSegment(overlay, outer[a], outer[b], color, radius, hidden);
    }
  });

  outer.forEach((p, i) => {
    const hidden = Boolean(landmarks[i]?.occluded);
    const r = hidden ? 0.011 : 0.014;
    const color = colorFor(i);
    const halo = new THREE.Mesh(
      isRight
        ? new THREE.CylinderGeometry(r + 0.0035, r + 0.0035, r * 0.22, 5)
        : new THREE.SphereGeometry(r + 0.0035, 22, 18),
      overlayMaterial(0xffffff, hidden ? 0.45 : 1),
    );
    halo.position.copy(p);
    halo.renderOrder = 22;
    overlay.add(halo);
    const mark = new THREE.Mesh(
      isRight
        ? new THREE.CylinderGeometry(r, r, r * 0.46, 5)
        : new THREE.SphereGeometry(r, 22, 18),
      overlayMaterial(color, hidden ? 0.5 : 1),
    );
    mark.position.copy(p);
    mark.renderOrder = 23;
    overlay.add(mark);
  });

  group.add(overlay);
}

function skinMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: 0xc48a68,
    roughness: 0.52,
    metalness: 0.02,
    clearcoat: 0.12,
    clearcoatRoughness: 0.55,
    sheen: 0.18,
    sheenRoughness: 0.55,
    sheenColor: new THREE.Color(0xc99674),
    side: THREE.FrontSide,
    depthWrite: true,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

function addPalm(group, pts, material) {
  const ids = [0, 5, 9, 13, 17];
  const center = new THREE.Vector3();
  ids.forEach((id) => center.add(pts[id]));
  center.multiplyScalar(1 / ids.length);
  center.lerp(new THREE.Vector3().addVectors(pts[0], pts[9]).multiplyScalar(0.5), 0.25);

  const across = pts[17].clone().sub(pts[5]);
  const along = pts[9].clone().sub(pts[0]);
  let normal = new THREE.Vector3().crossVectors(pts[5].clone().sub(pts[0]), pts[17].clone().sub(pts[0]));
  if (normal.lengthSq() < 1e-8) normal = new THREE.Vector3(0, 0, 1);
  normal.normalize();
  const x = across.lengthSq() < 1e-8 ? new THREE.Vector3(1, 0, 0) : across.clone().normalize();
  const y = along.lengthSq() < 1e-8 ? new THREE.Vector3(0, 1, 0) : along.clone().normalize();
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  y.crossVectors(z, x).normalize();

  const palm = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), material);
  palm.position.copy(center);
  palm.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  palm.scale.set(across.length() * 0.58, along.length() * 0.46, Math.max(0.075, across.length() * 0.2));
  group.add(palm);

  const thenar = new THREE.Mesh(new THREE.SphereGeometry(across.length() * 0.26, 28, 22), material);
  thenar.position.copy(pts[0].clone().add(pts[1]).add(pts[5]).multiplyScalar(1 / 3));
  group.add(thenar);
}

function addFinger(group, chain, radii, material) {
  for (let i = 0; i < chain.length - 1; i++) {
    const r0 = radii[i];
    const r1 = radii[Math.min(i + 1, radii.length - 1)];
    const mesh = bone(chain[i], chain[i + 1], r0, r1, material);
    if (mesh) group.add(mesh);
    const knuckle = new THREE.Mesh(new THREE.SphereGeometry(r0 * 1.05, 28, 22), material);
    knuckle.position.copy(chain[i]);
    group.add(knuckle);
  }
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(radii[radii.length - 1] * 1.08, 28, 22),
    material,
  );
  tip.position.copy(chain[chain.length - 1]);
  group.add(tip);
}

function asTriangles(faces) {
  if (!faces?.length) return [];
  if (Array.isArray(faces[0])) return faces;
  const tris = [];
  for (let i = 0; i + 2 < faces.length; i += 3) {
    tris.push([faces[i], faces[i + 1], faces[i + 2]]);
  }
  return tris;
}

function smoothVertices(verts, faces, iterations = 2) {
  const tris = asTriangles(faces);
  const adj = verts.map(() => new Set());
  tris.forEach(([a, b, c]) => {
    adj[a]?.add(b);
    adj[a]?.add(c);
    adj[b]?.add(a);
    adj[b]?.add(c);
    adj[c]?.add(a);
    adj[c]?.add(b);
  });
  let cur = verts.map((v) => v.clone());
  for (let i = 0; i < iterations; i++) {
    cur = cur.map((v, idx) => {
      const nbrs = adj[idx];
      if (!nbrs?.size) return v.clone();
      const avg = new THREE.Vector3();
      nbrs.forEach((j) => avg.add(cur[j]));
      avg.multiplyScalar(1 / nbrs.size);
      return v.clone().lerp(avg, 0.32);
    });
  }
  return cur;
}

function buildManoHand(pts, meshData, apply, landmarks, handedness) {
  const group = new THREE.Group();
  const verts = smoothVertices(meshData.vertices.map((v) => apply(v)), meshData.faces);
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(verts.length * 3);
  verts.forEach((v, i) => {
    pos[i * 3] = v.x;
    pos[i * 3 + 1] = v.y;
    pos[i * 3 + 2] = v.z;
  });
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(asTriangles(meshData.faces).flat());
  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshPhysicalMaterial({
      color: 0xc48a68,
      roughness: 0.5,
      metalness: 0.02,
      clearcoat: 0.12,
      clearcoatRoughness: 0.55,
      sheen: 0.18,
      sheenRoughness: 0.55,
      sheenColor: new THREE.Color(0xc99674),
      transparent: true,
      opacity: 0.98,
      side: THREE.FrontSide,
      depthWrite: true,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    }),
  );
  group.add(mesh);
  addSkeleton(group, pts, landmarks, handedness);
  return group;
}

function buildAnatomicalHand(pts, landmarks, handedness) {
  const group = new THREE.Group();
  const skin = skinMaterial();
  addPalm(group, pts, skin);
  FINGER_CHAINS.forEach((finger) => {
    const chain = finger.ids.map((id) => pts[id]);
    if (finger.key === "thumb") chain.unshift(pts[0]);
    const radii = finger.radii.map((r) => r * SKIN_SCALE);
    if (finger.key === "thumb") radii.unshift(radii[0] * 1.15);
    addFinger(group, chain, radii, skin);
  });
  addSkeleton(group, pts, landmarks, handedness);
  return group;
}

function disposeHand(hand) {
  if (!hand) return;
  hand.traverse((obj) => {
    if (obj.element) obj.element.remove();
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => m.dispose?.());
    }
  });
}

function makeView(container, cameraPos) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x161a22);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
  camera.position.copy(cameraPos);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  try {
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.32;
    pmrem.dispose();
  } catch {
    /* luces direccionales alcanzan si el env map no carga */
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.minDistance = 1.4;
  controls.maxDistance = 5;

  scene.add(new THREE.AmbientLight(0xffffff, 0.42));
  scene.add(new THREE.HemisphereLight(0xe8ddd2, 0x5a534c, 0.78));
  const key = new THREE.DirectionalLight(0xfff1e4, 1.05);
  key.position.set(1.6, 2.4, 2.0);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xc5d0e0, 0.28);
  fill.position.set(-2.1, 0.5, 0.9);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xe8d8c4, 0.32);
  rim.position.set(-0.3, 1.1, -2.1);
  scene.add(rim);

  return { scene, camera, renderer, controls, hand: null };
}

export class HandStudio3D {
  constructor() {
    this.views = {
      front: makeView(document.getElementById("view-front"), new THREE.Vector3(0, 0.15, 2.55)),
      threequarter: makeView(
        document.getElementById("view-threequarter"),
        new THREE.Vector3(1.7, 0.35, 1.85),
      ),
      side: makeView(document.getElementById("view-side"), new THREE.Vector3(2.55, 0.12, 0.15)),
    };
    this.resize();
    window.addEventListener("resize", () => this.resize());
    const tick = () => {
      Object.values(this.views).forEach((v) => {
        v.controls.update();
        v.renderer.render(v.scene, v.camera);
      });
      requestAnimationFrame(tick);
    };
    tick();
  }

  resize() {
    Object.values(this.views).forEach((v) => {
      const el = v.renderer.domElement.parentElement;
      const w = el.clientWidth || 320;
      const h = el.clientHeight || 320;
      v.camera.aspect = w / h;
      v.camera.updateProjectionMatrix();
      v.renderer.setSize(w, h, false);
    });
  }

  setHand(worldLandmarks, handedness, mesh, landmarks = []) {
    if (!usableWorld(worldLandmarks)) return;
    const aligned = alignmentFromLandmarks(worldLandmarks, handedness);
    Object.values(this.views).forEach((v) => {
      let next = null;
      try {
        next =
          usableMesh(mesh)
            ? buildManoHand(aligned.pts, mesh, aligned.apply, landmarks, handedness)
            : buildAnatomicalHand(aligned.pts, landmarks, handedness);
      } catch (err) {
        console.warn("No se pudo armar la malla MANO, se usa la mano anatómica", err);
        try {
          next = buildAnatomicalHand(aligned.pts, landmarks, handedness);
        } catch (fallbackErr) {
          console.warn("Tampoco se pudo armar la mano 3D", fallbackErr);
          return;
        }
      }
      if (v.hand) {
        v.scene.remove(v.hand);
        disposeHand(v.hand);
      }
      v.hand = next;
      v.scene.add(v.hand);
    });
    this.resize();
  }

  clear() {
    Object.values(this.views).forEach((v) => {
      if (v.hand) {
        v.scene.remove(v.hand);
        disposeHand(v.hand);
        v.hand = null;
      }
    });
  }
}
