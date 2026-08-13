import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CONNECTIONS, FINGER_CHAINS, colorFor, boneColor } from "./schema.js?v=35";

const SKIN_SCALE = 1.42;
const MCP_IDS = new Set([1, 5, 9, 13, 17]);
const TIP_IDS = new Set([4, 8, 12, 16, 20]);

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
  const scale = max > 0 ? 0.88 / max : 1;
  pts.forEach((p) => p.multiplyScalar(scale));
  const center = new THREE.Vector3();
  pts.forEach((p) => center.add(p));
  center.multiplyScalar(1 / pts.length);
  center.lerp(pts[0], 0.22);
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

function markRadius(id, hidden) {
  if (id === 0) return hidden ? 0.02 : 0.026;
  if (MCP_IDS.has(id)) return hidden ? 0.016 : 0.02;
  if (TIP_IDS.has(id)) return hidden ? 0.014 : 0.017;
  return hidden ? 0.013 : 0.016;
}

function tagOverlay(obj) {
  obj.userData.overlay = true;
  obj.traverse((child) => {
    child.userData.overlay = true;
  });
}

function applyDisplayMode(hand, mode) {
  if (!hand) return;
  const xray = mode === "xray";
  const hideSkin = mode === "skeleton";
  hand.traverse((obj) => {
    if (!obj.isMesh) return;
    if (obj.userData.overlay) {
      obj.visible = true;
      return;
    }
    if (hideSkin) {
      obj.visible = false;
      return;
    }
    obj.visible = true;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach((mat) => {
      if (!mat?.userData?.skin) return;
      mat.transparent = true;
      mat.opacity = xray ? 0.36 : mat.userData.solidOpacity ?? 0.92;
      mat.depthWrite = !xray;
    });
  });
}

function addSkeleton(group, pts, landmarks = [], handedness = "Unknown") {
  const isRight = handedness === "Right";
  const overlay = new THREE.Group();
  overlay.renderOrder = 20;

  CONNECTIONS.forEach(([a, b]) => {
    const hidden = landmarks[a]?.occluded || landmarks[b]?.occluded;
    const color = boneColor(b === 0 ? a : b);
    const radius = hidden ? 0.0034 : 0.005;
    const mesh = bone(pts[a], pts[b], radius, radius, overlayMaterial(color, hidden ? 0.55 : 0.96));
    if (mesh) {
      mesh.renderOrder = 21;
      overlay.add(mesh);
    }
  });

  pts.forEach((p, i) => {
    const hidden = Boolean(landmarks[i]?.occluded);
    const r = markRadius(i, hidden);
    const color = colorFor(i);
    const halo = new THREE.Mesh(
      isRight
        ? new THREE.CylinderGeometry(r + 0.004, r + 0.004, r * 0.22, 5)
        : new THREE.SphereGeometry(r + 0.004, 22, 18),
      overlayMaterial(0xffffff, hidden ? 0.45 : 1),
    );
    halo.position.copy(p);
    halo.renderOrder = 22;
    overlay.add(halo);
    const mark = new THREE.Mesh(
      isRight
        ? new THREE.CylinderGeometry(r, r, r * 0.46, 5)
        : new THREE.SphereGeometry(r, 22, 18),
      overlayMaterial(color, hidden ? 0.55 : 1),
    );
    mark.position.copy(p);
    mark.renderOrder = 23;
    overlay.add(mark);
  });

  tagOverlay(overlay);
  group.add(overlay);
}

function makeSkinMaterial(color, opacity, extra = {}) {
  const mat = new THREE.MeshPhysicalMaterial({
    color,
    roughness: extra.roughness ?? 0.48,
    metalness: 0.02,
    clearcoat: extra.clearcoat ?? 0.16,
    clearcoatRoughness: 0.5,
    sheen: 0.22,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(0xc99674),
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: opacity > 0.7,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  mat.userData.skin = true;
  mat.userData.solidOpacity = extra.solidOpacity ?? opacity;
  return mat;
}

function skinMaterial() {
  return makeSkinMaterial(0xc48a68, 0.36, { solidOpacity: 0.92 });
}

function creaseMaterial() {
  return makeSkinMaterial(0x8d5b42, 0.55, { roughness: 0.72, clearcoat: 0.04, solidOpacity: 0.95 });
}

function nailMaterial() {
  return makeSkinMaterial(0xf4d6c8, 0.7, { roughness: 0.22, clearcoat: 0.55, solidOpacity: 0.96 });
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
  palm.scale.set(across.length() * 0.52, along.length() * 0.42, Math.max(0.055, across.length() * 0.14));
  group.add(palm);

  const thenar = new THREE.Mesh(new THREE.SphereGeometry(across.length() * 0.2, 28, 22), material);
  thenar.position.copy(pts[0].clone().add(pts[1]).add(pts[5]).multiplyScalar(1 / 3));
  group.add(thenar);
}

function addJointCollar(group, joint, toward, radius, material) {
  const dir = toward.clone().sub(joint);
  if (dir.lengthSq() < 1e-8) return;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(radius * 1.02, radius * 0.2, 12, 24), material);
  ring.position.copy(joint);
  ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.normalize());
  group.add(ring);
}

function addNail(group, dip, tip, radius, material) {
  const dir = tip.clone().sub(dip);
  const len = dir.length();
  if (len < 1e-5) return;
  const axis = dir.normalize();
  const nail = new THREE.Mesh(new THREE.BoxGeometry(radius * 1.2, radius * 0.2, radius * 1.25), material);
  nail.position.copy(tip).addScaledVector(axis, radius * 0.12);
  nail.quaternion.setFromUnitVectors(Y_UP, axis);
  group.add(nail);
}

function addFinger(group, chain, radii, material, crease, nail) {
  for (let i = 0; i < chain.length - 1; i++) {
    const r0 = radii[i];
    const r1 = radii[Math.min(i + 1, radii.length - 1)];
    const mesh = bone(chain[i], chain[i + 1], r0 * 0.92, r1 * 0.88, material);
    if (mesh) group.add(mesh);
    const knuckle = new THREE.Mesh(new THREE.SphereGeometry(r0 * 1.12, 28, 22), material);
    knuckle.position.copy(chain[i]);
    group.add(knuckle);
    addJointCollar(group, chain[i], chain[i + 1], r0, crease);
  }
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(radii[radii.length - 1] * 0.95, 28, 22),
    material,
  );
  tip.position.copy(chain[chain.length - 1]);
  group.add(tip);
  addNail(group, chain[chain.length - 2], chain[chain.length - 1], radii[radii.length - 1], nail);
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

function buildManoHand(pts, meshData, apply, landmarks, handedness) {
  const group = new THREE.Group();
  const verts = meshData.vertices.map((v) => apply(v));
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

  const mesh = new THREE.Mesh(geo, makeSkinMaterial(0xc48a68, 0.36, { solidOpacity: 0.9 }));
  group.add(mesh);
  const crease = creaseMaterial();
  FINGER_CHAINS.forEach((finger) => {
    const ids = finger.ids;
    for (let i = 0; i < ids.length - 1; i++) {
      addJointCollar(group, pts[ids[i]], pts[ids[i + 1]], finger.radii[i] * SKIN_SCALE, crease);
    }
    addNail(
      group,
      pts[ids[ids.length - 2]],
      pts[ids[ids.length - 1]],
      finger.radii[finger.radii.length - 1] * SKIN_SCALE,
      nailMaterial(),
    );
  });
  addSkeleton(group, pts, landmarks, handedness);
  return group;
}

function buildAnatomicalHand(pts, landmarks, handedness) {
  const group = new THREE.Group();
  const skin = skinMaterial();
  const crease = creaseMaterial();
  const nail = nailMaterial();
  addPalm(group, pts, skin);
  FINGER_CHAINS.forEach((finger) => {
    const chain = finger.ids.map((id) => pts[id]);
    const radii = finger.radii.map((r) => r * SKIN_SCALE);
    addFinger(group, chain, radii, skin, crease, nail);
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

  scene.add(new THREE.AmbientLight(0xffffff, 0.28));
  scene.add(new THREE.HemisphereLight(0xf3e6d8, 0x3d3a38, 0.7));
  const key = new THREE.DirectionalLight(0xfff4ea, 1.25);
  key.position.set(-1.4, 2.6, 1.8);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9aa8bc, 0.32);
  fill.position.set(2.2, 0.2, 0.6);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffe2c4, 0.55);
  rim.position.set(0.2, 1.4, -2.2);
  scene.add(rim);

  return { scene, camera, renderer, controls, hand: null };
}

export class HandStudio3D {
  constructor() {
    this.views = {
      front: makeView(document.getElementById("view-front"), new THREE.Vector3(0, 0.08, 2.9)),
      threequarter: makeView(
        document.getElementById("view-threequarter"),
        new THREE.Vector3(1.95, 0.28, 2.15),
      ),
      side: makeView(document.getElementById("view-side"), new THREE.Vector3(2.9, 0.08, 0.18)),
    };
    this.displayMode = "xray";
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
      applyDisplayMode(v.hand, this.displayMode);
      v.scene.add(v.hand);
    });
    this.resize();
  }

  setDisplayMode(mode) {
    this.displayMode = mode;
    Object.values(this.views).forEach((v) => applyDisplayMode(v.hand, this.displayMode));
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
