import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { CONNECTIONS, FINGER_CHAINS, colorFor } from "./schema.js";

const Y_UP = new THREE.Vector3(0, 1, 0);

function toVec(p) {
  if (Array.isArray(p)) return new THREE.Vector3(p[0], -p[1], -p[2]);
  return new THREE.Vector3(p.x, -p.y, -p.z);
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
  const geo = new THREE.CylinderGeometry(top, r0, len, 20, 1, false);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(Y_UP, dir.clone().normalize());
  return mesh;
}

const SKIN_SCALE = 1.45;
const SURFACE_PAD = 0.034;
const TIP_PREV = { 4: 3, 8: 7, 12: 11, 16: 15, 20: 19 };

function overlayMaterial(color, opacity = 1) {
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  mat.renderOrder = 20;
  return mat;
}

function palmNormal(pts) {
  const n = new THREE.Vector3().crossVectors(
    pts[5].clone().sub(pts[0]),
    pts[17].clone().sub(pts[0]),
  );
  if (n.lengthSq() < 1e-8) return new THREE.Vector3(0, 0, 1);
  return n.normalize();
}

function radiusForId(id) {
  if (id === 0) return 0.09;
  for (const finger of FINGER_CHAINS) {
    const idx = finger.ids.indexOf(id);
    if (idx >= 0) return finger.radii[idx] * SKIN_SCALE;
  }
  return 0.05;
}

function surfacePoints(pts, normal) {
  return pts.map((p, id) => {
    const dir = normal.clone();
    const prev = TIP_PREV[id];
    if (prev != null) {
      dir.add(pts[id].clone().sub(pts[prev]).normalize().multiplyScalar(0.75)).normalize();
    }
    return p.clone().add(dir.multiplyScalar(radiusForId(id) + SURFACE_PAD));
  });
}

function addSkeleton(group, pts, landmarks = [], handedness = "Unknown") {
  const isRight = handedness === "Right";
  const normal = palmNormal(pts);
  const outer = surfacePoints(pts, normal);
  const overlay = new THREE.Group();
  overlay.renderOrder = 20;

  CONNECTIONS.forEach(([a, b]) => {
    const hidden = landmarks[a]?.occluded || landmarks[b]?.occluded;
    const color = colorFor(b === 0 ? a : b);
    const radius = hidden ? 0.01 : 0.016;
    const outline = bone(outer[a], outer[b], radius + 0.007, radius + 0.007, overlayMaterial(0x111111, hidden ? 0.3 : 0.95));
    const mesh = bone(outer[a], outer[b], radius, radius, overlayMaterial(color, hidden ? 0.45 : 1));
    if (outline) {
      outline.renderOrder = 20;
      overlay.add(outline);
    }
    if (mesh) {
      mesh.renderOrder = 21;
      overlay.add(mesh);
    }
  });

  outer.forEach((p, i) => {
    const hidden = Boolean(landmarks[i]?.occluded);
    const r = hidden ? 0.028 : 0.038;
    const color = colorFor(i);
    let mark;
    if (isRight) {
      mark = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, r * 0.46, 5),
        overlayMaterial(color, hidden ? 0.45 : 1),
      );
      mark.quaternion.setFromUnitVectors(Y_UP, normal);
      const rim = new THREE.Mesh(
        new THREE.CylinderGeometry(r + 0.008, r + 0.008, r * 0.2, 5),
        overlayMaterial(0xffffff, hidden ? 0.45 : 1),
      );
      mark.add(rim);
    } else {
      const halo = new THREE.Mesh(new THREE.SphereGeometry(r + 0.008, 24, 18), overlayMaterial(0x111111, hidden ? 0.4 : 1));
      halo.position.copy(p);
      halo.renderOrder = 22;
      overlay.add(halo);
      mark = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), overlayMaterial(color, hidden ? 0.45 : 1));
    }
    mark.position.copy(p);
    mark.renderOrder = 23;
    overlay.add(mark);

    const el = document.createElement("div");
    el.className = hidden ? "label3d occluded" : "label3d";
    el.textContent = hidden ? `${i}*` : String(i);
    const label = new CSS2DObject(el);
    label.position.copy(p);
    label.position.x += 0.05;
    label.position.y += 0.035;
    overlay.add(label);
  });

  group.add(overlay);
}

function skinMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: 0xe0b89a,
    roughness: 0.48,
    metalness: 0,
    clearcoat: 0.18,
    clearcoatRoughness: 0.55,
    sheen: 0.3,
    sheenColor: new THREE.Color(0xf4d2bc),
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    side: THREE.DoubleSide,
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

  const palm = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), material);
  palm.position.copy(center);
  palm.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  palm.scale.set(across.length() * 0.58, along.length() * 0.46, Math.max(0.075, across.length() * 0.2));
  group.add(palm);

  const thenar = new THREE.Mesh(new THREE.SphereGeometry(across.length() * 0.26, 20, 16), material);
  thenar.position.copy(pts[0].clone().add(pts[1]).add(pts[5]).multiplyScalar(1 / 3));
  group.add(thenar);
}

function addFinger(group, chain, radii, material) {
  for (let i = 0; i < chain.length - 1; i++) {
    const r0 = radii[i];
    const r1 = radii[Math.min(i + 1, radii.length - 1)];
    const mesh = bone(chain[i], chain[i + 1], r0, r1, material);
    if (mesh) group.add(mesh);
    const knuckle = new THREE.Mesh(new THREE.SphereGeometry(r0 * 1.05, 20, 16), material);
    knuckle.position.copy(chain[i]);
    group.add(knuckle);
  }
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(radii[radii.length - 1] * 1.08, 20, 16),
    material,
  );
  tip.position.copy(chain[chain.length - 1]);
  group.add(tip);
}

function buildAnatomicalHand(pts, landmarks, handedness) {
  const group = new THREE.Group();
  const body = new THREE.Group();
  body.renderOrder = 0;
  const skin = skinMaterial();
  addPalm(body, pts, skin);
  FINGER_CHAINS.forEach((finger) => {
    const chain = finger.ids.map((id) => pts[id]);
    if (finger.key === "thumb") chain.unshift(pts[0]);
    const radii = finger.radii.map((r) => r * SKIN_SCALE);
    if (finger.key === "thumb") radii.unshift(radii[0] * 1.15);
    addFinger(body, chain, radii, skin);
  });
  group.add(body);
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
  scene.background = new THREE.Color(0xd6d6d6);
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
  renderer.toneMappingExposure = 1.12;
  container.appendChild(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.domElement.style.position = "absolute";
  labels.domElement.style.inset = "0";
  labels.domElement.style.pointerEvents = "none";
  container.appendChild(labels.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.minDistance = 1.4;
  controls.maxDistance = 5;

  scene.add(new THREE.AmbientLight(0xffffff, 0.28));
  scene.add(new THREE.HemisphereLight(0xf7f1e8, 0x8b8680, 0.8));
  const key = new THREE.DirectionalLight(0xfff7ee, 1.05);
  key.position.set(1.6, 2.4, 2.0);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xd7e4ff, 0.42);
  fill.position.set(-2.1, 0.5, 0.9);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xfff1dc, 0.5);
  rim.position.set(-0.3, 1.1, -2.1);
  scene.add(rim);

  return { scene, camera, renderer, labels, controls, hand: null };
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
        v.labels.render(v.scene, v.camera);
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
      v.labels.setSize(w, h);
    });
  }

  setHand(worldLandmarks, handedness, mesh, landmarks = []) {
    const aligned = alignmentFromLandmarks(worldLandmarks, handedness);
    Object.values(this.views).forEach((v) => {
      if (v.hand) {
        v.scene.remove(v.hand);
        disposeHand(v.hand);
      }
      v.hand = buildAnatomicalHand(aligned.pts, landmarks, handedness);
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
