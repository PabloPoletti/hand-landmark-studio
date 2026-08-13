import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { CONNECTIONS, FINGER_CHAINS, colorFor } from "./schema.js";

const Y_UP = new THREE.Vector3(0, 1, 0);

function toVec(p) {
  return new THREE.Vector3(p.x, -p.y, -p.z);
}

export function alignWorldLandmarks(world, handedness) {
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
  return pts;
}

function bone(a, b, radius, material) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-5) return null;
  const geo = new THREE.CylinderGeometry(radius, radius, len, 10, 1, false);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(Y_UP, dir.clone().normalize());
  return mesh;
}

function palmGeometry(pts) {
  const ids = [0, 1, 5, 9, 13, 17];
  const ring = ids.map((i) => pts[i]);
  const n = new THREE.Vector3()
    .crossVectors(
      new THREE.Vector3().subVectors(pts[5], pts[0]),
      new THREE.Vector3().subVectors(pts[17], pts[0]),
    )
    .normalize();
  const thick = 0.018;
  const top = ring.map((p) => p.clone().addScaledVector(n, thick));
  const bot = ring.map((p) => p.clone().addScaledVector(n, -thick));
  const cTop = new THREE.Vector3();
  const cBot = new THREE.Vector3();
  top.forEach((p) => cTop.add(p));
  bot.forEach((p) => cBot.add(p));
  cTop.multiplyScalar(1 / top.length);
  cBot.multiplyScalar(1 / bot.length);

  const positions = [];
  const push = (a, b, c) => {
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
  };
  for (let i = 0; i < top.length; i++) {
    const j = (i + 1) % top.length;
    push(cTop, top[i], top[j]);
    push(cBot, bot[j], bot[i]);
    push(top[i], bot[i], bot[j]);
    push(top[i], bot[j], top[j]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

function buildGhostHand(pts) {
  const group = new THREE.Group();
  const skin = new THREE.MeshPhysicalMaterial({
    color: 0xd9d4cc,
    roughness: 0.55,
    metalness: 0,
    transparent: true,
    opacity: 0.28,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  group.add(new THREE.Mesh(palmGeometry(pts), skin));

  const webbing = [
    [0, 1, 5],
    [0, 5, 9],
    [0, 9, 13],
    [0, 13, 17],
    [5, 9, 13],
    [9, 13, 17],
  ];
  webbing.forEach(([a, b, c]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([pts[a], pts[b], pts[c]]);
    geo.setIndex([0, 1, 2]);
    geo.computeVertexNormals();
    group.add(new THREE.Mesh(geo, skin));
  });

  FINGER_CHAINS.forEach((finger) => {
    const chain = finger.ids.map((id) => pts[id]);
    if (finger.key === "thumb") chain.unshift(pts[0]);
    for (let i = 0; i < chain.length - 1; i++) {
      const mesh = bone(chain[i], chain[i + 1], 0.032, skin);
      if (mesh) group.add(mesh);
    }
  });

  CONNECTIONS.forEach(([a, b]) => {
    const mesh = bone(
      pts[a],
      pts[b],
      0.01,
      new THREE.MeshBasicMaterial({ color: colorFor(b === 0 ? a : b) }),
    );
    if (mesh) group.add(mesh);
  });

  pts.forEach((p, i) => {
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 12, 10),
      new THREE.MeshBasicMaterial({ color: colorFor(i) }),
    );
    ball.position.copy(p);
    group.add(ball);

    const el = document.createElement("div");
    el.className = "label3d";
    el.textContent = String(i);
    const label = new CSS2DObject(el);
    label.position.copy(p);
    label.center.set(0.5, 1.35);
    group.add(label);
  });

  return group;
}

function makeView(container, cameraPos) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xd6d6d6);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
  camera.position.copy(cameraPos);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.15));
  const key = new THREE.DirectionalLight(0xffffff, 0.7);
  key.position.set(1.4, 2.2, 1.8);
  scene.add(key);

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

  setHand(worldLandmarks, handedness) {
    const pts = alignWorldLandmarks(worldLandmarks, handedness);
    Object.values(this.views).forEach((v) => {
      if (v.hand) v.scene.remove(v.hand);
      v.hand = buildGhostHand(pts);
      v.scene.add(v.hand);
    });
    this.resize();
  }

  clear() {
    Object.values(this.views).forEach((v) => {
      if (v.hand) {
        v.scene.remove(v.hand);
        v.hand = null;
      }
    });
  }
}
