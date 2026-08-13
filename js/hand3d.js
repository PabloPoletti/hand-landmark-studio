import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { CONNECTIONS, FINGER_CHAINS, colorFor } from "./schema.js";
import { createMarker, isLeftHand, setMarker } from "./notation.js";

const raycaster = new THREE.Raycaster();

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

function addSkeleton(group, pts, landmarks = [], handedness = "Right") {
  const left = isLeftHand(handedness);
  const bones = [];
  const joints = [];

  CONNECTIONS.forEach(([a, b]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([pts[a], pts[b]]);
    const mat = new THREE.LineDashedMaterial({
      color: colorFor(b === 0 ? a : b),
      dashSize: 1,
      gapSize: 0,
      transparent: true,
      opacity: 1,
    });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    line.userData = { a, b };
    bones.push(line);
    group.add(line);
  });

  pts.forEach((p, i) => {
    const hidden = Boolean(landmarks[i]?.occluded);
    const el = createMarker(i, colorFor(i), left, hidden);
    const label = new CSS2DObject(el);
    label.position.copy(p);
    label.center.set(0.5, 0.5);
    group.add(label);
    joints.push({ el, index: i, left });
  });

  group.userData.pts = pts;
  group.userData.bones = bones;
  group.userData.joints = joints;
  group.userData.left = left;
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
  geo.setIndex(meshData.faces.flat());
  geo.computeVertexNormals();

  const occluder = new THREE.Mesh(
    geo,
    new THREE.MeshPhysicalMaterial({
      color: 0xc9c4bc,
      roughness: 0.42,
      metalness: 0,
      transparent: true,
      opacity: 0.4,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  group.add(occluder);
  addSkeleton(group, pts, landmarks, handedness);
  group.userData.occluder = occluder;
  return group;
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

function buildGhostHand(pts, landmarks, handedness) {
  const group = new THREE.Group();
  const occluder = new THREE.Group();
  const skin = new THREE.MeshPhysicalMaterial({
    color: 0xd9d4cc,
    roughness: 0.55,
    metalness: 0,
    transparent: true,
    opacity: 0.28,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  occluder.add(new THREE.Mesh(palmGeometry(pts), skin));
  [
    [0, 1, 5],
    [0, 5, 9],
    [0, 9, 13],
    [0, 13, 17],
    [5, 9, 13],
    [9, 13, 17],
  ].forEach(([a, b, c]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([pts[a], pts[b], pts[c]]);
    geo.setIndex([0, 1, 2]);
    geo.computeVertexNormals();
    occluder.add(new THREE.Mesh(geo, skin));
  });

  FINGER_CHAINS.forEach((finger) => {
    const chain = finger.ids.map((id) => pts[id]);
    if (finger.key === "thumb") chain.unshift(pts[0]);
    for (let i = 0; i < chain.length - 1; i++) {
      const mesh = bone(chain[i], chain[i + 1], 0.028, skin);
      if (mesh) occluder.add(mesh);
    }
  });

  group.add(occluder);
  addSkeleton(group, pts, landmarks, handedness);
  group.userData.occluder = occluder;
  return group;
}

function hiddenFromView(camera, pts, occluder) {
  const cam = camera.position;
  const palm = new THREE.Vector3()
    .add(pts[0])
    .add(pts[5])
    .add(pts[9])
    .add(pts[13])
    .add(pts[17])
    .multiplyScalar(0.2);
  return pts.map((point) => {
    const dir = point.clone().sub(cam);
    const dist = dir.length();
    if (dist < 1e-5) return false;
    dir.multiplyScalar(1 / dist);
    if (occluder) {
      raycaster.set(cam, dir);
      const hits = raycaster.intersectObject(occluder, true);
      if (hits.length && hits[0].distance < dist - 0.03) return true;
    }
    return point.clone().sub(palm).dot(dir) < -0.02;
  });
}

function updateViewOcclusion(view) {
  const hand = view.hand;
  if (!hand?.userData?.pts) return;
  const { pts, bones, joints, occluder, left } = hand.userData;
  const hidden = hiddenFromView(view.camera, pts, occluder);
  joints.forEach((joint) => {
    const occ = hidden[joint.index];
    if (joint.occluded === occ) return;
    joint.occluded = occ;
    setMarker(joint.el, joint.index, colorFor(joint.index), left, occ);
  });
  bones.forEach((line) => {
    const occ = hidden[line.userData.a] || hidden[line.userData.b];
    line.material.dashSize = occ ? 0.045 : 2;
    line.material.gapSize = occ ? 0.035 : 0;
    line.material.opacity = occ ? 0.55 : 1;
    line.material.needsUpdate = true;
  });
}

function disposeHand(hand) {
  if (!hand) return;
  hand.traverse((obj) => {
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
        updateViewOcclusion(v);
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
    const hasMesh = mesh?.vertices?.length && mesh?.faces?.length;
    Object.values(this.views).forEach((v) => {
      if (v.hand) {
        v.scene.remove(v.hand);
        disposeHand(v.hand);
      }
      v.hand = hasMesh
        ? buildManoHand(aligned.pts, mesh, aligned.apply, landmarks, handedness)
        : buildGhostHand(aligned.pts, landmarks, handedness);
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
