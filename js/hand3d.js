import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { CONNECTIONS, colorFor } from "./schema.js";
import { isLeftHand } from "./notation.js";

const raycaster = new THREE.Raycaster();
const Y_UP = new THREE.Vector3(0, 1, 0);

function toVec(p) {
  if (Array.isArray(p)) return new THREE.Vector3(p[0], -p[1], -p[2]);
  return new THREE.Vector3(p.x, -p.y, -p.z);
}

function alignmentFromLandmarks(world) {
  const pts = world.map(toVec);
  const center = new THREE.Vector3();
  pts.forEach((p) => center.add(p));
  center.multiplyScalar(1 / pts.length);
  pts.forEach((p) => p.sub(center));
  let max = 0;
  pts.forEach((p) => {
    max = Math.max(max, p.length());
  });
  const scale = max > 0 ? 1.05 / max : 1;
  pts.forEach((p) => p.multiplyScalar(scale));
  return {
    pts,
    apply(raw) {
      return toVec(raw).sub(center).multiplyScalar(scale);
    },
  };
}

function bone(a, b, radius, material) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-5) return null;
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 8, 1, false), material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(Y_UP, dir.clone().normalize());
  return mesh;
}

function makeJoint(left, color) {
  const group = new THREE.Group();
  const fill = left
    ? new THREE.Mesh(
        new THREE.SphereGeometry(0.02, 12, 10),
        new THREE.MeshBasicMaterial({ color, depthTest: true }),
      )
    : new THREE.Mesh(
        new THREE.CircleGeometry(0.024, 5),
        new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, depthTest: true }),
      );
  fill.name = "fill";
  const ring = new THREE.Mesh(
    left
      ? new THREE.TorusGeometry(0.022, 0.0035, 8, 18)
      : new THREE.RingGeometry(0.018, 0.026, 5),
    new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, depthTest: false }),
  );
  ring.name = "ring";
  ring.visible = false;
  const slash = new THREE.Mesh(
    new THREE.CylinderGeometry(0.0028, 0.0028, 0.038, 6),
    new THREE.MeshBasicMaterial({ color, depthTest: false }),
  );
  slash.name = "slash";
  slash.rotation.z = Math.PI / 4;
  slash.visible = false;
  group.add(fill, ring, slash);
  return group;
}

function setJointOccluded(marker, occluded) {
  marker.getObjectByName("fill").visible = !occluded;
  marker.getObjectByName("ring").visible = occluded;
  marker.getObjectByName("slash").visible = occluded;
}

function addSkeleton(group, pts, handedness) {
  const left = isLeftHand(handedness);
  const bones = [];
  const joints = [];

  CONNECTIONS.forEach(([a, b]) => {
    const mat = new THREE.MeshBasicMaterial({
      color: colorFor(b === 0 ? a : b),
      transparent: true,
      opacity: 1,
    });
    const mesh = bone(pts[a], pts[b], 0.009, mat);
    if (!mesh) return;
    mesh.userData = { a, b };
    bones.push(mesh);
    group.add(mesh);
  });

  pts.forEach((p, i) => {
    const marker = makeJoint(left, colorFor(i));
    marker.position.copy(p);
    group.add(marker);

    const el = document.createElement("div");
    el.className = "lm-id";
    el.textContent = String(i);
    const label = new CSS2DObject(el);
    label.position.copy(p);
    group.add(label);
    joints.push({ marker, label, el, index: i });
  });

  group.userData.pts = pts;
  group.userData.bones = bones;
  group.userData.joints = joints;
}

function buildManoHand(pts, meshData, apply, handedness) {
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
      color: 0xcfc9c0,
      roughness: 0.45,
      metalness: 0,
      transparent: true,
      opacity: 0.38,
      side: THREE.DoubleSide,
      depthWrite: true,
    }),
  );
  group.add(occluder);
  addSkeleton(group, pts, handedness);
  group.userData.occluder = occluder;
  return group;
}

function buildGhostHand(pts, handedness) {
  const group = new THREE.Group();
  addSkeleton(group, pts, handedness);
  group.userData.occluder = null;
  return group;
}

function hiddenFromView(camera, pts, occluder) {
  const cam = camera.position;
  return pts.map((point) => {
    const dir = point.clone().sub(cam);
    const dist = dir.length();
    if (dist < 1e-5) return false;
    dir.multiplyScalar(1 / dist);
    if (occluder) {
      raycaster.set(cam, dir);
      const hits = raycaster.intersectObject(occluder, true);
      if (hits.length && hits[0].distance < dist - 0.02) return true;
    }
    const palm = pts[0].clone().add(pts[9]).multiplyScalar(0.5);
    return point.clone().sub(palm).dot(dir) < -0.03;
  });
}

function updateViewOcclusion(view) {
  const hand = view.hand;
  if (!hand?.userData?.pts) return;
  const { pts, bones, joints, occluder } = hand.userData;
  const hidden = hiddenFromView(view.camera, pts, occluder);
  const camQuat = view.camera.quaternion;
  joints.forEach((joint) => {
    const occ = hidden[joint.index];
    setJointOccluded(joint.marker, occ);
    joint.marker.quaternion.copy(camQuat);
    joint.el.style.display = occ ? "none" : "";
  });
  bones.forEach((mesh) => {
    const occ = hidden[mesh.userData.a] || hidden[mesh.userData.b];
    mesh.material.opacity = occ ? 0.28 : 1;
    mesh.material.transparent = true;
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
  const key = new THREE.DirectionalLight(0xffffff, 0.75);
  key.position.set(1.4, 2.2, 1.8);
  scene.add(key);

  return { scene, camera, renderer, labels, controls, hand: null };
}

export class HandStudio3D {
  constructor() {
    this.views = {
      front: makeView(document.getElementById("view-front"), new THREE.Vector3(0, 0.05, 2.5)),
      threequarter: makeView(
        document.getElementById("view-threequarter"),
        new THREE.Vector3(1.65, 0.3, 1.85),
      ),
      side: makeView(document.getElementById("view-side"), new THREE.Vector3(2.5, 0.08, 0.12)),
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

  setHand(worldLandmarks, handedness, mesh) {
    const aligned = alignmentFromLandmarks(worldLandmarks);
    const hasMesh = mesh?.vertices?.length && mesh?.faces?.length;
    Object.values(this.views).forEach((v) => {
      if (v.hand) {
        v.scene.remove(v.hand);
        disposeHand(v.hand);
      }
      v.hand = hasMesh
        ? buildManoHand(aligned.pts, mesh, aligned.apply, handedness)
        : buildGhostHand(aligned.pts, handedness);
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
