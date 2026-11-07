// Robot Cuts Hair: replays DimOS-planned xArm7 moves and simulates scissors and hair.
// World frame matches DimOS: metres, z up, robot base at the origin.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const params = new URLSearchParams(location.search);
const DEMO = params.has("demo");
const MOBILE = matchMedia("(max-width: 720px)").matches;

const FLOOR_Z = -1.16; // the person sits in a chair, so the floor is well below the head
const STRANDS = MOBILE ? 14000 : 42000;
const CLUMPS = MOBILE ? 500 : 900;
const SEGMENTS = 8;
const HAIR_BANDS = 24;
const STRAND_WIDTH = MOBILE ? 0.00042 : 0.00026; // half width, metres
const MAX_PIECES = MOBILE ? 14000 : 40000;
const PIECE_NODES = 4;
// The tool, in the arm's tool frame (link_tcp): tool z points into the head, the fingers
// move along y. Keep in sync with haircut/head_model.py and verify_plans.py.
const PLANE_Z = -0.0167; // scissors lie at the finger pads
const HANDLE_LEN = 0.05; // finger ring to pivot
const BLADE_LEN = 0.07;
const BLADE_CENTER = new THREE.Vector3(HANDLE_LEN + BLADE_LEN / 2, 0, PLANE_Z);
const OPEN_ANGLE = THREE.MathUtils.degToRad(18);
const EDGE_OVERLAP = 0.002; // the edges cross the pivot's centre line, like real scissors
// Comb, in the tool frame: x along the blades, +z toward the scalp. It's a flat fan of tines
// just under the blades, shaped to fit inside the open jaws, so whatever it lifts ends up
// between the blades. Going in, hair passes between the tines; coming up, they carry it.
const COMB_X0 = 0.066;
const COMB_X1 = 0.122;
const COMB_FAN_FROM = 0.0476; // where the fan's edges would meet, at the open blades' pivot
const COMB_FAN = 0.317; // half-width per metre along x: the open jaws' slope
const COMB_FAN_MARGIN = 0.003; // inside the blade edges
const COMB_TOP = PLANE_Z + 0.0015; // the lower blade rides on it
const COMB_SPINE_END = 0.128; // the spine sits past the blade tips, clear of their closing
const COMB_SLOTS = 10; // gaps between the 11 tines
const COMB_MIDDLE = new THREE.Vector3((COMB_X0 + COMB_X1) / 2, 0, COMB_TOP + 0.0015);
const combHalfWidth = (x) => (x - COMB_FAN_FROM) * COMB_FAN - COMB_FAN_MARGIN;

const DRIVE_OPEN = 0.574; // pads 31 mm apart, blades at 18 degrees
const DRIVE_CLOSED = 0.825; // pads 3 mm apart, blades shut
const DRIVE_SPEED = 1.2; // rad/s; the URDF allows up to 2
const PAD_OFFSET = 0.0268; // finger link origin to its pad, from the model at full close
// Stop-and-snip cycle.
const SETTLE_TIME = 0.2; // seconds stopped before closing, while the hair on the comb settles

// Hair physics, stepped at a fixed rate so it doesn't depend on the frame rate.
const SIM_DT = 1 / 60;
const MAX_STEPS_PER_FRAME = 6;
const GRAVITY_STEP = 9.81 * SIM_DT * SIM_DT; // metres per step squared
// Real hair (70 micron keratin, E about 4 GPa) is stiff for its weight: held out flat, 3 cm of
// it droops about a millimetre and 6 cm about 30 degrees. This is weight per length over
// twice the bending stiffness, in 1/m^2, and sets how much each segment bends under gravity.
const HAIR_SAG = 5200;
const STRAND_DAMPING = 0.85;
const PIECE_DAMPING = 0.86; // air drag on a falling clipping
const CONTACT_BAND = 0.003; // how far under a surface still counts as touching it
const QUIET_MOVE = 0.00015; // metres per step below which a strand counts as still
const QUIET_STEPS = 30;
const SNAG_SPEED = 0.012; // m/s sideways with hair on the comb that would snag it in real life
const OPEN_ABOVE = 0.04; // metres off the scalp before the blades may reopen
const MAX_CONTACT_POINTS = 2500;

// Hair-to-hair contact uses a voxel grid: how crowded each cell is, and how fast its hair moves.
const GRID_CELLS = 56;
const GRID_SIZE = 0.006; // metres per cell
const HAIR_FRICTION = 0.22; // share of the velocity difference to the cell removed per step
const HAIR_PUSH = 0.000035; // metres per step for each unit of crowding difference
const HAIR_PUSH_MAX = 0.0007;
const WAKE_CELL_SPEED = 0.001; // metres per step of the hair around a sleeping strand that wakes it
const MAX_AWAKE = MOBILE ? 4000 : 10000;

// Mechanical figures. Hair: about 70 microns across, E about 4 GPa, breaks at about 0.8 N
// (a usual textbook value), dry friction about 0.2. Each simulated strand stands in for
// several real hairs on a head of about 100,000.
const HAIRS_PER_STRAND = 100000 / STRANDS;
const HAIR_EA = 15.4; // newtons per unit strain: stiffness times area
const HAIR_BREAK = 0.8; // newtons
const HOOK_SLIP_MIN = 0.1; // newtons before a caught strand slides along, varies by strand
const HOOK_SLIP_RANGE = 0.8;
const KNOT_SLIP = 0.25; // newtons before a knot pulls apart
const KNOT_STIFFNESS = 30; // newtons per metre
const MAX_KNOTS = MOBILE ? 2000 : 7000;
const KNOT_SPEED = 0.03 * SIM_DT; // metres per step of hair against its neighbours that can knot it
const CUT_FORCE = 0.012; // newtons to shear one hair with sharp blades, an estimate
const GRIPPER_FORCE = 30; // newtons, xArm Gripper's stated maximum
const LOAD_LIMIT = 20; // newtons of pull from hair at which the arm holds still
const SCISSOR_MASS = 0.15; // kilograms, an estimate
// The tool's outline for contact, two boxes: the blades and comb, then the fingers and palm.
const TOOL_ZONES = [
  [0.045, COMB_SPINE_END + 0.002, -0.03, 0.03, PLANE_Z - 0.012, COMB_TOP + 0.005],
  [-0.032, 0.032, -0.032, 0.032, -0.13, 0.004],
];
const inToolZone = (x, y, z, margin) => {
  for (const b of TOOL_ZONES) {
    if (x > b[0] - margin && x < b[1] + margin && y > b[2] - margin && y < b[3] + margin && z > b[4] - margin && z < b[5] + margin) return true;
  }
  return false;
};
// Solid parts as boxes: gripper fingers and palm, then the rail above the blades.
const TOOL_BOXES = [
  [-0.012, 0.012, 0.012, 0.036, -0.09, 0.002], // one finger
  [-0.012, 0.012, -0.036, -0.012, -0.09, 0.002], // the other
  [-0.04, 0.04, -0.045, 0.045, -0.15, -0.085], // palm
  [0.03, COMB_SPINE_END, -0.003, 0.003, PLANE_Z - 0.0105, PLANE_Z - 0.0055],
  [COMB_SPINE_END - 0.006, COMB_SPINE_END, -0.024, 0.024, COMB_TOP, COMB_TOP + 0.0025],
];

const HAIR_COLOURS = [
  ["Brown", "#4a2c18"],
  ["Black", "#1a1512"],
  ["Auburn", "#74331a"],
  ["Blond", "#b58f52"],
  ["Silver", "#a7a5a0"],
];

const $ = (id) => document.getElementById(id);
const setStatus = (text) => { $("status").textContent = text; };

// --- scene -----------------------------------------------------------------
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: DEMO });
renderer.setPixelRatio(DEMO ? 1 : Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = !MOBILE;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#10302a");
scene.fog = new THREE.Fog("#10302a", 4.2, 8);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(36, 1, 0.02, 20);
camera.up.set(0, 0, 1);
// On a phone the controls take the bottom of the screen, so frame the top half of the person there.
const WIDE = MOBILE
  ? { pos: new THREE.Vector3(0.95, -2.1, 0.32), target: new THREE.Vector3(0.42, -0.05, -0.42) }
  : { pos: new THREE.Vector3(0.95, -2.75, 0.42), target: new THREE.Vector3(0.36, -0.1, -0.42) };
const CLOSE = { pos: new THREE.Vector3(0.86, -0.66, 0.36), target: new THREE.Vector3(0.45, 0, 0.14) };
camera.position.copy(WIDE.pos);
const controls = new OrbitControls(camera, canvas);
controls.target.copy(WIDE.target);
controls.enableDamping = true;
controls.minDistance = 0.25;
controls.maxDistance = 4.5;
controls.maxPolarAngle = Math.PI * 0.62;
let cameraGoal = null;

const key = new THREE.DirectionalLight("#fff1d9", 2.6);
key.position.set(1.1, -1.0, 2.0);
key.target.position.set(0.3, 0, 0);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -1.5, right: 1.5, top: 1.6, bottom: -1.6, near: 0.5, far: 6 });
key.shadow.bias = -0.0005;
scene.add(key, key.target, new THREE.HemisphereLight("#d9efe6", "#2c211a", 0.7));

function resize() {
  const { innerWidth: w, innerHeight: h } = window;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w / h < 1 ? 64 : 36; // tall screens need a wider lens to fit arm and head
  // Keep the action clear of the side panel on desktop, and above it on phones.
  if (DEMO) camera.clearViewOffset();
  else if (w > 720) camera.setViewOffset(w, h, 150, 0, w, h);
  else camera.setViewOffset(w, h, 0, h * 0.2, w, h);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

function buildShop(plans) {
  // Checkered floor.
  const size = 512;
  const tile = document.createElement("canvas");
  tile.width = tile.height = size;
  const g = tile.getContext("2d");
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      g.fillStyle = (i + j) % 2 ? "#e9dfc8" : "#23211d";
      g.fillRect((i * size) / 2, (j * size) / 2, size / 2, size / 2);
    }
  }
  const floorTex = new THREE.CanvasTexture(tile);
  floorTex.colorSpace = THREE.SRGBColorSpace;
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.repeat.set(18, 18);
  floorTex.anisotropy = 8;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(9, 9),
    new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.45 }),
  );
  floor.position.set(0.3, 0, FLOOR_Z);
  floor.receiveShadow = true;
  scene.add(floor);

  // Pedestal the robot is bolted to.
  const pedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.1, 0.13, -FLOOR_Z, 48),
    new THREE.MeshStandardMaterial({ color: "#2b2a27", roughness: 0.4, metalness: 0.6 }),
  );
  pedestal.rotation.x = Math.PI / 2;
  pedestal.position.set(0, 0, FLOOR_Z / 2 - 0.001);
  pedestal.castShadow = pedestal.receiveShadow = true;
  scene.add(pedestal);

  // Barber pole, for atmosphere.
  const stripes = document.createElement("canvas");
  stripes.width = stripes.height = 256;
  const s = stripes.getContext("2d");
  s.fillStyle = "#f3ead7";
  s.fillRect(0, 0, 256, 256);
  s.lineWidth = 42;
  [["#c8321f", 0], ["#1f4e8c", 128]].forEach(([c, o]) => {
    s.strokeStyle = c;
    for (let k = -256; k < 512; k += 256) {
      s.beginPath();
      s.moveTo(k + o, 256 + 20);
      s.lineTo(k + o + 256, -20);
      s.stroke();
    }
  });
  const poleTex = new THREE.CanvasTexture(stripes);
  poleTex.colorSpace = THREE.SRGBColorSpace;
  poleTex.wrapS = poleTex.wrapT = THREE.RepeatWrapping;
  poleTex.repeat.set(1, 2.4);
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.06, 0.8, 32, 1, true),
    new THREE.MeshStandardMaterial({ map: poleTex, roughness: 0.25 }),
  );
  pole.rotation.x = Math.PI / 2;
  pole.position.set(1.15, 1.35, 0.16);
  pole.castShadow = true;
  scene.add(pole);
  const brass = new THREE.MeshStandardMaterial({ color: "#b98a3e", metalness: 1, roughness: 0.3 });
  for (const z of [0.56, -0.24]) {
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), brass);
    cap.position.set(1.15, 1.35, z);
    scene.add(cap);
  }
  return { poleTex };
}

// The person, sitting in a barber chair. x0 is the middle of the body, y points to their back.
// The cape's top matches the shoulders the planner avoided.
function buildClient(plans) {
  const group = new THREE.Group();
  const x0 = plans.shoulders.center[0];
  group.position.set(x0, 0, 0);

  const capeMat = new THREE.MeshStandardMaterial({ color: "#1b2d4a", roughness: 0.78, side: THREE.DoubleSide });
  const leather = new THREE.MeshStandardMaterial({ color: "#7a1f18", roughness: 0.42, metalness: 0.05 });
  const chrome = new THREE.MeshStandardMaterial({ color: "#c9ccce", roughness: 0.18, metalness: 1 });
  const trousers = new THREE.MeshStandardMaterial({ color: "#2c2a2a", roughness: 0.85 });
  const shoeMat = new THREE.MeshStandardMaterial({ color: "#20130c", roughness: 0.32 });
  const skin = new THREE.MeshPhysicalMaterial({ color: "#d9a489", roughness: 0.55, sheen: 0.4, sheenColor: new THREE.Color("#ff9c84") });

  const add = (geometry, material, x = 0, y = 0, z = 0) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.castShadow = mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };
  const rounded = (w, d, h, r = 0.02) => new RoundedBoxGeometry(w, d, h, 4, r);
  // A tube along straight points, radius changing along the way.
  const limb = (points, radii, material) => {
    const curve = new THREE.CatmullRomCurve3(points.map((q) => new THREE.Vector3(...q)));
    const geometry = new THREE.TubeGeometry(curve, 24, 1, 14, false);
    const pos = geometry.attributes.position;
    const middle = new THREE.Vector3();
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      const u = Math.floor(i / 15) / 24;
      const f = u * (radii.length - 1);
      const a = Math.min(Math.floor(f), radii.length - 2);
      curve.getPointAt(u, middle);
      v.fromBufferAttribute(pos, i).sub(middle).multiplyScalar(radii[a] + (radii[a + 1] - radii[a]) * (f - a)).add(middle);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geometry.computeVertexNormals();
    return add(geometry, material);
  };

  // Chair: base, hydraulic column, seat, back, arms, footrest.
  const cyl = (rTop, rBottom, h) => new THREE.CylinderGeometry(rTop, rBottom, h, 40).rotateX(Math.PI / 2);
  add(cyl(0.3, 0.32, 0.03), chrome, 0, 0, FLOOR_Z + 0.015);
  add(cyl(0.05, 0.12, 0.06), chrome, 0, 0, FLOOR_Z + 0.06);
  add(cyl(0.05, 0.05, 0.36), chrome, 0, 0, FLOOR_Z + 0.26);
  add(rounded(0.3, 0.3, 0.05), chrome, 0, 0, -0.76);
  add(rounded(0.52, 0.46, 0.09, 0.035), leather, 0, -0.02, -0.69);
  add(rounded(0.48, 0.09, 0.5, 0.04), leather, 0, 0.215, -0.42);
  add(rounded(0.05, 0.05, 0.3), chrome, 0, 0.215, -0.69);
  for (const side of [-1, 1]) {
    add(rounded(0.09, 0.46, 0.04), leather, side * 0.315, -0.12, -0.47);
    limb([[side * 0.3, 0.14, -0.68], [side * 0.32, 0.14, -0.57], [side * 0.315, 0.1, -0.49]], [0.016, 0.016], chrome);
    limb([[side * 0.3, -0.26, -0.68], [side * 0.32, -0.26, -0.57], [side * 0.315, -0.26, -0.49]], [0.016, 0.016], chrome);
  }
  limb([[0, -0.2, -0.75], [0, -0.42, -0.82], [0, -0.58, -1.05]], [0.02, 0.02], chrome);
  add(rounded(0.42, 0.24, 0.018, 0.008), chrome, 0, -0.6, -1.07);

  // Legs from under the cape, on the footrest.
  for (const side of [-1, 1]) {
    limb([[side * 0.11, -0.46, -0.7], [side * 0.115, -0.53, -0.82], [side * 0.115, -0.565, -1.0]], [0.058, 0.048, 0.039], trousers);
    const shoe = new THREE.SphereGeometry(1, 28, 18);
    const m = add(shoe, shoeMat, side * 0.115, -0.64, -1.03);
    m.scale.set(0.047, 0.135, 0.032);
    add(rounded(0.09, 0.09, 0.04), shoeMat, side * 0.115, -0.565, -1.022);
  }

  // Arms on the armrests, under the cape to the wrist.
  for (const side of [-1, 1]) {
    limb([[side * 0.2, 0.0, -0.33], [side * 0.285, -0.08, -0.42], [side * 0.315, -0.22, -0.43]], [0.06, 0.05, 0.043], capeMat);
    limb([[side * 0.315, -0.2, -0.428], [side * 0.315, -0.27, -0.43]], [0.034, 0.03], skin);
    const palm = add(new THREE.SphereGeometry(1, 20, 14), skin, side * 0.315, -0.3, -0.43);
    palm.scale.set(0.037, 0.05, 0.016);
    for (let f = 0; f < 4; f++) {
      const fx = side * (0.292 + f * 0.015);
      limb([[fx, -0.33, -0.43], [fx, -0.365, -0.434], [fx, -0.385, -0.446]], [0.0085, 0.008, 0.007], skin);
    }
    limb([[side * 0.355 - (side > 0 ? 0.075 : -0.075), -0.28, -0.432], [side * 0.28, -0.315, -0.438], [side * 0.28, -0.335, -0.444]], [0.01, 0.009, 0.008], skin);
  }

  // Cape: rings from the neck to the hem, draped forward over the thighs, with folds that grow
  // toward the hem. Each ring: height, half width, reach to the front, reach to the back.
  const rings = [
    [-0.078, 0.062, 0.055, 0.086], [-0.09, 0.08, 0.056, 0.098], [-0.1, 0.098, 0.058, 0.104],
    [-0.112, 0.138, 0.072, 0.112], [-0.135, 0.19, 0.098, 0.122], [-0.175, 0.218, 0.122, 0.13],
    [-0.25, 0.225, 0.14, 0.136],
    [-0.38, 0.215, 0.155, 0.135], [-0.5, 0.215, 0.21, 0.14], [-0.6, 0.235, 0.36, 0.15],
    [-0.68, 0.255, 0.5, 0.16], [-0.84, 0.28, 0.55, 0.17],
  ];
  const shape = new THREE.CatmullRomCurve3(rings.map((r) => new THREE.Vector3(r[1], r[2], r[0])));
  const backs = new THREE.CatmullRomCurve3(rings.map((r) => new THREE.Vector3(r[3], 0, r[0])));
  const ringAt = new THREE.Vector3();
  const backAt = new THREE.Vector3();
  const around = 128;
  const down = 60;
  const positions = [];
  const index = [];
  for (let j = 0; j <= down; j++) {
    const t = j / down;
    shape.getPoint(t, ringAt);
    backs.getPoint(t, backAt);
    const z = ringAt.z;
    const fold = THREE.MathUtils.smoothstep(t, 0.3, 1) * 0.017;
    for (let i = 0; i <= around; i++) {
      const th = (i / around) * Math.PI * 2;
      const c = Math.cos(th);
      const sn = Math.sin(th);
      const wave = Math.sin(th * 9 + 1.3 + t * 2) * 0.6 + Math.sin(th * 17 + t * 3) * 0.3 + Math.sin(th * 5 + 0.7) * 0.5;
      const reach = sn < 0 ? ringAt.y : backAt.x;
      positions.push((ringAt.x + fold * wave) * c, (reach + fold * wave) * sn, z + (j === down ? 0.012 * wave : 0));
    }
  }
  for (let j = 0; j < down; j++) {
    for (let i = 0; i < around; i++) {
      const v = j * (around + 1) + i;
      index.push(v, v + around + 1, v + 1, v + 1, v + around + 1, v + around + 2);
    }
  }
  const capeGeometry = new THREE.BufferGeometry();
  capeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  capeGeometry.setIndex(index);
  capeGeometry.computeVertexNormals();
  add(capeGeometry, capeMat);
  return group;
}

// What a falling clipping lands on: the highest surface under each spot, from the real meshes.
class Landing {
  constructor(group) {
    this.cell = 0.015;
    this.x0 = -1.2;
    this.y0 = -2;
    this.nx = Math.ceil(3 / this.cell);
    this.ny = Math.ceil(4 / this.cell);
    this.z = new Float32Array(this.nx * this.ny).fill(FLOOR_Z + 0.001);
    group.updateMatrixWorld(true);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    group.traverse((o) => {
      if (!o.isMesh || o.isInstancedMesh) return;
      const pos = o.geometry.attributes.position;
      const idx = o.geometry.index;
      const tris = idx ? idx.count / 3 : pos.count / 3;
      for (let t = 0; t < tris; t++) {
        const i0 = idx ? idx.getX(t * 3) : t * 3;
        const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
        const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
        a.fromBufferAttribute(pos, i0).applyMatrix4(o.matrixWorld);
        b.fromBufferAttribute(pos, i1).applyMatrix4(o.matrixWorld);
        c.fromBufferAttribute(pos, i2).applyMatrix4(o.matrixWorld);
        const n = Math.max(2, Math.ceil(Math.max(a.distanceTo(b), a.distanceTo(c)) / (this.cell * 0.7)));
        for (let u = 0; u <= n; u++) {
          for (let v = 0; v <= n - u; v++) {
            const w = n - u - v;
            this.raise((a.x * u + b.x * v + c.x * w) / n, (a.y * u + b.y * v + c.y * w) / n, (a.z * u + b.z * v + c.z * w) / n);
          }
        }
      }
    });
  }

  raise(x, y, z) {
    const i = Math.floor((x - this.x0) / this.cell);
    const j = Math.floor((y - this.y0) / this.cell);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return;
    const at = i * this.ny + j;
    if (z + 0.002 > this.z[at]) this.z[at] = z + 0.002;
  }

  height(x, y) {
    const i = Math.floor((x - this.x0) / this.cell);
    const j = Math.floor((y - this.y0) / this.cell);
    if (i < 1 || j < 1 || i >= this.nx - 1 || j >= this.ny - 1) return FLOOR_Z + 0.001;
    return this.z[i * this.ny + j];
  }

  // Which way is downhill here, per metre; used for sliding down the cape.
  slopeX(x, y) {
    return (this.height(x + this.cell, y) - this.height(x - this.cell, y)) / (2 * this.cell);
  }

  slopeY(x, y) {
    return (this.height(x, y + this.cell) - this.height(x, y - this.cell)) / (2 * this.cell);
  }
}

// --- robot -----------------------------------------------------------------
class Robot {
  constructor(kinematics, gltfScene) {
    this.root = new THREE.Group();
    this.joints = new Map();
    this.links = new Map([["world", this.root]]);
    const link = (name) => {
      if (!this.links.has(name)) this.links.set(name, new THREE.Group());
      return this.links.get(name);
    };
    for (const j of kinematics.joints) {
      const origin = new THREE.Group();
      origin.position.fromArray(j.xyz);
      origin.rotation.set(j.rpy[0], j.rpy[1], j.rpy[2], "ZYX");
      link(j.parent).add(origin);
      const child = link(j.child);
      origin.add(child);
      this.joints.set(j.name, { ...j, child, axis: new THREE.Vector3().fromArray(j.axis) });
    }

    const body = new THREE.MeshStandardMaterial({ color: "#ece8df", roughness: 0.38, metalness: 0.1 });
    const grip = new THREE.MeshStandardMaterial({ color: "#2a2b2d", roughness: 0.45, metalness: 0.3 });
    gltfScene.updateMatrixWorld(true);
    gltfScene.traverse((o) => {
      if (!o.isMesh) return;
      const name = o.name.replace(/_\d+$/, "");
      const target = this.links.get(name) ?? this.links.get(o.parent?.name);
      if (!target) return;
      const geometry = toCreasedNormals(o.geometry.clone(), 0.65);
      const gripper = /knuckle|finger|gripper/.test(name);
      const mesh = new THREE.Mesh(geometry, gripper ? grip : body);
      mesh.castShadow = mesh.receiveShadow = true;
      target.add(mesh);
    });
    this.tcp = link("link_tcp");
    this.finger = link("left_finger");
    this.drive = DRIVE_CLOSED;
    this.setGripper(this.drive);
  }

  // Half the gap between the finger pads, read from the arm's actual finger position.
  padHalfGap() {
    this.tmp ??= new THREE.Vector3();
    this.finger.updateWorldMatrix(true, false);
    this.tcp.updateWorldMatrix(true, false);
    this.tmp.setFromMatrixPosition(this.finger.matrixWorld);
    this.tcp.worldToLocal(this.tmp);
    return Math.abs(this.tmp.y) - PAD_OFFSET;
  }

  setGripper(drive) {
    this.drive = drive;
    for (const j of this.joints.values()) {
      if (j.name === "drive_joint") this.applyJoint(j, drive);
      else if (j.mimic?.joint === "drive_joint") {
        this.applyJoint(j, drive * j.mimic.multiplier + j.mimic.offset);
      }
    }
  }

  applyJoint(j, angle) {
    j.child.quaternion.setFromAxisAngle(j.axis, angle);
  }

  setArm(q) {
    for (let i = 0; i < 7; i++) this.applyJoint(this.joints.get(`joint${i + 1}`), q[i]);
  }
}

// Scissors with a ring around each gripper finger, a rail that guides the pivot, and a
// comb under the blades. Nothing here animates by itself: update() takes the gap between
// the fingers and solves the linkage.
function buildTool() {
  const group = new THREE.Group();
  group.position.z = PLANE_Z;
  const steel = new THREE.MeshStandardMaterial({ color: "#d9dde0", metalness: 1, roughness: 0.2 });
  const handleMat = new THREE.MeshStandardMaterial({ color: "#c8321f", roughness: 0.35 });
  const frameMat = new THREE.MeshStandardMaterial({ color: "#35373a", metalness: 0.6, roughness: 0.4 });
  const combMat = new THREE.MeshStandardMaterial({ color: "#16171a", roughness: 0.5 });

  const bladeShape = new THREE.Shape();
  bladeShape.moveTo(-0.005, 0.0005 + EDGE_OVERLAP);
  bladeShape.lineTo(BLADE_LEN, EDGE_OVERLAP);
  bladeShape.lineTo(BLADE_LEN - 0.012, -0.0075);
  bladeShape.lineTo(-0.005, -0.008);
  bladeShape.closePath();

  // `side` is the finger this half's ring goes around (+1 or -1 along y).
  const makeHalf = (side) => {
    const half = new THREE.Group();
    const z = side > 0 ? -0.0014 : 0.0002;
    const blade = new THREE.Mesh(
      new THREE.ExtrudeGeometry(bladeShape, { depth: 0.0012, bevelEnabled: false }),
      steel,
    );
    blade.position.z = z;
    blade.scale.y = side; // the sharp edges face each other
    half.add(blade);
    const shank = new THREE.Mesh(new THREE.BoxGeometry(HANDLE_LEN - 0.006, 0.0055, 0.0032), handleMat);
    shank.position.set(-(HANDLE_LEN - 0.006) / 2, 0, z);
    half.add(shank);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.0095, 0.0024, 12, 32), handleMat);
    ring.position.set(-HANDLE_LEN, 0, z);
    half.add(ring);
    for (const m of [blade, shank, ring]) m.castShadow = true;
    return half;
  };
  const a = makeHalf(1);
  const b = makeHalf(-1);
  const scissors = new THREE.Group();
  scissors.add(a, b);
  const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.009, 16), steel);
  pin.rotation.x = Math.PI / 2;
  pin.position.z = -0.003;
  scissors.add(pin);
  group.add(scissors);

  const box = (w, d, h, x, y, z, material) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), material);
    m.position.set(x, y, z);
    m.castShadow = true;
    group.add(m);
    return m;
  };
  // Rail fixed to the gripper's palm; the pivot pin rides along it as the blades open.
  const railZ = -0.008;
  const railEnd = HANDLE_LEN + BLADE_LEN + 0.006;
  box(railEnd - 0.03, 0.005, 0.004, (railEnd + 0.03) / 2, 0, railZ, frameMat);
  box(0.006, 0.012, 0.03, 0.033, 0, railZ - 0.015, frameMat); // strut up to the palm
  box(0.006, 0.005, 0.015, COMB_SPINE_END - 0.002, 0, railZ + 0.0065, frameMat); // drop down to the comb

  // Comb under the blades: a spine past the blade tips, with tines fanning back toward the
  // pivot so they stay inside the jaws. Hair stands up between the tines.
  const local = (z) => z - PLANE_Z; // this group sits at the blade plane
  const spineX = COMB_SPINE_END - 0.002;
  const farWidth = combHalfWidth(COMB_X1);
  box(0.004, farWidth * 2 + 0.002, 0.0025, spineX, 0, local(COMB_TOP) + 0.00125, combMat);
  const tines = 11;
  const tine = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 0.0013, 0.0015), combMat, tines);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const zAxis = new THREE.Vector3(0, 0, 1);
  const at = new THREE.Vector3();
  const size = new THREE.Vector3();
  for (let i = 0; i < tines; i++) {
    const f = (i / (tines - 1)) * 2 - 1;
    const y0 = combHalfWidth(COMB_X0) * f;
    const y1 = farWidth * f;
    const x1 = spineX;
    const length = Math.hypot(x1 - COMB_X0, y1 - y0);
    q.setFromAxisAngle(zAxis, Math.atan2(y1 - y0, x1 - COMB_X0));
    at.set((COMB_X0 + x1) / 2, (y0 + y1) / 2, local(COMB_TOP) + 0.00075);
    tine.setMatrixAt(i, m.compose(at, q, size.set(length, 1, 1)));
  }
  tine.castShadow = true;
  group.add(tine);

  group.angle = 0;
  group.pivotX = HANDLE_LEN;
  group.update = (padHalfGap) => {
    // The ring is held at the pad, so the blade angle follows from the triangle
    // finger - pivot - centre line.
    const g = Math.max(0, Math.min(HANDLE_LEN, padHalfGap));
    group.angle = Math.asin(g / HANDLE_LEN);
    group.pivotX = Math.sqrt(HANDLE_LEN * HANDLE_LEN - g * g);
    scissors.position.x = group.pivotX;
    a.rotation.z = -group.angle; // ring on +y finger, blade swings to -y
    b.rotation.z = group.angle;
  };
  group.update(0);
  return group;
}

// --- head ------------------------------------------------------------------
// Distance from the skull centre to the scanned surface, by direction.
class Scalp {
  constructor(data) {
    this.center = new THREE.Vector3().fromArray(data.center);
    this.lon = data.lon;
    this.lat = data.lat;
    this.map = data.radius_mm10;
    this.maxRadius = Math.max(...data.radius_mm10) * 1e-4;
    // A direction grid, so the physics doesn't need trigonometry for every point.
    const n = (this.gridSize = 64);
    this.grid = new Float32Array(n * n * n);
    for (let i = 0, at = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++, at++) {
          const x = (i / (n - 1)) * 2 - 1;
          const y = (j / (n - 1)) * 2 - 1;
          const z = (k / (n - 1)) * 2 - 1;
          const r = Math.hypot(x, y, z) || 1;
          this.grid[at] = this.radius(x / r, y / r, z / r);
        }
      }
    }
  }

  // The same radius, read from the grid; x, y, z must be a unit direction.
  fastRadius(x, y, z) {
    const n = this.gridSize;
    const s = (n - 1) * 0.5;
    const i = ((x + 1) * s + 0.5) | 0;
    const j = ((y + 1) * s + 0.5) | 0;
    const k = ((z + 1) * s + 0.5) | 0;
    return this.grid[(i * n + j) * n + k];
  }

  // Radius in metres toward a unit direction (bilinear lookup).
  radius(x, y, z) {
    const u = ((Math.atan2(y, x) + Math.PI) / (2 * Math.PI)) * this.lon - 0.5;
    const v = ((Math.asin(Math.max(-1, Math.min(1, z))) + Math.PI / 2) / Math.PI) * this.lat - 0.5;
    const i0 = Math.floor(u);
    const j0 = Math.max(0, Math.min(this.lat - 2, Math.floor(v)));
    const fu = u - i0;
    const fv = Math.max(0, Math.min(1, v - j0));
    const at = (i, j) => this.map[j * this.lon + ((i + this.lon) % this.lon)];
    const a = at(i0, j0) * (1 - fu) + at(i0 + 1, j0) * fu;
    const b = at(i0, j0 + 1) * (1 - fu) + at(i0 + 1, j0 + 1) * fu;
    return (a * (1 - fv) + b * fv) * 1e-4;
  }

  // How far a world point is outside the head surface, in metres.
  height(p) {
    const dx = p.x - this.center.x;
    const dy = p.y - this.center.y;
    const dz = p.z - this.center.z;
    const r = Math.hypot(dx, dy, dz) || 1e-6;
    return r - this.radius(dx / r, dy / r, dz / r);
  }
}

// The hairline: for a unit direction from the skull centre, does hair grow there?
// The face looks toward -y. Kept in step with HAIRLINE_GLSL below.
function hairlineZ(y) {
  const f = Math.max(-1, Math.min(1, y));
  return f < -0.2 ? -0.14 + 0.66 * Math.pow((-0.2 - f) / 0.8, 1.3) : -0.14 - (f + 0.2) * 0.52;
}
const onScalp = (x, y, z) => z > hairlineZ(y) && !(Math.abs(x) > 0.84 && z < 0.06 && y > -0.3 && y < 0.42);
const HAIRLINE_GLSL = /* glsl */ `
  float hairMask(vec3 n) {
    float f = clamp(n.y, -1.0, 1.0);
    float line = f < -0.2 ? -0.14 + 0.66 * pow((-0.2 - f) / 0.8, 1.3) : -0.14 - (f + 0.2) * 0.52;
    float mask = smoothstep(line - 0.015, line + 0.1, n.z);
    float ear = smoothstep(0.8, 0.86, abs(n.x)) * (1.0 - smoothstep(0.02, 0.1, n.z))
      * smoothstep(-0.35, -0.25, n.y) * (1.0 - smoothstep(0.37, 0.47, n.y));
    return mask * (1.0 - ear);
  }`;

function buildHead(gltf, maps, scalp, place) {
  const source = gltf.scene.getObjectByProperty("type", "Mesh");
  const shadowUniform = { value: new THREE.Color("#3a2416") };
  const material = new THREE.MeshPhysicalMaterial({
    map: maps.colour,
    normalMap: maps.normal,
    normalScale: new THREE.Vector2(0.8, 0.8),
    roughness: 0.52,
    sheen: 0.45,
    sheenColor: new THREE.Color("#ff9c84"),
    sheenRoughness: 0.5,
    clearcoat: 0.08,
    clearcoatRoughness: 0.55,
    envMapIntensity: 0.55,
  });
  // Skin under the hair is in shade and tinted by the roots.
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uSkullCenter = { value: scalp.center };
    shader.uniforms.uScalpTint = shadowUniform;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWorld;")
      .replace("#include <project_vertex>", "#include <project_vertex>\nvWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>\nvarying vec3 vWorld;\nuniform vec3 uSkullCenter;\nuniform vec3 uScalpTint;\n${HAIRLINE_GLSL}`,
      )
      .replace(
        "#include <map_fragment>",
        `#include <map_fragment>
         if (vWorld.z < -0.1) discard; // the scan's chest and back, which the cape covers
         diffuseColor.rgb = mix(diffuseColor.rgb, uScalpTint * 0.55, hairMask(normalize(vWorld - uSkullCenter)) * 0.88);`,
      );
  };
  const mesh = new THREE.Mesh(source.geometry, material);
  mesh.matrixAutoUpdate = false;
  const a = place.axes;
  const s = place.scale;
  const t = place.translation;
  mesh.matrix.set(
    a[0][0] * s, a[0][1] * s, a[0][2] * s, t[0],
    a[1][0] * s, a[1][1] * s, a[1][2] * s, t[1],
    a[2][0] * s, a[2][1] * s, a[2][2] * s, t[2],
    0, 0, 0, 1,
  );
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.setScalpTint = (hex) => shadowUniform.value.set(hex);
  return mesh;
}

// --- hair ------------------------------------------------------------------
// How each hairstyle grows. n is the direction from the skull centre to the root, top is 0 at
// the sides to 1 on the crown, front is 1 at the forehead. Each guide hair starts along a mix of
// straight out and `comb`, then each step follows `comb`, sags, and is pulled to the head.
const HAIRSTYLES = [
  {
    name: "Back", label: "Combed back",
    length: [0.038, 0.05],
    start: [0.34, 0.56, 0.62],
    comb: (n, front, c) => c.set(Math.sign(n.x || 1) * Math.min(1, Math.abs(n.x) * 3) * 0.55, 0.85, -0.55 + 1.2 * front),
    follow: 0.13, sag: [0.03, 0.009], frontSag: 0.8, hug: [0.058, 0.036], wave: [0.05, 0.8], frizz: 1,
  },
  {
    name: "Side", label: "Side part",
    length: [0.035, 0.045],
    start: [0.26, 0.36, 0.75],
    // A parting on one side of the crown: hair runs away from it across the head and down.
    comb: (n, front, c) => {
      const away = Math.sign(n.x - 0.32) || 1;
      return c.set(away * 0.95, 0.28 - 0.5 * front, -0.3 + 0.35 * front);
    },
    follow: 0.14, sag: [0.03, 0.01], frontSag: 0.6, hug: [0.06, 0.038], wave: [0.03, 0.8], frizz: 0.7,
  },
  {
    name: "Wavy", label: "Wavy",
    length: [0.05, 0.05],
    start: [0.5, 0.5, 0.5],
    comb: (n, front, c) => c.set(Math.sign(n.x || 1) * Math.min(1, Math.abs(n.x) * 3) * 0.7, 0.5, -0.65 + 0.9 * front),
    follow: 0.1, sag: [0.03, 0.008], frontSag: 0.7, hug: [0.05, 0.03], wave: [0.2, 1.15], frizz: 0.8,
  },
  {
    name: "Long", label: "Long",
    length: [0.135, 0.04],
    start: [0.26, 0.2, 0.75],
    comb: (n, front, c) => c.set(Math.sign(n.x || 1) * Math.min(1, Math.abs(n.x) * 3) * 0.8, 0.45, -0.75 + 1.1 * front),
    follow: 0.1, sag: [0.05, 0.024], frontSag: 0.5, hug: [0.085, 0.03], wave: [0.05, 0.6], frizz: 0.7, gather: 0.2,
  },
  {
    name: "Spiky", label: "Spiky",
    length: [0.03, 0.025],
    start: [1, 0.2, 0.15],
    comb: (n, front, c) => c.set(Math.sign(n.x || 1) * 0.15, 0.3, 0.2),
    follow: 0.02, sag: [0, 0], frontSag: 0, hug: [0, 0], wave: [0.1, 0.6], frizz: 1,
  },
];

// How crowded the hair is in each cell of a grid around the head, and how it's moving there.
// Sleeping strands are counted in `still`; awake ones are added every step.
class HairGrid {
  constructor(center) {
    const n = GRID_CELLS;
    this.n = n;
    this.x0 = center.x - (n * GRID_SIZE) / 2;
    this.y0 = center.y - (n * GRID_SIZE) / 2;
    this.z0 = center.z - (n * GRID_SIZE) / 2;
    const size = n * n * n;
    this.still = new Float32Array(size);
    this.rest = new Float32Array(size);
    this.moving = new Float32Array(size);
    this.mx = new Float32Array(size);
    this.my = new Float32Array(size);
    this.mz = new Float32Array(size);
    this.lastNode = new Int32Array(size);
    this.touched = new Int32Array(size);
    this.touchedCount = 0;
  }

  // The cell for a point, or -1 outside the grid or on its edge.
  index(x, y, z) {
    const n = this.n;
    const i = Math.floor((x - this.x0) / GRID_SIZE);
    const j = Math.floor((y - this.y0) / GRID_SIZE);
    const k = Math.floor((z - this.z0) / GRID_SIZE);
    if (i < 1 || j < 1 || k < 1 || i >= n - 1 || j >= n - 1 || k >= n - 1) return -1;
    return (i * n + j) * n + k;
  }

  clearMoving() {
    const { touched, moving, mx, my, mz } = this;
    for (let t = 0; t < this.touchedCount; t++) {
      const c = touched[t];
      moving[c] = mx[c] = my[c] = mz[c] = 0;
    }
    this.touchedCount = 0;
  }

  // The crowding each cell started with is what its hair is happy at.
  takeRest() {
    const { still, rest, n } = this;
    const nn = n * n;
    rest.fill(0);
    for (let c = nn + n + 1; c < rest.length - nn - n - 1; c++) {
      // At least what's there now, so hair at rest feels no push at all.
      rest[c] = Math.max(still[c], (still[c] * 2 + still[c - 1] + still[c + 1] + still[c - n] + still[c + n] + still[c - nn] + still[c + nn]) / 8);
    }
  }

  // How much over its resting crowding a cell is.
  over(c) {
    const extra = this.still[c] + this.moving[c] - this.rest[c] * 1.25 - 3;
    return extra > 0 ? extra : 0;
  }
}

// Strands grow in clumps that follow a guide hair: up off the scalp, then away from
// the crown and down with gravity, staying outside the head.
class Hair {
  constructor(scalp, landing) {
    this.scalp = scalp;
    this.landing = landing;
    this.center = scalp.center;
    const pointsPerStrand = SEGMENTS + 1;
    const vertsPerStrand = pointsPerStrand * 2;
    this.polyline = new Float32Array(STRANDS * pointsPerStrand * 3);
    this.length = new Float32Array(STRANDS).fill(1);
    this.fullLength = new Float32Array(STRANDS);
    // Physics state. Only strands something has touched are simulated.
    this.lastNode = new Uint8Array(STRANDS).fill(SEGMENTS); // nodes past this were cut off
    this.lastLen = new Float32Array(STRANDS); // length of the final, possibly cut, segment
    this.restDir = new Float32Array(STRANDS * SEGMENTS * 3);
    this.segLen = new Float32Array(STRANDS * SEGMENTS);
    this.awake = new Uint8Array(STRANDS);
    this.quiet = new Uint8Array(STRANDS);
    this.active = new Int32Array(STRANDS);
    this.activeCount = 0;
    this.stepCount = 0;
    this.contactCount = 0; // points resting on the comb this step
    this.contactPoints = new Float32Array(MAX_CONTACT_POINTS * 3);
    this.stiff = new Float32Array(SEGMENTS + 1);
    for (let k = 1; k <= SEGMENTS; k++) {
      const u = (k - 1) / (SEGMENTS - 1);
      this.stiff[k] = 0.35 - 0.17 * u; // how fast a bent segment springs back, per step
    }
    // Strands are grouped into bands across the head, and sorted within each, so the patch
    // under the tool can be sent to the GPU as a few short ranges.
    this.bandOf = new Uint8Array(STRANDS);
    this.dirtyMin = new Int32Array(HAIR_BANDS).fill(2147483647);
    this.dirtyMax = new Int32Array(HAIR_BANDS).fill(-1);
    this.lengthsDirty = false;
    const verts = STRANDS * vertsPerStrand;
    this.positions = new Float32Array(verts * 3);
    this.lengths = new Float32Array(verts).fill(1);
    this.tangents = new Float32Array(verts * 3);
    const uAttr = new Float32Array(verts);
    const sideAttr = new Float32Array(verts);
    const shadeAttr = new Float32Array(verts);
    const index = new Uint32Array(STRANDS * SEGMENTS * 6);

    this.style = HAIRSTYLES[0];
    this.grow(this.style, uAttr, sideAttr, shadeAttr, index);

    const geometry = new THREE.BufferGeometry();
    this.positionAttr = new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", this.positionAttr);
    this.lengthAttr = new THREE.BufferAttribute(this.lengths, 1).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("aLength", this.lengthAttr);
    this.tangentAttr = new THREE.BufferAttribute(this.tangents, 3).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("aTangent", this.tangentAttr);
    geometry.setAttribute("aU", new THREE.BufferAttribute(uAttr, 1));
    geometry.setAttribute("aSide", new THREE.BufferAttribute(sideAttr, 1));
    geometry.setAttribute("aShade", new THREE.BufferAttribute(shadeAttr, 1));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uColour: { value: new THREE.Color() },
        uKeyDir: { value: key.position.clone().sub(key.target.position).normalize() },
        uKeyColour: { value: key.color.clone().multiplyScalar(key.intensity) },
        uWidth: { value: STRAND_WIDTH },
      },
      vertexShader: HAIR_VERTEX,
      fragmentShader: HAIR_FRAGMENT,
      side: THREE.DoubleSide,
      alphaToCoverage: true,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;

    this.pos = this.polyline.slice();
    this.prev = this.polyline.slice();
    this.grid = new HairGrid(this.center);
    for (let i = 0; i < STRANDS; i++) this.addToGrid(i, 1);
    this.grid.takeRest();
    this.pieces = new Pieces(scalp, this.material, this.grid);
    this.cutCount = 0;
    this.cutTotal = 0;
    // Strands caught on the tool, and knots between strands.
    this.bladeForce = 0;
    this.hookNode = new Int8Array(STRANDS).fill(-1);
    this.hookAt = new Float32Array(STRANDS * 3); // where on the tool, in its frame
    this.hookSlip = new Float32Array(STRANDS);
    this.hookedCount = 0;
    this.knotA = new Int32Array(MAX_KNOTS);
    this.knotB = new Int32Array(MAX_KNOTS);
    this.knotLen = new Float32Array(MAX_KNOTS);
    this.knotCount = 0;
    this.knotsMade = 0;
    this.knotsSlipped = 0;
    this.slid = 0;
    this.broken = 0;
    this.toolForce = new THREE.Vector3(); // newtons the hair pulls the tool with
    this.toolLoad = 0;
    this.seed = 12345;
    for (let i = 0; i < STRANDS; i++) this.writeStrand(i);
    this.dirtyMin.fill(2147483647);
    this.dirtyMax.fill(-1);
  }

  // Lay out every strand for a hairstyle. The same random seed gives the same roots every time.
  grow(style, uAttr, sideAttr, shadeAttr, index) {
    const pointsPerStrand = SEGMENTS + 1;
    const vertsPerStrand = pointsPerStrand * 2;
    const scalp = this.scalp;
    const rand = mulberry(11);
    const unit = () => {
      const z = rand() * 2 - 1;
      const a = rand() * Math.PI * 2;
      const r = Math.sqrt(1 - z * z);
      return new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), z);
    };
    const scalpDir = () => {
      for (;;) {
        const n = unit();
        // Hair thins out toward the hairline instead of stopping dead.
        if (onScalp(n.x, n.y, n.z) && rand() < 0.12 + (n.z - hairlineZ(n.y)) / 0.09) return n;
      }
    };

    const combed = new THREE.Vector3();
    const guideSteps = 14;
    const guides = [];
    for (let c = 0; c < CLUMPS; c++) {
      const n = scalpDir();
      // Longer on top, short at the sides and nape.
      const top = THREE.MathUtils.smoothstep(n.z, -0.1, 0.7);
      const length = (style.length[0] + style.length[1] * top) * (0.88 + rand() * 0.24);
      // Combed back and down, and outward to each side from a soft centre line.
      const front = THREE.MathUtils.smoothstep(-n.y, 0.2, 0.75);
      style.comb(n, front, combed);
      const flow = combed.clone().addScaledVector(n, -combed.dot(n));
      if (flow.lengthSq() < 1e-4) flow.set(0, 1, 0).addScaledVector(n, -n.y);
      flow.normalize();
      const wave = unit().multiplyScalar(style.wave[0]);
      const wavePhase = rand() * Math.PI * 2;
      const d = n.clone().multiplyScalar(style.start[0] + style.start[1] * top).addScaledVector(flow, style.start[2]).add(unit().multiplyScalar(0.08)).normalize();
      const p = n.clone().multiplyScalar(scalp.radius(n.x, n.y, n.z)).add(this.center);
      const pts = [];
      const step = length / guideSteps;
      for (let k = 0; k <= guideSteps; k++) {
        pts.push(p.clone());
        // Once it's below the skull, hair just hangs: nothing pulls it in toward the neck.
        const hanging = THREE.MathUtils.smoothstep(this.center.z - p.z, -0.02, 0.04);
        d.addScaledVector(flow, style.follow * (1 - 0.85 * hanging)).add(new THREE.Vector3(0, 0, -(style.sag[0] + style.sag[1] * k) * (1 - style.frontSag * front)));
        // Pull back toward the scalp as it grows, so the hair lies on the head.
        d.addScaledVector(n, -(style.hug[0] - style.hug[1] * top) * k * (1 - hanging));
        d.addScaledVector(wave, Math.sin(k * style.wave[1] + wavePhase));
        d.normalize();
        p.addScaledVector(d, step);
        this.pushOut(p, 0.0016);
        this.restOnBody(p);
      }
      guides.push({ n, pts });
    }

    const bandFor = (d) => Math.min(HAIR_BANDS - 1, Math.floor(((d.y + 1) / 2) * HAIR_BANDS));
    const roots = Array.from({ length: STRANDS }, scalpDir).sort((a, b) => bandFor(a) - bandFor(b) || a.x - b.x);

    const n = new THREE.Vector3();
    const p = new THREE.Vector3();
    const prev = new THREE.Vector3();
    const q = new THREE.Vector3();
    const frizz = new THREE.Vector3();
    const dirs = this.restDir;
    this.fullLength.fill(0);
    for (let i = 0; i < STRANDS; i++) {
      n.copy(roots[i]);
      this.bandOf[i] = bandFor(n);
      // Nearest clump within a sample of them: fast, and a little messy is natural.
      let guide = guides[0];
      let best = -2;
      for (let t = 0; t < 48; t++) {
        const g = guides[Math.floor(rand() * CLUMPS)];
        const dot = g.n.dot(n);
        if (dot > best) {
          best = dot;
          guide = g;
        }
      }
      const root = n.clone().multiplyScalar(scalp.radius(n.x, n.y, n.z)).add(this.center);
      const rootOffset = root.clone().sub(guide.pts[0]);
      const keep = 0.8 + rand() * 0.2; // uneven tips
      const shade = 0.72 + rand() * 0.42;
      const stray = rand() < 0.02 ? 2.6 : 1; // a few flyaways
      frizz.copy(unit());
      const base = i * pointsPerStrand * 3;
      const vBase = i * vertsPerStrand;
      for (let k = 0; k <= SEGMENTS; k++) {
        const u = (k / SEGMENTS) * keep;
        const f = u * guideSteps;
        const g0 = Math.min(Math.floor(f), guideSteps - 1);
        p.lerpVectors(guide.pts[g0], guide.pts[g0 + 1], f - g0);
        // Strands stay apart at the root and gather toward the clump's tip.
        p.addScaledVector(rootOffset, 1 - (style.gather ?? 0.72) * u);
        p.addScaledVector(frizz, 0.0028 * style.frizz * u * u * stray * guideSteps * 0.2);
        this.pushOut(p, 0.0012);
        this.restOnBody(p);
        this.polyline.set([p.x, p.y, p.z], base + k * 3);
        if (k > 0) {
          q.copy(p).sub(prev);
          const segment = Math.max(q.length(), 1e-5);
          this.fullLength[i] += segment;
          this.segLen[i * SEGMENTS + k - 1] = segment;
          q.divideScalar(segment);
          dirs.set([q.x, q.y, q.z], (i * SEGMENTS + k - 1) * 3);
          if (k === SEGMENTS) this.lastLen[i] = segment;
        }
        prev.copy(p);
        for (const s of [0, 1]) {
          const v = vBase + k * 2 + s;
          uAttr[v] = k / SEGMENTS;
          sideAttr[v] = s ? 1 : -1;
          shadeAttr[v] = shade;
        }
      }
      const ii = i * SEGMENTS * 6;
      for (let k = 0; k < SEGMENTS; k++) {
        const a = vBase + k * 2;
        index.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], ii + k * 6);
      }
    }

  }

  // Change to another hairstyle: a fresh, uncut head of hair.
  setStyle(style) {
    this.style = style;
    this.grow(style, new Float32Array(this.lengths.length), new Float32Array(this.lengths.length), new Float32Array(this.lengths.length), new Uint32Array(STRANDS * SEGMENTS * 6));
    this.reset();
    this.grid.takeRest();
  }

  restOnBody(p) {
    if (p.z < -0.05) {
      const top = this.landing.height(p.x, p.y);
      if (p.z < top) p.z = top;
    }
  }

  pushOut(p, margin) {
    const c = this.center;
    const dx = p.x - c.x;
    const dy = p.y - c.y;
    const dz = p.z - c.z;
    const r = Math.hypot(dx, dy, dz) || 1e-6;
    const min = this.scalp.radius(dx / r, dy / r, dz / r) + margin;
    if (r < min) {
      const s = min / r;
      p.set(c.x + dx * s, c.y + dy * s, c.z + dz * s);
    }
  }

  setColour(hex) {
    this.material.uniforms.uColour.value.set(hex);
  }

  addToGrid(i, amount) {
    const b = i * (SEGMENTS + 1) * 3;
    const { pos, grid } = this;
    for (let k = 1; k <= this.lastNode[i]; k++) {
      const at = b + k * 3;
      const c = grid.index(pos[at], pos[at + 1], pos[at + 2]);
      if (c >= 0) grid.still[c] = Math.max(0, grid.still[c] + amount);
    }
  }

  random() {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) | 0;
    return (this.seed >>> 0) / 4294967296;
  }

  // Points past the last node collapse onto it, leaving a blunt cut end.
  writeStrand(i) {
    const { pos, positions, tangents, lengths } = this;
    const last = this.lastNode[i];
    const verts = (SEGMENTS + 1) * 2;
    const first = i * verts;
    const b = i * (SEGMENTS + 1) * 3;
    let at = first * 3;
    for (let k = 0; k <= SEGMENTS; k++) {
      const n = b + (k < last ? k : last) * 3;
      const a = b + (k < last ? k : last - 1) * 3;
      let tx = pos[a + 3] - pos[a];
      let ty = pos[a + 4] - pos[a + 1];
      let tz = pos[a + 5] - pos[a + 2];
      const r = 1 / (Math.sqrt(tx * tx + ty * ty + tz * tz) || 1e-6);
      tx *= r;
      ty *= r;
      tz *= r;
      for (let s = 0; s < 2; s++, at += 3) {
        positions[at] = pos[n];
        positions[at + 1] = pos[n + 1];
        positions[at + 2] = pos[n + 2];
        tangents[at] = tx;
        tangents[at + 1] = ty;
        tangents[at + 2] = tz;
      }
    }
    lengths.fill(this.length[i], first, first + verts);
    const band = this.bandOf[i];
    if (first < this.dirtyMin[band]) this.dirtyMin[band] = first;
    if (first + verts > this.dirtyMax[band]) this.dirtyMax[band] = first + verts;
  }

  // Send what changed to the GPU, once per drawn frame.
  flush() {
    for (let a = 0; a < this.activeCount; a++) this.writeStrand(this.active[a]);
    let any = false;
    for (let band = 0; band < HAIR_BANDS; band++) {
      if (this.dirtyMax[band] < 0) continue;
      const start = this.dirtyMin[band];
      const count = this.dirtyMax[band] - start;
      this.positionAttr.addUpdateRange(start * 3, count * 3);
      this.tangentAttr.addUpdateRange(start * 3, count * 3);
      if (this.lengthsDirty) this.lengthAttr.addUpdateRange(start, count);
      this.dirtyMin[band] = 2147483647;
      this.dirtyMax[band] = -1;
      any = true;
    }
    if (any) {
      this.positionAttr.needsUpdate = true;
      this.tangentAttr.needsUpdate = true;
      if (this.lengthsDirty) this.lengthAttr.needsUpdate = true;
      this.lengthsDirty = false;
    }
    this.pieces.flush();
  }

  // Strands stay asleep and untouched until part of the tool reaches them.
  wake(frame) {
    const inv = frame.inverse.elements;
    const w = frame.world.elements;
    const { pos, awake, active, lastNode, fullLength } = this;
    const margin = 0.006;
    // A sphere around the tool, for a quick first test on each root.
    const cx = w[12] + w[0] * 0.05 + w[8] * -0.05;
    const cy = w[13] + w[1] * 0.05 + w[9] * -0.05;
    const cz = w[14] + w[2] * 0.05 + w[10] * -0.05;
    const reach = 0.135;
    const { grid } = this;
    const { moving, still, mx, my, mz } = grid;
    const wakeSpeed2 = WAKE_CELL_SPEED * WAKE_CELL_SPEED;
    for (let i = 0; i < STRANDS; i++) {
      if (awake[i] || this.activeCount >= MAX_AWAKE) continue;
      const b = i * (SEGMENTS + 1) * 3;
      const dx = pos[b] - cx;
      const dy = pos[b + 1] - cy;
      const dz = pos[b + 2] - cz;
      const limit = reach + fullLength[i];
      if (dx * dx + dy * dy + dz * dz > limit * limit) continue;
      const last = lastNode[i];
      for (let k = 1; k <= last; k += k === last - 1 ? 1 : 2) {
        const at = b + k * 3;
        const X = pos[at];
        const Y = pos[at + 1];
        const Z = pos[at + 2];
        const x = inv[0] * X + inv[4] * Y + inv[8] * Z + inv[12];
        const y = inv[1] * X + inv[5] * Y + inv[9] * Z + inv[13];
        const z = inv[2] * X + inv[6] * Y + inv[10] * Z + inv[14];
        // Woken by the tool, or by hair moving past it.
        let woken = inToolZone(x, y, z, margin);
        if (!woken) {
          const c = grid.index(X, Y, Z);
          if (c >= 0 && moving[c] > 0) {
            const all = moving[c] + still[c];
            woken = (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / (all * all) > wakeSpeed2;
          }
        }
        if (!woken) continue;
        this.wakeStrand(i);
        break;
      }
    }
  }

  wakeStrand(i) {
    if (this.awake[i]) return;
    this.awake[i] = 1;
    this.quiet[i] = 0;
    this.active[this.activeCount++] = i;
    this.addToGrid(i, -1);
    const b = i * (SEGMENTS + 1) * 3;
    for (let k = 0; k <= SEGMENTS * 3 + 2; k++) this.prev[b + k] = this.pos[b + k];
  }

  // One physics step for every awake strand. Each is a chain of nodes from a fixed root:
  // it keeps its length, its curl (relative to the segment before it), feels gravity, and
  // can't pass through the scalp or the tool. Nothing else moves hair.
  step(frame, angle, pivotX, goingDown, sidewaysFast) {
    const { grid } = this;
    // Everything awake goes into the grid first, with how it's moving.
    grid.clearMoving();
    {
      const { pos, prev, active, lastNode } = this;
      const { moving, mx, my, mz, touched, lastNode: cellNode } = grid;
      for (let a = 0; a < this.activeCount; a++) {
        const i = active[a];
        const b = i * (SEGMENTS + 1) * 3;
        for (let k = 1; k <= lastNode[i]; k++) {
          const at = b + k * 3;
          const c = grid.index(pos[at], pos[at + 1], pos[at + 2]);
          if (c < 0) continue;
          if (moving[c] === 0) touched[grid.touchedCount++] = c;
          moving[c] += 1;
          mx[c] += pos[at] - prev[at];
          my[c] += pos[at + 1] - prev[at + 1];
          mz[c] += pos[at + 2] - prev[at + 2];
          cellNode[c] = i * (SEGMENTS + 1) + k;
        }
      }
    }
    if (this.stepCount++ % 2 === 0) this.wake(frame);
    const contactPoints = this.contactPoints;
    let contacts = 0;
    const { moving: gridMoving, still: gridStill, mx: gridX, my: gridY, mz: gridZ, lastNode: cellNode } = grid;
    const gn = grid.n;
    const gnn = gn * gn;
    const { hookNode, hookAt, hookSlip } = this;
    let forceX = 0;
    let forceY = 0;
    let forceZ = 0;
    let hooked = 0;
    const { pos, prev, restDir, segLen, lastNode, lastLen, stiff, awake, quiet, active } = this;
    const inv = frame.inverse.elements;
    const w = frame.world.elements;
    const scalp = this.scalp;
    const cX = this.center.x;
    const cY = this.center.y;
    const cZ = this.center.z;
    const maxR = scalp.maxRadius + 0.0012;
    const maxR2 = maxR * maxR;
    const tanA = Math.tan(angle);
    const pad2 = QUIET_MOVE * QUIET_MOVE;
    const { fullLength, length } = this;
    let kept = 0;
    for (let a = 0; a < this.activeCount; a++) {
      const i = active[a];
      const b = i * (SEGMENTS + 1) * 3;
      const last = lastNode[i];
      let px = pos[b];
      let py = pos[b + 1];
      let pz = pos[b + 2];
      // Rotation of the segment before this one from its resting direction (Rodrigues form).
      let vx = 0;
      let vy = 0;
      let vz = 0;
      let ic = 0.5;
      let moved = 0;
      let touching = false;
      let outboard = fullLength[i] * length[i]; // hair beyond this point, whose weight bends it
      // If this strand is caught on the tool, where that point is now, and how far along the hair.
      const rootX = px;
      const rootY = py;
      const rootZ = pz;
      let hk = hookNode[i];
      let hookX = 0;
      let hookY = 0;
      let hookZ = 0;
      let hookArc = 0;
      let arcSoFar = 0;
      if (hk > 0) {
        const h3 = i * 3;
        const hx = hookAt[h3];
        const hy = hookAt[h3 + 1];
        const hz = hookAt[h3 + 2];
        hookX = w[0] * hx + w[4] * hy + w[8] * hz + w[12];
        hookY = w[1] * hx + w[5] * hy + w[9] * hz + w[13];
        hookZ = w[2] * hx + w[6] * hy + w[10] * hz + w[14];
        for (let j = 0; j < hk; j++) hookArc += segLen[i * SEGMENTS + j];
        touching = true;
      }
      for (let k = 1; k <= last; k++) {
        const at = b + k * 3;
        const seg = i * SEGMENTS + k - 1;
        const r3 = seg * 3;
        const rx = restDir[r3];
        const ry = restDir[r3 + 1];
        const rz = restDir[r3 + 2];
        const L = k === last ? lastLen[i] : segLen[seg];

        // The direction this segment's curl asks for, given how its parent has turned. The
        // resting shape already includes gravity's bend, so only the change from it acts.
        const c1x = vy * rz - vz * ry;
        const c1y = vz * rx - vx * rz;
        const c1z = vx * ry - vy * rx;
        const sag = HAIR_SAG * outboard * outboard * L;
        outboard -= L;
        const ux = vy + vz * vx * ic;
        const uy = -vx + vz * vy * ic;
        const uz = -(vx * vx + vy * vy) * ic;
        const tx = rx + c1x + (vy * c1z - vz * c1y) * ic + sag * ux;
        const ty = ry + c1y + (vz * c1x - vx * c1z) * ic + sag * uy;
        const tz = rz + c1z + (vx * c1y - vy * c1x) * ic + sag * uz;

        const x0 = pos[at];
        const y0 = pos[at + 1];
        const z0 = pos[at + 2];
        let velX = x0 - prev[at];
        let velY = y0 - prev[at + 1];
        let velZ = z0 - prev[at + 2];
        // Other hair: friction with what's moving around it, and being pushed out of a crowd.
        let pushX = 0;
        let pushY = 0;
        let pushZ = 0;
        const cell = grid.index(x0, y0, z0);
        if (cell >= 0) {
          const all = gridMoving[cell] + gridStill[cell];
          if (all > 0) {
            const relX = gridX[cell] / all - velX;
            const relY = gridY[cell] / all - velY;
            const relZ = gridZ[cell] / all - velZ;
            velX += relX * HAIR_FRICTION;
            velY += relY * HAIR_FRICTION;
            velZ += relZ * HAIR_FRICTION;
            // A caught strand dragged through crowded hair is how knots start.
            const slip2 = relX * relX + relY * relY + relZ * relZ;
            if (hk > 0 && slip2 > KNOT_SPEED * KNOT_SPEED && all > 12 && this.knotCount < MAX_KNOTS && this.random() < 0.004) {
              const other = cellNode[cell];
              if (other !== 0 && Math.floor(other / (SEGMENTS + 1)) !== i) {
                const q = this.knotCount++;
                this.knotA[q] = i * (SEGMENTS + 1) + k;
                this.knotB[q] = other;
                const oa = other * 3;
                this.knotLen[q] = Math.max(0.001, Math.hypot(pos[oa] - x0, pos[oa + 1] - y0, pos[oa + 2] - z0));
                this.knotsMade++;
              }
            }
          }
          pushX = (grid.over(cell - gnn) - grid.over(cell + gnn)) * HAIR_PUSH;
          pushY = (grid.over(cell - gn) - grid.over(cell + gn)) * HAIR_PUSH;
          pushZ = (grid.over(cell - 1) - grid.over(cell + 1)) * HAIR_PUSH;
          pushX = pushX > HAIR_PUSH_MAX ? HAIR_PUSH_MAX : pushX < -HAIR_PUSH_MAX ? -HAIR_PUSH_MAX : pushX;
          pushY = pushY > HAIR_PUSH_MAX ? HAIR_PUSH_MAX : pushY < -HAIR_PUSH_MAX ? -HAIR_PUSH_MAX : pushY;
          pushZ = pushZ > HAIR_PUSH_MAX ? HAIR_PUSH_MAX : pushZ < -HAIR_PUSH_MAX ? -HAIR_PUSH_MAX : pushZ;
        }
        let nx = x0 + velX * STRAND_DAMPING + pushX;
        let ny = y0 + velY * STRAND_DAMPING + pushY;
        let nz = z0 + velZ * STRAND_DAMPING + pushZ;
        prev[at] = x0;
        prev[at + 1] = y0;
        prev[at + 2] = z0;

        const S = stiff[k];
        let dx = nx - px;
        let dy = ny - py;
        let dz = nz - pz;
        dx += S * (tx * L - dx);
        dy += S * (ty * L - dy);
        dz += S * (tz * L - dz);
        let scale = L / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9);
        nx = px + dx * scale;
        ny = py + dy * scale;
        nz = pz + dz * scale;

        // A strand caught on the tool: the caught point goes with it, as far as the hair reaches,
        // and the hair before it straightens toward it.
        arcSoFar += L;
        const isHook = hk > 0 && k === hk;
        if (hk > 0 && k <= hk) {
          let tx2 = hookX;
          let ty2 = hookY;
          let tz2 = hookZ;
          let share = 1;
          if (!isHook) {
            const f = arcSoFar / hookArc;
            tx2 = rootX + (hookX - rootX) * f;
            ty2 = rootY + (hookY - rootY) * f;
            tz2 = rootZ + (hookZ - rootZ) * f;
            share = 0.25;
          }
          dx = nx + (tx2 - nx) * share - px;
          dy = ny + (ty2 - ny) * share - py;
          dz = nz + (tz2 - nz) * share - pz;
          scale = L / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9);
          nx = px + dx * scale;
          ny = py + dy * scale;
          nz = pz + dz * scale;
        }

        let pressed = false;
        // The tool. Coordinates in its frame: x along the blades, y travel, z toward the scalp.
        const lx = inv[0] * nx + inv[4] * ny + inv[8] * nz + inv[12];
        {
          const ly = inv[1] * nx + inv[5] * ny + inv[9] * nz + inv[13];
          const lz = inv[2] * nx + inv[6] * ny + inv[10] * nz + inv[14];
          if (!isHook && inToolZone(lx, ly, lz, 0)) {
            touching = true;
            let py2 = 0;
            let pz2 = 0;
            let plx = 0;
            // Comb: going down, hair passes between the tines. Otherwise a point just under
            // the top rests on it and gets carried up; one well below stays free. On the comb
            // it settles into the nearest gap between tines. Moved sideways, a tine catches it.
            const onFan = lx > COMB_X0 && lx < COMB_X1;
            const half = onFan ? combHalfWidth(lx) : 0;
            if (onFan && Math.abs(ly) < half) {
              const d = lz - COMB_TOP;
              if (!goingDown && d > 0 && d < CONTACT_BAND) {
                pz2 -= d;
                const f = ly / half;
                const slot = (Math.floor(((f + 1) / 2) * COMB_SLOTS) + 0.5) / COMB_SLOTS;
                py2 += ((slot * 2 - 1) * half - ly) * 0.2;
                if (contacts < MAX_CONTACT_POINTS) {
                  contactPoints[contacts * 3] = nx;
                  contactPoints[contacts * 3 + 1] = ny;
                  contactPoints[contacts * 3 + 2] = nz;
                }
                contacts++;
              }
              if (sidewaysFast && k >= 2 && hk < 0 && d > -0.0015 && d < CONTACT_BAND && this.random() < 0.03) {
                hk = hookNode[i] = k;
                hookAt[i * 3] = lx;
                hookAt[i * 3 + 1] = ly;
                hookAt[i * 3 + 2] = COMB_TOP;
                hookSlip[i] = HOOK_SLIP_MIN + HOOK_SLIP_RANGE * this.random();
              }
            }
            // Each blade is a thin plate: a point near it goes back out the side it's nearer.
            const xp = lx - pivotX;
            const nearPlate = lz - PLANE_Z;
            if (xp > 0 && xp < BLADE_LEN && nearPlate > -0.0015 && nearPlate < 0.0015) {
              const edge = xp * tanA - EDGE_OVERLAP;
              const off = ly < 0 ? -ly : ly;
              if (off >= edge && off < edge + 0.008) {
                pz2 += nearPlate >= 0 ? 0.0015 - nearPlate : -0.0015 - nearPlate;
                if (sidewaysFast && k >= 2 && hk < 0 && this.random() < 0.015) {
                  hk = hookNode[i] = k;
                  hookAt[i * 3] = lx;
                  hookAt[i * 3 + 1] = ly;
                  hookAt[i * 3 + 2] = PLANE_Z;
                  hookSlip[i] = HOOK_SLIP_MIN + HOOK_SLIP_RANGE * this.random();
                }
              }
            }
            // Solid boxes: out by the shortest way.
            for (const box of TOOL_BOXES) {
              if (lx > box[0] && lx < box[1] && ly > box[2] && ly < box[3] && lz > box[4] && lz < box[5]) {
                const e0 = lx - box[0];
                const e1 = box[1] - lx;
                const e2 = ly - box[2];
                const e3 = box[3] - ly;
                const e4 = lz - box[4];
                const e5 = box[5] - lz;
                const least = Math.min(e0, e1, e2, e3, e4, e5);
                if (least === e0) plx -= e0;
                else if (least === e1) plx += e1;
                else if (least === e2) py2 -= e2;
                else if (least === e3) py2 += e3;
                else if (least === e4) pz2 -= e4;
                else pz2 += e5;
              }
            }
            if (plx !== 0 || py2 !== 0 || pz2 !== 0) {
              const wx = w[0] * plx + w[4] * py2 + w[8] * pz2;
              const wy = w[1] * plx + w[5] * py2 + w[9] * pz2;
              const wz = w[2] * plx + w[6] * py2 + w[10] * pz2;
              nx += wx;
              ny += wy;
              nz += wz;
              // The push bends the strand along its length, not just at this one point.
              for (let j = 1, share = 0.5; j <= 3 && k - j >= 1; j++, share *= 0.5) {
                const up = at - j * 3;
                pos[up] += wx * share;
                pos[up + 1] += wy * share;
                pos[up + 2] += wz * share;
              }
              // A push doesn't launch the hair: it just gets out of the way.
              pressed = true;
              // Hair doesn't stretch: if it's taut, this drags the point back under the
              // comb, which is the strand sliding through between the tines.
              dx = nx - px;
              dy = ny - py;
              dz = nz - pz;
              scale = L / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9);
              nx = px + dx * scale;
              ny = py + dy * scale;
              nz = pz + dz * scale;
            }
          }
        }

        // The scalp.
        const ox = nx - cX;
        const oy = ny - cY;
        const oz = nz - cZ;
        const r2 = ox * ox + oy * oy + oz * oz;
        if (r2 < maxR2) {
          const r = Math.sqrt(r2) || 1e-6;
          const least = scalp.fastRadius(ox / r, oy / r, oz / r) + 0.0012;
          if (r < least) {
            const out = least / r;
            nx = cX + ox * out;
            ny = cY + oy * out;
            nz = cZ + oz * out;
          }
        }

        // The cape and chair, for hair long enough to reach them.
        if (nz < -0.05) {
          const top = this.landing.height(nx, ny);
          if (nz < top) nz = top;
        }

        pos[at] = nx;
        pos[at + 1] = ny;
        pos[at + 2] = nz;
        if (pressed) {
          prev[at] = nx;
          prev[at + 1] = ny;
          prev[at + 2] = nz;
        }
        const mx = nx - x0;
        const my = ny - y0;
        const mz = nz - z0;
        const m2 = mx * mx + my * my + mz * mz;
        if (m2 > moved) moved = m2;

        const dl = 1 / (Math.sqrt((nx - px) * (nx - px) + (ny - py) * (ny - py) + (nz - pz) * (nz - pz)) || 1e-9);
        const ex = (nx - px) * dl;
        const ey = (ny - py) * dl;
        const ez = (nz - pz) * dl;
        vx = ry * ez - rz * ey;
        vy = rz * ex - rx * ez;
        vz = rx * ey - ry * ex;
        ic = 1 / Math.max(0.02, 1 + rx * ex + ry * ey + rz * ez);
        px = nx;
        py = ny;
        pz = nz;
      }
      // Tension in a caught strand: hair stretches only a little, so once it's taut the point
      // slides along the hair, or the hair snaps.
      if (hk > 0 && hookArc > 0) {
        hooked++;
        const hx = hookX - rootX;
        const hy = hookY - rootY;
        const hz = hookZ - rootZ;
        const reach2 = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1e-6;
        const strain = (reach2 - hookArc) / hookArc;
        if (strain > 0) {
          const tension = strain * HAIR_EA;
          const pull = (Math.min(tension, HAIR_BREAK) * HAIRS_PER_STRAND) / reach2;
          forceX -= hx * pull;
          forceY -= hy * pull;
          forceZ -= hz * pull;
          if (tension >= HAIR_BREAK) {
            this.snapStrand(i, hk);
            hookNode[i] = -1;
          } else if (tension > hookSlip[i]) {
            this.slid++;
            hookNode[i] = hk + 1 >= lastNode[i] ? -1 : hk + 1;
          }
          hookSlip[i] *= 0.985; // it creeps under load
        }
      }
      if (!touching && moved < pad2) {
        if (++quiet[i] > QUIET_STEPS) {
          awake[i] = 0;
          this.addToGrid(i, 1);
          this.writeStrand(i);
          continue;
        }
      } else quiet[i] = 0;
      active[kept++] = i;
    }
    this.activeCount = kept;
    this.contactCount = contacts;
    this.hookedCount = hooked;
    this.toolForce.set(forceX, forceY, forceZ);
    this.toolLoad = Math.sqrt(forceX * forceX + forceY * forceY + forceZ * forceZ);
    this.solveKnots();
    this.pieces.step(frame, angle, pivotX);
  }

  // The hair broke at node k: everything past it becomes a clipping.
  snapStrand(i, k) {
    const b = i * (SEGMENTS + 1) * 3;
    const last = this.lastNode[i];
    if (k >= last || k < 1) return;
    this.pieces.spawn(this.pos, this.prev, b, k - 1, 1, last);
    this.lastNode[i] = k;
    this.lastLen[i] = this.segLen[i * SEGMENTS + k - 1];
    if (this.length[i] === 1) this.cutCount++;
    this.length[i] = k / SEGMENTS;
    this.lengthsDirty = true;
    this.broken++;
  }

  // A knot holds two strands together at a point. It tightens as they pull, and comes apart
  // if pulled hard enough.
  solveKnots() {
    const { pos, knotA, knotB, knotLen, lastNode, awake } = this;
    const per = SEGMENTS + 1;
    let kept = 0;
    for (let q = 0; q < this.knotCount; q++) {
      const a = knotA[q];
      const b = knotB[q];
      const sa = Math.floor(a / per);
      const sb = Math.floor(b / per);
      const na = a - sa * per;
      const nb = b - sb * per;
      if (na > lastNode[sa] || nb > lastNode[sb]) continue; // a cut took it off
      let len = knotLen[q];
      if (awake[sa] || awake[sb]) {
        const pa = a * 3;
        const pb = b * 3;
        const dx = pos[pb] - pos[pa];
        const dy = pos[pb + 1] - pos[pa + 1];
        const dz = pos[pb + 2] - pos[pa + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > len) {
          if ((d - len) * KNOT_STIFFNESS > KNOT_SLIP) {
            this.knotsSlipped++;
            continue;
          }
          if (d - len > 0.001) {
            this.wakeStrand(sa);
            this.wakeStrand(sb);
          }
          const f = (0.5 * (d - len)) / d;
          pos[pa] += dx * f;
          pos[pa + 1] += dy * f;
          pos[pa + 2] += dz * f;
          pos[pb] -= dx * f;
          pos[pb + 1] -= dy * f;
          pos[pb + 2] -= dz * f;
        }
      }
      knotA[kept] = a;
      knotB[kept] = b;
      knotLen[kept] = len;
      kept++;
    }
    this.knotCount = kept;
  }

  // For the readout: how many awake strands reach the blade plane, and how many are in the jaws.
  bladeStats(frame, angle, pivotX) {
    const e = frame.inverse.elements;
    const tan = Math.tan(angle);
    const { pos, active, lastNode } = this;
    let rising = 0;
    let inJaws = 0;
    for (let a = 0; a < this.activeCount; a++) {
      const i = active[a];
      const b = i * (SEGMENTS + 1) * 3;
      let z0 = e[2] * pos[b] + e[6] * pos[b + 1] + e[10] * pos[b + 2] + e[14];
      for (let k = 0; k < lastNode[i]; k++) {
        const at = b + (k + 1) * 3;
        const z1 = e[2] * pos[at] + e[6] * pos[at + 1] + e[10] * pos[at + 2] + e[14];
        if (z0 > PLANE_Z && z1 <= PLANE_Z) {
          rising++;
          const t = (z0 - PLANE_Z) / (z0 - z1);
          const X = pos[at - 3] + (pos[at] - pos[at - 3]) * t;
          const Y = pos[at - 2] + (pos[at + 1] - pos[at - 2]) * t;
          const Z = pos[at - 1] + (pos[at + 2] - pos[at - 1]) * t;
          const xp = e[0] * X + e[4] * Y + e[8] * Z + e[12] - pivotX;
          const y = e[1] * X + e[5] * Y + e[9] * Z + e[13];
          if (xp > 0 && xp < BLADE_LEN && Math.abs(y) < xp * tan - EDGE_OVERLAP) inJaws++;
          break;
        }
        z0 = z1;
      }
    }
    return { rising, inJaws };
  }

  // The blades closed from half-angle `before` to `angle` this step. A strand is cut where it
  // rises through the blade plane or lies on the comb, if a closing edge swept across that point.
  shear(frame, angle, before, pivotX) {
    const e = frame.inverse.elements;
    const tanNow = Math.tan(angle);
    const tanBefore = Math.tan(before);
    const { pos, prev, active, lastNode, lastLen, segLen } = this;
    let changed = 0;
    let bladeForce = 0;
    for (let a = 0; a < this.activeCount; a++) {
      const i = active[a];
      const b = i * (SEGMENTS + 1) * 3;
      const last = lastNode[i];
      let z0 = e[2] * pos[b] + e[6] * pos[b + 1] + e[10] * pos[b + 2] + e[14];
      for (let k = 0; k < last; k++) {
        const at = b + k * 3;
        const z1 = e[2] * pos[at + 3] + e[6] * pos[at + 4] + e[10] * pos[at + 5] + e[14];
        // Rising through the blade plane, or lying on the comb, which the lower blade rides on.
        const rising = z0 > PLANE_Z && z1 <= PLANE_Z;
        const onComb = !rising && k + 1 < last && z1 > PLANE_Z && z1 < COMB_TOP + 0.001;
        if (rising || onComb) {
          const t = rising ? (z0 - PLANE_Z) / (z0 - z1) : 1;
          const X = pos[at] + (pos[at + 3] - pos[at]) * t;
          const Y = pos[at + 1] + (pos[at + 4] - pos[at + 1]) * t;
          const Z = pos[at + 2] + (pos[at + 5] - pos[at + 2]) * t;
          const xp = e[0] * X + e[4] * Y + e[8] * Z + e[12] - pivotX;
          if (xp > 0 && xp < BLADE_LEN) {
            const y = e[1] * X + e[5] * Y + e[9] * Z + e[13];
            const off = y < 0 ? -y : y;
            if (off < xp * tanBefore - EDGE_OVERLAP + 0.0005 && off >= xp * tanNow - EDGE_OVERLAP) {
              this.pieces.spawn(pos, prev, b, k, t, last);
              const seg = i * SEGMENTS + k;
              const full = k + 1 === last ? lastLen[i] : segLen[seg];
              const kept = Math.max(full * t, 1e-4);
              lastLen[i] = kept;
              const node = at + 3;
              pos[node] = X;
              pos[node + 1] = Y;
              pos[node + 2] = Z;
              prev[node] = X;
              prev[node + 1] = Y;
              prev[node + 2] = Z;
              lastNode[i] = k + 1;
              if (this.length[i] === 1) this.cutCount++;
              this.cutTotal++;
              this.length[i] = (k + kept / segLen[seg]) / SEGMENTS;
              this.lengthsDirty = true;
              this.writeStrand(i);
              changed++;
              // What this hair asks of the fingers: its cutting force, levered about the pivot.
              bladeForce += CUT_FORCE * HAIRS_PER_STRAND * (xp / HANDLE_LEN);
              break;
            }
          }
        }
        z0 = z1;
      }
    }
    this.bladeForce = bladeForce;
    return changed;
  }

  // Back to a full head of hair, and sweep up the floor.
  reset() {
    this.pos.set(this.polyline);
    this.prev.set(this.polyline);
    this.length.fill(1);
    this.lastNode.fill(SEGMENTS);
    for (let i = 0; i < STRANDS; i++) {
      this.lastLen[i] = this.segLen[i * SEGMENTS + SEGMENTS - 1];
      this.writeStrand(i);
    }
    this.awake.fill(0);
    this.activeCount = 0;
    this.cutCount = 0;
    this.contactCount = 0;
    this.hookNode.fill(-1);
    this.hookedCount = 0;
    this.knotCount = 0;
    this.knotsMade = this.knotsSlipped = this.slid = this.broken = 0;
    this.toolForce.set(0, 0, 0);
    this.toolLoad = 0;
    this.grid.clearMoving();
    this.grid.still.fill(0);
    for (let i = 0; i < STRANDS; i++) this.addToGrid(i, 1);
    this.dirtyMin.fill(2147483647);
    this.dirtyMax.fill(-1);
    for (const attr of [this.positionAttr, this.tangentAttr, this.lengthAttr]) {
      attr.clearUpdateRanges();
      attr.needsUpdate = true; // no ranges: the whole buffer goes up
    }
    this.pieces.reset();
  }

  percentCut() {
    return (this.cutCount / STRANDS) * 100;
  }
}

// Camera-facing ribbons lit with a hair model: light depends on the strand's direction
// rather than a surface normal, which gives hair its streaky highlights.
const HAIR_VERTEX = /* glsl */ `
  attribute vec3 aTangent;
  attribute float aU;
  attribute float aSide;
  attribute float aShade;
  attribute float aLength;
  uniform float uWidth;
  varying vec3 vTangent;
  varying vec3 vView;
  varying float vU;
  varying float vSide;
  varying float vShade;
  varying float vDepth;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vec3 view = normalize(cameraPosition - world.xyz);
    vec3 side = normalize(cross(aTangent, view) + vec3(1e-5));
    float width = uWidth * (1.0 - 0.72 * aU * aU) * max(1.0, length(cameraPosition - world.xyz) * 1.4);
    world.xyz += side * aSide * width;
    vTangent = aTangent;
    vView = view;
    vU = aU;
    vSide = aSide;
    vShade = aShade;
    vDepth = aU / max(aLength, 0.001);
    gl_Position = projectionMatrix * viewMatrix * world;
  }`;

const HAIR_FRAGMENT = /* glsl */ `
  uniform vec3 uColour;
  uniform vec3 uKeyDir;
  uniform vec3 uKeyColour;
  varying vec3 vTangent;
  varying vec3 vView;
  varying float vU;
  varying float vSide;
  varying float vShade;
  varying float vDepth;
  float strandSpec(vec3 t, vec3 h, float power) {
    float d = dot(t, h);
    return pow(max(0.0, 1.0 - d * d), power * 0.5);
  }
  void main() {
    vec3 t = normalize(vTangent);
    vec3 v = normalize(vView);
    vec3 l = uKeyDir;
    vec3 h = normalize(l + v);
    float tl = dot(t, l);
    float wrap = sqrt(max(0.0, 1.0 - tl * tl));
    // Roots sit in the shade of the hair above them.
    float occlusion = mix(0.42, 1.0, smoothstep(0.0, 0.6, vDepth));
    vec3 base = uColour * vShade;
    vec3 diffuse = base * (0.32 + 0.68 * wrap) * uKeyColour * 0.34 * occlusion;
    vec3 ambient = base * vec3(0.46, 0.5, 0.46) * occlusion;
    // A sharp white highlight and a broader one tinted by the hair colour.
    vec3 shifted = normalize(t + 0.12 * cross(v, t));
    vec3 spec = vec3(0.09) * strandSpec(t, h, 110.0)
      + (base + 0.08) * 0.32 * strandSpec(shifted, h, 26.0);
    vec3 colour = diffuse + ambient + spec * uKeyColour * 0.34 * occlusion;
    gl_FragColor = vec4(colour, 1.0 - smoothstep(0.55, 1.0, abs(vSide)) * 0.75);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

// Cut hair: short free chains that carry on from how the strand was moving, fall through
// the air, slide off the head, and stay where they land.
class Pieces {
  constructor(scalp, material, grid) {
    this.scalp = scalp;
    this.grid = grid;
    this.landing = null; // set once the person and chair exist
    this.center = scalp.center;
    const nodes = MAX_PIECES * PIECE_NODES;
    this.pos = new Float32Array(nodes * 3);
    this.prev = new Float32Array(nodes * 3);
    this.segment = new Float32Array(MAX_PIECES);
    this.still = new Uint8Array(MAX_PIECES);
    this.live = [];
    this.next = 0;
    this.used = 0;
    this.positions = new Float32Array(nodes * 2 * 3);
    this.tangents = new Float32Array(nodes * 2 * 3);
    const u = new Float32Array(nodes * 2);
    const side = new Float32Array(nodes * 2);
    const shade = new Float32Array(nodes * 2);
    const index = new Uint32Array(MAX_PIECES * (PIECE_NODES - 1) * 6);
    const rand = mulberry(29);
    for (let p = 0, at = 0, ii = 0; p < MAX_PIECES; p++) {
      const tone = 0.72 + rand() * 0.42;
      const base = at;
      for (let k = 0; k < PIECE_NODES; k++) {
        for (let s = 0; s < 2; s++, at++) {
          u[at] = 0.55 + 0.45 * (k / (PIECE_NODES - 1)); // same thin, tapered look as the tips
          side[at] = s ? 1 : -1;
          shade[at] = tone;
        }
      }
      for (let k = 0; k < PIECE_NODES - 1; k++, ii += 6) {
        const v = base + k * 2;
        index.set([v, v + 1, v + 2, v + 1, v + 3, v + 2], ii);
      }
    }
    const geometry = new THREE.BufferGeometry();
    this.positionAttr = new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage);
    this.tangentAttr = new THREE.BufferAttribute(this.tangents, 3).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", this.positionAttr);
    geometry.setAttribute("aTangent", this.tangentAttr);
    geometry.setAttribute("aU", new THREE.BufferAttribute(u, 1));
    geometry.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
    geometry.setAttribute("aShade", new THREE.BufferAttribute(shade, 1));
    geometry.setAttribute("aLength", new THREE.BufferAttribute(new Float32Array(nodes * 2).fill(1), 1));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.dirtyMin = Infinity;
    this.dirtyMax = -1;
  }

  // Part of a strand from t along segment k out to its last node, resampled along its length.
  spawn(pos, prev, b, k, t, last) {
    const count = last - k + 1;
    const pts = new Array(count);
    const start = b + k * 3;
    pts[0] = [
      pos[start] + (pos[start + 3] - pos[start]) * t,
      pos[start + 1] + (pos[start + 4] - pos[start + 1]) * t,
      pos[start + 2] + (pos[start + 5] - pos[start + 2]) * t,
      prev[start] + (prev[start + 3] - prev[start]) * t,
      prev[start + 1] + (prev[start + 4] - prev[start + 1]) * t,
      prev[start + 2] + (prev[start + 5] - prev[start + 2]) * t,
    ];
    let total = 0;
    const along = new Array(count).fill(0);
    for (let j = 1; j < count; j++) {
      const at = start + j * 3;
      pts[j] = [pos[at], pos[at + 1], pos[at + 2], prev[at], prev[at + 1], prev[at + 2]];
      total += Math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1], pts[j][2] - pts[j - 1][2]);
      along[j] = total;
    }
    if (total < 0.0015) return; // too small to see
    const slot = this.next;
    this.next = (this.next + 1) % MAX_PIECES;
    this.used = Math.max(this.used, slot + 1);
    this.segment[slot] = total / (PIECE_NODES - 1);
    this.still[slot] = 0;
    const base = slot * PIECE_NODES * 3;
    let j = 0;
    for (let n = 0; n < PIECE_NODES; n++) {
      const want = (n / (PIECE_NODES - 1)) * total;
      while (j < count - 2 && along[j + 1] < want) j++;
      const w = Math.min(1, (want - along[j]) / (along[j + 1] - along[j] || 1));
      for (let c = 0; c < 3; c++) {
        this.pos[base + n * 3 + c] = pts[j][c] + (pts[j + 1][c] - pts[j][c]) * w;
        this.prev[base + n * 3 + c] = pts[j][c + 3] + (pts[j + 1][c + 3] - pts[j][c + 3]) * w;
      }
    }
    this.live.push(slot);
  }

  step(frame, angle, pivotX) {
    if (!this.live.length) return;
    const { pos, prev, segment, still, scalp, grid, landing } = this;
    const inv = frame.inverse.elements;
    const w = frame.world.elements;
    const tanA = Math.tan(angle);
    const cX = this.center.x;
    const cY = this.center.y;
    const cZ = this.center.z;
    const maxR = scalp.maxRadius + 0.0015;
    let kept = 0;
    for (let a = 0; a < this.live.length; a++) {
      const slot = this.live[a];
      const base = slot * PIECE_NODES * 3;
      let moved = 0;
      for (let n = 0; n < PIECE_NODES; n++) {
        const at = base + n * 3;
        const x = pos[at];
        const y = pos[at + 1];
        const z = pos[at + 2];
        // Inside hair a clipping is slowed a lot, so it can catch there instead of falling through.
        const c = grid.index(x, y, z);
        const crowd = c >= 0 ? grid.still[c] + grid.moving[c] : 0;
        const drag = PIECE_DAMPING * (crowd > 40 ? 0.25 : 1 - crowd * 0.019);
        const fall = crowd > 40 ? 0 : GRAVITY_STEP;
        pos[at] = x + (x - prev[at]) * drag;
        pos[at + 1] = y + (y - prev[at + 1]) * drag;
        pos[at + 2] = z + (z - prev[at + 2]) * drag - fall;
        prev[at] = x;
        prev[at + 1] = y;
        prev[at + 2] = z;
      }
      // Keep its length.
      const L = segment[slot];
      for (let pass = 0; pass < 2; pass++) {
        for (let n = 0; n < PIECE_NODES - 1; n++) {
          const at = base + n * 3;
          const dx = pos[at + 3] - pos[at];
          const dy = pos[at + 4] - pos[at + 1];
          const dz = pos[at + 5] - pos[at + 2];
          const f = (0.5 * (Math.sqrt(dx * dx + dy * dy + dz * dz) - L)) / (Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9);
          pos[at] += dx * f;
          pos[at + 1] += dy * f;
          pos[at + 2] += dz * f;
          pos[at + 3] -= dx * f;
          pos[at + 4] -= dy * f;
          pos[at + 5] -= dz * f;
        }
      }
      for (let n = 0; n < PIECE_NODES; n++) {
        const at = base + n * 3;
        const ox = pos[at] - cX;
        const oy = pos[at + 1] - cY;
        const oz = pos[at + 2] - cZ;
        const r = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1e-6;
        if (r < maxR) {
          const least = scalp.fastRadius(ox / r, oy / r, oz / r) + 0.0015;
          if (r < least) {
            const out = least / r;
            pos[at] = cX + ox * out;
            pos[at + 1] = cY + oy * out;
            pos[at + 2] = cZ + oz * out;
            // Rubbing on the scalp slows it.
            prev[at] += (pos[at] - prev[at]) * 0.5;
            prev[at + 1] += (pos[at + 1] - prev[at + 1]) * 0.5;
            prev[at + 2] += (pos[at + 2] - prev[at + 2]) * 0.5;
          }
        }
        // It can land on the comb or a blade, then slide off when the tool tilts or moves.
        {
          const X = pos[at];
          const Y = pos[at + 1];
          const Z = pos[at + 2];
          const lx = inv[0] * X + inv[4] * Y + inv[8] * Z + inv[12];
          if (lx > 0.04 && lx < COMB_SPINE_END) {
            const ly = inv[1] * X + inv[5] * Y + inv[9] * Z + inv[13];
            const lz = inv[2] * X + inv[6] * Y + inv[10] * Z + inv[14];
            const off = ly < 0 ? -ly : ly;
            const inJaw = lx - pivotX < BLADE_LEN && off < (lx - pivotX) * tanA + 0.008;
            if (lz > PLANE_Z - 0.0005 && lz < PLANE_Z + 0.006 && (inJaw || (lx > COMB_X0 && lx < COMB_X1 && off < combHalfWidth(lx)))) {
              const up = lz - (PLANE_Z - 0.0005);
              pos[at] -= w[8] * up;
              pos[at + 1] -= w[9] * up;
              pos[at + 2] -= w[10] * up;
              prev[at + 2] = pos[at + 2];
            }
          }
        }
        // The real surfaces below: cape, chair, legs, floor. It slides down anything steep.
        const rest = landing ? landing.height(pos[at], pos[at + 1]) : FLOOR_Z + 0.001;
        if (pos[at + 2] < rest) {
          pos[at + 2] = rest;
          prev[at + 2] = rest;
          prev[at] += (pos[at] - prev[at]) * 0.45;
          prev[at + 1] += (pos[at + 1] - prev[at + 1]) * 0.45;
          if (landing) {
            const gx = landing.slopeX(pos[at], pos[at + 1]);
            const gy = landing.slopeY(pos[at], pos[at + 1]);
            if (gx * gx + gy * gy > 0.5) {
              pos[at] -= Math.max(-1, Math.min(1, gx)) * GRAVITY_STEP * 0.5;
              pos[at + 1] -= Math.max(-1, Math.min(1, gy)) * GRAVITY_STEP * 0.5;
            }
          }
        }
        const mx = pos[at] - prev[at];
        const my = pos[at + 1] - prev[at + 1];
        const mz = pos[at + 2] - prev[at + 2];
        const m2 = mx * mx + my * my + mz * mz;
        if (m2 > moved) moved = m2;
      }
      this.write(slot);
      if (moved < 1e-8) {
        if (++still[slot] > 10) continue; // come to rest; it stays where it is
      } else still[slot] = 0;
      this.live[kept++] = slot;
    }
    this.live.length = kept;
  }

  write(slot) {
    const { pos, positions, tangents } = this;
    const base = slot * PIECE_NODES * 3;
    let at = slot * PIECE_NODES * 2 * 3;
    for (let n = 0; n < PIECE_NODES; n++) {
      const node = base + n * 3;
      const a = base + (n < PIECE_NODES - 1 ? n : n - 1) * 3;
      let tx = pos[a + 3] - pos[a];
      let ty = pos[a + 4] - pos[a + 1];
      let tz = pos[a + 5] - pos[a + 2];
      const r = 1 / (Math.sqrt(tx * tx + ty * ty + tz * tz) || 1e-6);
      tx *= r;
      ty *= r;
      tz *= r;
      for (let s = 0; s < 2; s++, at += 3) {
        positions[at] = pos[node];
        positions[at + 1] = pos[node + 1];
        positions[at + 2] = pos[node + 2];
        tangents[at] = tx;
        tangents[at + 1] = ty;
        tangents[at + 2] = tz;
      }
    }
    if (slot < this.dirtyMin) this.dirtyMin = slot;
    if (slot > this.dirtyMax) this.dirtyMax = slot;
  }

  flush() {
    if (this.dirtyMax < 0) return;
    const per = PIECE_NODES * 2 * 3;
    const start = this.dirtyMin * per;
    const count = (this.dirtyMax - this.dirtyMin + 1) * per;
    this.positionAttr.addUpdateRange(start, count);
    this.tangentAttr.addUpdateRange(start, count);
    this.positionAttr.needsUpdate = true;
    this.tangentAttr.needsUpdate = true;
    this.mesh.geometry.setDrawRange(0, this.used * (PIECE_NODES - 1) * 6);
    this.dirtyMin = Infinity;
    this.dirtyMax = -1;
  }

  reset() {
    this.live.length = 0;
    this.next = 0;
    this.used = 0;
    this.mesh.geometry.setDrawRange(0, 0);
  }
}

function mulberry(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- driving by hand -------------------------------------------------------
// Inverse kinematics for when the visitor drives: damped least squares on the arm's own joints,
// holding the same tool attitude the planned moves use, and keeping the tool off the scalp.
class ArmIK {
  constructor(robot, kinematics, scalp) {
    this.robot = robot;
    this.scalp = scalp;
    this.joints = [];
    for (let i = 1; i <= 7; i++) {
      const j = robot.joints.get(`joint${i}`);
      const spec = kinematics.joints.find((k) => k.name === `joint${i}`);
      this.joints.push({ ...j, origin: j.child.parent, lower: spec.lower, upper: spec.upper, velocity: spec.velocity });
    }
    this.p = new THREE.Vector3();
    this.a = new THREE.Vector3();
    this.tip = new THREE.Vector3();
    this.toolPoints = [];
    for (const x of [COMB_X0, 0.09, COMB_X1]) {
      const h = combHalfWidth(x);
      this.toolPoints.push([x, h, COMB_TOP + 0.0025], [x, 0, COMB_TOP + 0.0025], [x, -h, COMB_TOP + 0.0025]);
    }
    this.toolPoints.push([0, 0, 0], [0, 0.03, 0], [0, -0.03, 0], [0.05, 0, PLANE_Z]);
  }

  // The tool's axes for a blade position: same 60% tilt toward the scalp's normal as the plans.
  frameFor(position, out) {
    const c = this.scalp.center;
    const n = this.p.copy(position).sub(c).normalize();
    const up = new THREE.Vector3(0, 0, 1);
    const axis = new THREE.Vector3().crossVectors(up, n);
    const sn = axis.length();
    const z = new THREE.Vector3();
    if (sn < 1e-6) z.copy(n);
    else {
      axis.divideScalar(sn);
      z.copy(up).applyAxisAngle(axis, 0.6 * Math.atan2(sn, up.dot(n)));
    }
    z.negate();
    const x = new THREE.Vector3(1, 0, 0).addScaledVector(z, -z.x);
    if (x.lengthSq() < 1e-4) x.set(0, 1, 0).addScaledVector(z, -z.y);
    x.normalize();
    const y = new THREE.Vector3().crossVectors(z, x);
    return out.makeBasis(x, y, z);
  }

  // Lift a blade position until no sampled part of the tool is within 3 mm of the scalp.
  keepOffHead(position) {
    const m = new THREE.Matrix4();
    const c = this.scalp.center;
    const v = new THREE.Vector3();
    for (let pass = 0; pass < 6; pass++) {
      this.frameFor(position, m);
      let worst = 0;
      for (const q of this.toolPoints) {
        v.set(q[0] - BLADE_CENTER.x, q[1] - BLADE_CENTER.y, q[2] - BLADE_CENTER.z).applyMatrix4(m).add(position).sub(c);
        const r = v.length() || 1e-6;
        const short = this.scalp.fastRadius(v.x / r, v.y / r, v.z / r) + 0.003 - r;
        if (short > worst) worst = short;
      }
      if (worst <= 0) return;
      v.copy(position).sub(c).normalize();
      position.addScaledVector(v, worst + 0.0005);
    }
  }

  solve(player, goal) {
    this.keepOffHead(goal);
    const robot = this.robot;
    robot.root.updateMatrixWorld(true);
    const wanted = this.frameFor(goal, new THREE.Matrix4());
    const tcpM = robot.tcp.matrixWorld;
    const here = this.tip.copy(BLADE_CENTER).applyMatrix4(tcpM).clone();
    const e = [goal.x - here.x, goal.y - here.y, goal.z - here.z];
    // Rotation error, from the difference of the two attitudes.
    const qa = new THREE.Quaternion().setFromRotationMatrix(wanted);
    const qb = new THREE.Quaternion().setFromRotationMatrix(tcpM);
    const d = qa.multiply(qb.invert());
    if (d.w < 0) d.set(-d.x, -d.y, -d.z, -d.w);
    e.push(2 * d.x, 2 * d.y, 2 * d.z);
    const J = [];
    for (const j of this.joints) {
      this.p.setFromMatrixPosition(j.origin.matrixWorld);
      this.a.copy(j.axis).transformDirection(j.origin.matrixWorld);
      const rx = here.x - this.p.x;
      const ry = here.y - this.p.y;
      const rz = here.z - this.p.z;
      J.push([
        this.a.y * rz - this.a.z * ry,
        this.a.z * rx - this.a.x * rz,
        this.a.x * ry - this.a.y * rx,
        this.a.x, this.a.y, this.a.z,
      ]);
    }
    // (J Jt + damping) x = e, then dq = Jt x.
    const A = [];
    for (let r = 0; r < 6; r++) {
      A.push(new Array(7).fill(0));
      for (let c = 0; c < 6; c++) {
        let sum = r === c ? 0.003 : 0;
        for (let k = 0; k < 7; k++) sum += J[k][r] * J[k][c];
        A[r][c] = sum;
      }
      A[r][6] = e[r];
    }
    for (let c = 0; c < 6; c++) {
      let best = c;
      for (let r = c + 1; r < 6; r++) if (Math.abs(A[r][c]) > Math.abs(A[best][c])) best = r;
      [A[c], A[best]] = [A[best], A[c]];
      const pivot = A[c][c] || 1e-9;
      for (let r = 0; r < 6; r++) {
        if (r === c) continue;
        const f = A[r][c] / pivot;
        for (let k = c; k <= 6; k++) A[r][k] -= f * A[c][k];
      }
    }
    const x = A.map((row, i) => row[6] / (row[i] || 1e-9));
    for (let i = 0; i < 7; i++) {
      let dq = 0;
      for (let r = 0; r < 6; r++) dq += J[i][r] * x[r];
      // A gentle pull toward the ready posture keeps the elbow sensible.
      dq += (player.readyQ[i] - player.q[i]) * 0.004;
      const most = this.joints[i].velocity * SIM_DT * 0.5;
      dq = Math.max(-most, Math.min(most, dq));
      player.q[i] = Math.max(this.joints[i].lower, Math.min(this.joints[i].upper, player.q[i] + dq));
    }
    robot.setArm(player.q);
  }
}

// --- playback --------------------------------------------------------------
class Player {
  constructor(plans, robot, tool, hair) {
    Object.assign(this, { plans, robot, tool, hair });
    this.queue = [];
    this.segment = null;
    this.time = 0;
    this.speed = 1;
    this.paused = false;
    this.endsAt = "home"; // where the arm will be once the queue has played: home | ready
    this.height = 0;
    this.activeRow = null;
    this.snips = 0;
    this.cycle = "moving"; // moving | settling | closing
    this.settle = 0;
    this.blades = new THREE.Vector3();
    this.lastBlades = new THREE.Vector3();
    this.frame = { inverse: new THREE.Matrix4(), world: null };
    this.q = new Array(7).fill(0);
    this.clock = 0; // simulated seconds
    this.downSpeed = 0; // m/s toward the scalp along the tool's axis
    this.sideSpeed = 0; // m/s across it
    this.maxSideInHair = 0;
    this.maxHooked = 0;
    this.snags = 0;
    this.lastSnipCut = 0;
    this.cutBefore = 0;
    this.simMs = 0;
    this.log = [];
    this.held = false; // the arm is holding still against the pull of snagged hair
    this.heldTime = 0;
    this.manual = false; // the visitor is driving
    this.closeHeld = false;
    this.goal = new THREE.Vector3();
    this.target = new THREE.Vector3();
    this.ik = null;
    this.readyQ = plans.home_to_ready.q[plans.home_to_ready.q.length - 1];
    this.reopen = false;
    robot.setArm(this.q);
    robot.root.updateMatrixWorld(true);
    this.updateTool();
  }

  get busy() {
    return !!this.segment || this.queue.length > 0;
  }

  note(text) {
    this.log.push(`${this.clock.toFixed(1).padStart(6)}s  ${text}`);
    if (this.log.length > 40) this.log.shift();
  }

  enqueueRows(rows, thenHome = false) {
    if (this.endsAt === "home") this.queue.push({ kind: "move", traj: this.plans.home_to_ready, label: "home to ready" });
    for (const row of rows) this.queue.push({ kind: "row", row, height: this.height });
    this.endsAt = "ready";
    if (thenHome) {
      this.queue.push({ kind: "move", traj: this.plans.ready_to_home, label: "ready to home" });
      this.endsAt = "home";
    }
  }

  // The bad way, on purpose: blades open and comb in the hair, dragged along a row.
  enqueueDrag() {
    if (this.endsAt === "home") this.queue.push({ kind: "move", traj: this.plans.home_to_ready, label: "home to ready" });
    const d = this.plans.drag;
    const shared = { kind: "move", rowIndex: d.row, height: d.height, dragging: true };
    this.queue.push(
      { ...shared, traj: d.approach, label: "approach" },
      { ...shared, traj: d.down, label: "into hair" },
      { ...shared, traj: d.stroke, label: "drag" },
      { ...shared, traj: d.lift, label: "lift" },
      { ...shared, traj: d.leave, label: "leave" },
    );
    this.endsAt = "ready";
    this.note("drag through the hair: on purpose, to see it snag");
  }

  // Rows expand into their moves at the moment they start, so the cut height chosen when
  // they were queued is the one used.
  nextSegment() {
    while (!this.segment && this.queue.length) {
      const item = this.queue.shift();
      if (item.kind === "row") {
        const row = this.plans.heights[item.height].rows[item.row];
        const shared = { kind: "move", rowIndex: item.row, height: item.height, stations: row.stations.length };
        const moves = [{ ...shared, traj: row.approach, label: "approach" }];
        row.stations.forEach((station, s) => {
          moves.push({ ...shared, traj: station.travel, label: "travel", station: s });
          moves.push({ ...shared, traj: station.raise, label: "raise", station: s, snipAfter: true });
        });
        moves.push({ ...shared, traj: row.lift, label: "lift" }, { ...shared, traj: row.leave, label: "leave" });
        this.queue.unshift(...moves);
        this.note(`row ${item.row + 1}: ${row.stations.length} stations at ${(this.plans.heights[item.height].clearance * 100).toFixed(1)} cm`);
        continue;
      }
      this.segment = item;
      this.time = 0;
      this.activeRow = item.rowIndex ?? null;
      this.cycle = "moving";
      this.reopen = false;
      if (item.label === "raise") this.note(`s${item.station + 1}: comb at the bottom, lifting`);
    }
  }

  // True while snagged hair pulls harder than the arm is set to allow: it holds until the hair
  // has given way well below that.
  checkLoad(dt) {
    const load = this.hair.toolLoad;
    if (!this.held && load > LOAD_LIMIT) {
      this.held = true;
      this.heldTime = 0;
      this.note(`hair pulling ${load.toFixed(0)} N: arm holding`);
    } else if (this.held && load < LOAD_LIMIT * 0.5) {
      this.held = false;
      this.note(`hair gave way after ${this.heldTime.toFixed(1)} s`);
    }
    if (this.held) this.heldTime += dt;
    return this.held;
  }

  startDriving() {
    this.manual = true;
    this.target.copy(this.blades);
    this.goal.copy(this.blades);
    this.note("you have the arm: live inverse kinematics, not DimOS");
  }

  // Give the arm back: up out of the hair, then a smooth joint move to the ready posture. This
  // move is worked out here and isn't checked by DimOS.
  stopDriving() {
    this.manual = false;
    const from = [...this.q];
    const steps = 75;
    const q = [];
    for (let k = 0; k <= steps; k++) {
      const u = k / steps;
      const s = u * u * (3 - 2 * u);
      q.push(from.map((v, j) => v + (this.readyQ[j] - v) * s));
    }
    this.queue.push({ kind: "move", traj: { dt: 1 / 30, duration: steps / 30, q }, label: "return to ready" });
    this.endsAt = "ready";
    this.note("handing back to the plans");
  }

  // One step of the arm following the visitor's pointer.
  driveStep(dt) {
    if (this.checkLoad(dt)) return;
    // Slower once it's near the hair, the way a careful controller would go.
    const near = this.hair.scalp.height(this.target) < 0.09;
    const limit = (near ? 0.1 : 0.3) * dt;
    const gap = this.target.distanceTo(this.goal);
    if (gap > 1e-6) this.target.lerp(this.goal, Math.min(1, limit / gap));
    this.ik.solve(this, this.target);
  }

  // Move the gripper toward a drive value at its real speed; true once there. Hair being cut
  // pushes back: the fingers slow as the force approaches what the gripper can give.
  moveGripper(target, dt) {
    const closing = target > this.robot.drive;
    const load = closing ? Math.min(0.92, this.hair.bladeForce / GRIPPER_FORCE) : 0;
    const step = DRIVE_SPEED * dt * (1 - load);
    const delta = target - this.robot.drive;
    this.robot.setGripper(Math.abs(delta) <= step ? target : this.robot.drive + Math.sign(delta) * step);
    return Math.abs(delta) <= step;
  }

  updateTool() {
    this.robot.root.updateMatrixWorld(true);
    this.tool.update(this.robot.padHalfGap());
    this.frame.world = this.robot.tcp.matrixWorld;
    this.frame.inverse.copy(this.frame.world).invert();
    this.blades.copy(BLADE_CENTER).applyMatrix4(this.frame.world);
  }

  update(dt) {
    const started = performance.now();
    this.clock += dt;
    this.nextSegment();
    const angleBefore = this.tool.angle;

    if (this.segment) {
      const { traj, label } = this.segment;
      if (this.cycle === "moving") {
        // Snagged hair can pull harder than the arm is set to allow: it waits for the hair to give.
        if (!this.checkLoad(dt)) this.time += dt; // the arm only moves in this state
        // Blades stay shut except from above the hair on the way to a station until its snip.
        if (label === "raise" || (this.segment.dragging && (label === "into hair" || label === "drag"))) this.moveGripper(DRIVE_OPEN, dt);
        else if (label === "travel") {
          if (this.hair.scalp.height(this.blades) > OPEN_ABOVE) this.reopen = true;
          if (this.reopen) this.moveGripper(DRIVE_OPEN, dt);
        } else this.moveGripper(DRIVE_CLOSED, dt);
        if (this.time >= traj.duration) {
          if (this.segment.snipAfter) {
            this.cycle = "settling";
            this.settle = 0;
          } else this.segment = null;
        }
      } else if (this.cycle === "settling") {
        this.settle += dt;
        if (this.settle >= SETTLE_TIME) {
          this.cycle = "closing";
          this.cutBefore = this.hair.cutTotal;
        }
      } else if (this.cycle === "closing" && this.moveGripper(DRIVE_CLOSED, dt)) {
        this.snips++;
        this.lastSnipCut = this.hair.cutTotal - this.cutBefore;
        this.note(`s${this.segment.station + 1}: snip, ${this.lastSnipCut} cuts`);
        this.segment = null;
        this.cycle = "moving";
      }

      const f = Math.min(this.time / traj.dt, traj.q.length - 1);
      const i = Math.min(Math.floor(f), traj.q.length - 2);
      const w = f - i;
      for (let j = 0; j < 7; j++) this.q[j] = traj.q[i][j] + (traj.q[i + 1][j] - traj.q[i][j]) * w;
      this.robot.setArm(this.q);
      this.lastBlades.copy(this.blades);
      this.updateTool();
      if (this.cycle === "closing") {
        this.hair.shear(this.frame, this.tool.angle, angleBefore, this.tool.pivotX);
      }
    } else if (this.manual) {
      this.activeRow = null;
      this.lastBlades.copy(this.blades);
      this.driveStep(dt);
      this.moveGripper(this.closeHeld ? DRIVE_CLOSED : DRIVE_OPEN, dt);
      this.updateTool();
      if (this.tool.angle < angleBefore) this.hair.shear(this.frame, this.tool.angle, angleBefore, this.tool.pivotX);
      else this.hair.bladeForce = 0;
    } else {
      this.activeRow = null;
      this.moveGripper(DRIVE_CLOSED, dt);
      this.lastBlades.copy(this.blades);
      this.updateTool();
    }

    // How the blades are moving, split along and across the tool's own axis.
    const w = this.frame.world.elements;
    const mx = this.blades.x - this.lastBlades.x;
    const my = this.blades.y - this.lastBlades.y;
    const mz = this.blades.z - this.lastBlades.z;
    const along = mx * w[8] + my * w[9] + mz * w[10];
    this.downSpeed = along / dt;
    this.sideSpeed = Math.sqrt(Math.max(0, mx * mx + my * my + mz * mz - along * along)) / dt;

    if (this.cycle !== "closing" && !this.manual) this.hair.bladeForce = 0;
    this.hair.step(this.frame, this.tool.angle, this.tool.pivotX, this.downSpeed > 0.005, this.sideSpeed > SNAG_SPEED);

    if (this.hair.contactCount > 0 && this.sideSpeed > this.maxSideInHair) this.maxSideInHair = this.sideSpeed;
    if (this.hair.contactCount > 0 && this.sideSpeed > SNAG_SPEED) this.snags++;
    if (this.hair.hookedCount > this.maxHooked) this.maxHooked = this.hair.hookedCount;
    this.simMs += (performance.now() - started - this.simMs) * 0.1;
  }
}

// What the arm is doing, in plain words, for the status line and the readout.
const PHASES = [
  ["over", "moving over, above the hair"],
  ["down", "going down into the hair"],
  ["lift", "lifting the hair with the comb"],
  ["settle", "letting the hair settle"],
  ["snip", "closing the blades"],
];
function phaseOf(player) {
  const seg = player.segment;
  let key = "idle";
  if (seg) {
    if (player.cycle === "settling") key = "settle";
    else if (player.cycle === "closing") key = "snip";
    else if (seg.label === "raise") key = "lift";
    else if (seg.label === "travel") key = player.downSpeed > 0.005 && player.sideSpeed < SNAG_SPEED ? "down" : "over";
    else key = seg.label;
  }
  const phase = PHASES.find(([k]) => k === key);
  return { key, text: phase ? phase[1] : key };
}

// What each joint has to hold: the weight of everything past it, plus the hair's pull on the
// tool. A static figure, from the URDF's masses; it leaves out what moving takes.
class HoldingTorque {
  constructor(kinematics, robot) {
    this.joints = [];
    const links = kinematics.links ?? {};
    for (let i = 1; i <= 7; i++) {
      const spec = robot.joints.get(`joint${i}`);
      const carried = [];
      const below = [spec.child];
      while (below.length) {
        const group = below.pop();
        for (const [name, link] of robot.links) {
          if (link === group && links[name]) carried.push({ group, mass: links[name].mass, com: new THREE.Vector3().fromArray(links[name].com) });
        }
        group.traverse((o) => {
          if (o === group) return;
          for (const [name, link] of robot.links) {
            if (link === o && links[name]) carried.push({ group: o, mass: links[name].mass, com: new THREE.Vector3().fromArray(links[name].com) });
          }
        });
      }
      this.joints.push({ origin: spec.child.parent, axis: spec.axis, carried });
    }
    this.out = new Array(7).fill(0);
  }

  update(robot, blades, force) {
    robot.root.updateMatrixWorld(true);
    const p = new THREE.Vector3();
    const a = new THREE.Vector3();
    const r = new THREE.Vector3();
    const c = new THREE.Vector3();
    const weight = new THREE.Vector3();
    this.joints.forEach((j, i) => {
      p.setFromMatrixPosition(j.origin.matrixWorld);
      a.copy(j.axis).transformDirection(j.origin.matrixWorld);
      let t = 0;
      for (const link of j.carried) {
        r.copy(link.com).applyMatrix4(link.group.matrixWorld).sub(p);
        weight.set(0, 0, -9.81 * link.mass);
        t += c.crossVectors(r, weight).dot(a);
      }
      r.copy(blades).sub(p);
      weight.set(0, 0, -9.81 * SCISSOR_MASS).add(force);
      t += c.crossVectors(r, weight).dot(a);
      this.out[i] = -t;
    });
    return this.out;
  }
}

// The live readout: everything here is read from the running simulation.
function buildInside(kinematics, robot) {
  const torqueModel = new HoldingTorque(kinematics, robot);
  const joints = kinematics.joints.filter((j) => /^joint\d$/.test(j.name));
  const box = $("inside-body");
  const row = (label, id) => `<div class="kv"><span>${label}</span><b id="in-${id}">-</b></div>`;
  box.innerHTML = `
    <h3>Program</h3>
    <div class="steps" id="in-steps">${PHASES.map(([k]) => `<span data-k="${k}">${k}</span>`).join("")}</div>
    ${row("Move", "move")}${row("Plan clock", "clock")}${row("Plan sample", "sample")}
    <pre id="in-log"></pre>
    <h3>Scissors, world mm</h3>
    <div class="xyz"><span>x</span><span>y</span><span>z</span></div>
    ${row("Blades", "blade")}${row("Wrist point", "tcp")}${row("Tool axis", "axis")}
    ${row("Blades off scalp", "bh")}${row("Comb off scalp", "ch")}
    ${row("Speed down", "down")}${row("Speed sideways", "side")}
    ${row("Sideways in hair, max", "snagmax")}${row("Snag events", "snags")}
    <h3>Gripper and blades</h3>
    ${row("Drive joint", "drive")}${row("Finger gap", "gap")}${row("Blade half-angle", "angle")}${row("Pivot from fingers", "pivot")}
    <h3>Joints: degrees, holding torque N·m</h3>
    <div class="joints">${joints.map((j, i) => `<div class="joint"><span>J${i + 1}</span><i><em id="in-jb${i}"></em></i><b id="in-j${i}">-</b><b id="in-t${i}">-</b></div>`).join("")}</div>
    <h3>Hair physics</h3>
    ${row("Strands awake", "awake")}${row("Points on comb", "contact")}${row("Through blade plane", "rising")}${row("Between the blades", "jaws")}
    ${row("Strands cut", "cut")}${row("Last snip", "lastsnip")}${row("Clippings falling", "falling")}${row("Clippings total", "pieces")}
    <h3>Snags and forces</h3>
    ${row("Strands caught", "hooked")}${row("Slid along the tool", "slid")}${row("Strands snapped", "broken")}
    ${row("Knots now", "knots")}${row("Knots pulled apart", "kslip")}
    ${row("Hair pull on tool", "load")}${row("Force on blades", "bforce")}${row("Most crowded cell", "crowd")}
    <h3>Performance</h3>
    ${row("Sim step", "sim")}${row("Frame rate", "fps")}
`;
  const get = (id) => document.getElementById(`in-${id}`);
  const ids = {};
  for (const id of ["move", "clock", "sample", "blade", "tcp", "axis", "bh", "ch", "down", "side", "snagmax", "snags", "drive", "gap", "angle", "pivot", "awake", "contact", "rising", "jaws", "cut", "lastsnip", "falling", "pieces", "sim", "fps", "log", "steps", "hooked", "slid", "broken", "knots", "kslip", "load", "bforce", "crowd"]) ids[id] = get(id);
  const jointValue = joints.map((_, i) => get(`j${i}`));
  const jointBar = joints.map((_, i) => get(`jb${i}`));
  const jointTorque = joints.map((_, i) => get(`t${i}`));
  const steps = [...ids.steps.children];
  const mm = (v) => (v * 1000).toFixed(1).padStart(7);
  const xyz = (v) => `${mm(v[0])}${mm(v[1])}${mm(v[2])}`;
  const v = new THREE.Vector3();
  let lastLog = "";
  return {
    update(player, hair, robot, tool, fps) {
      if (!document.getElementById("inside").open) return;
      const seg = player.segment;
      const phase = phaseOf(player);
      for (const s of steps) s.classList.toggle("on", s.dataset.k === phase.key);
      ids.move.textContent = seg
        ? `${seg.rowIndex !== undefined ? `r${seg.rowIndex + 1} ` : ""}${seg.station !== undefined ? `s${seg.station + 1}/${seg.stations} ` : ""}${seg.label}`
        : "idle";
      if (seg) {
        const t = Math.min(player.time, seg.traj.duration);
        ids.clock.textContent = `${t.toFixed(2)} / ${seg.traj.duration.toFixed(2)} s`;
        ids.sample.textContent = `${Math.min(Math.round(t / seg.traj.dt), seg.traj.q.length - 1)} / ${seg.traj.q.length - 1}`;
      } else {
        ids.clock.textContent = "-";
        ids.sample.textContent = "-";
      }
      const w = player.frame.world.elements;
      ids.blade.textContent = xyz([player.blades.x, player.blades.y, player.blades.z]);
      ids.tcp.textContent = xyz([w[12], w[13], w[14]]);
      ids.axis.textContent = `${w[8].toFixed(2).padStart(7)}${w[9].toFixed(2).padStart(7)}${w[10].toFixed(2).padStart(7)}`;
      ids.bh.textContent = `${(hair.scalp.height(player.blades) * 1000).toFixed(1)} mm`;
      v.copy(COMB_MIDDLE).applyMatrix4(player.frame.world);
      ids.ch.textContent = `${(hair.scalp.height(v) * 1000).toFixed(1)} mm`;
      ids.down.textContent = `${(player.downSpeed * 1000).toFixed(1)} mm/s`;
      ids.side.textContent = `${(player.sideSpeed * 1000).toFixed(1)} mm/s`;
      ids.snagmax.textContent = `${(player.maxSideInHair * 1000).toFixed(1)} mm/s`;
      ids.snags.textContent = String(player.snags);
      ids.drive.textContent = `${robot.drive.toFixed(3)} rad`;
      ids.gap.textContent = `${(robot.padHalfGap() * 2000).toFixed(1)} mm`;
      ids.angle.textContent = `${THREE.MathUtils.radToDeg(tool.angle).toFixed(1)}°`;
      ids.pivot.textContent = `${(tool.pivotX * 1000).toFixed(1)} mm`;
      const torques = torqueModel.update(robot, player.blades, hair.toolForce);
      joints.forEach((j, i) => {
        jointValue[i].textContent = THREE.MathUtils.radToDeg(player.q[i]).toFixed(1);
        jointTorque[i].textContent = torques[i].toFixed(1);
        jointBar[i].style.left = `${((player.q[i] - j.lower) / (j.upper - j.lower)) * 100}%`;
      });
      const stats = hair.bladeStats(player.frame, tool.angle, tool.pivotX);
      ids.awake.textContent = `${hair.activeCount.toLocaleString("en-US")} / ${STRANDS.toLocaleString("en-US")}`;
      ids.contact.textContent = hair.contactCount.toLocaleString("en-US");
      ids.rising.textContent = stats.rising.toLocaleString("en-US");
      ids.jaws.textContent = stats.inJaws.toLocaleString("en-US");
      ids.cut.textContent = hair.cutCount.toLocaleString("en-US");
      ids.lastsnip.textContent = player.lastSnipCut.toLocaleString("en-US");
      ids.falling.textContent = hair.pieces.live.length.toLocaleString("en-US");
      ids.pieces.textContent = hair.pieces.used.toLocaleString("en-US");
      ids.hooked.textContent = hair.hookedCount.toLocaleString("en-US");
      ids.slid.textContent = hair.slid.toLocaleString("en-US");
      ids.broken.textContent = hair.broken.toLocaleString("en-US");
      ids.knots.textContent = hair.knotCount.toLocaleString("en-US");
      ids.kslip.textContent = hair.knotsSlipped.toLocaleString("en-US");
      ids.load.textContent = `${hair.toolLoad.toFixed(1)} N`;
      ids.bforce.textContent = `${hair.bladeForce.toFixed(1)} / ${GRIPPER_FORCE} N`;
      let crowd = 0;
      for (let t = 0; t < hair.grid.touchedCount; t++) {
        const c = hair.grid.touched[t];
        const here = hair.grid.moving[c] + hair.grid.still[c];
        if (here > crowd) crowd = here;
      }
      ids.crowd.textContent = crowd.toFixed(0);
      ids.sim.textContent = `${player.simMs.toFixed(1)} ms`;
      ids.fps.textContent = `${fps.toFixed(0)} fps`;
      const log = player.log.slice(-4).join("\n");
      if (log !== lastLog) {
        ids.log.textContent = log;
        lastLog = log;
      }
    },
  };
}

// --- app -------------------------------------------------------------------
async function main() {
  resize();
  const loader = new GLTFLoader();
  const textures = new THREE.TextureLoader();
  const [plans, kinematics, gltf, scalpData, headGltf, colour, normal] = await Promise.all([
    fetch("plans.json").then(need).then((r) => r.json()),
    fetch("robot/kinematics.json").then(need).then((r) => r.json()),
    loader.loadAsync("robot/xarm7.glb"),
    fetch("head/scalp.json").then(need).then((r) => r.json()),
    loader.loadAsync("head/LeePerrySmith.glb"),
    textures.loadAsync("head/Map-COL.jpg"),
    textures.loadAsync("head/Infinite-Level_02_Tangent_SmoothUV.jpg"),
  ]);
  colour.colorSpace = THREE.SRGBColorSpace;
  colour.anisotropy = normal.anisotropy = 8;

  const { poleTex } = buildShop(plans);
  const client = buildClient(plans);
  scene.add(client);
  const robot = new Robot(kinematics, gltf.scene);
  scene.add(robot.root);
  const tool = buildTool();
  robot.tcp.add(tool);
  const scalp = new Scalp(scalpData);
  const head = buildHead(headGltf, { colour, normal }, scalp, scalpData);
  scene.add(head);
  const landing = new Landing(client);
  const hair = new Hair(scalp, landing);
  scene.add(hair.mesh, hair.pieces.mesh);
  hair.pieces.landing = landing;
  const setHairColour = (hex) => {
    hair.setColour(hex);
    head.setScalpTint(hex);
  };
  setHairColour(HAIR_COLOURS[0][1]);
  const player = new Player(plans, robot, tool, hair);
  player.ik = new ArmIK(robot, kinematics, scalp);

  // Controls.
  const nRows = plans.heights[0].rows.length;
  const buttons = (host, items, onClick) =>
    items.map((item, i) => {
      const b = document.createElement("button");
      b.textContent = item;
      b.addEventListener("click", () => onClick(i, b));
      host.append(b);
      return b;
    });
  const select = (list, i) => list.forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));

  const rowButtons = buttons($("rows"), [...Array(nRows).keys()].map((i) => String(i + 1)), (i) => cutRow(i));
  rowButtons.forEach((b, i) => b.setAttribute("aria-label", `Cut row ${i + 1}`));
  const heightButtons = buttons(
    $("heights"),
    plans.heights.map((h) => `${(h.clearance * 100).toFixed(h.clearance * 100 % 1 ? 1 : 0)} cm`),
    (i) => { player.height = i; select(heightButtons, i); },
  );
  select(heightButtons, 0);
  const speedValues = [0.5, 1, 2, 4];
  const speedButtons = buttons($("speeds"), speedValues.map((s) => `${s}×`), (i) => {
    player.speed = speedValues[i];
    select(speedButtons, i);
  });
  select(speedButtons, 1);
  const styleButtons = buttons($("styles"), HAIRSTYLES.map((h) => h.name), (i) => {
    hair.setStyle(HAIRSTYLES[i]);
    select(styleButtons, i);
    player.note(`hairstyle: ${HAIRSTYLES[i].label.toLowerCase()}`);
  });
  styleButtons.forEach((b, i) => b.setAttribute("aria-label", HAIRSTYLES[i].label));
  select(styleButtons, 0);
  const colourButtons = HAIR_COLOURS.map(([name, hex], i) => {
    const b = document.createElement("button");
    b.style.background = hex;
    b.style.setProperty("--c", hex);
    b.setAttribute("aria-label", name);
    b.title = name;
    b.addEventListener("click", () => { setHairColour(hex); select(colourButtons, i); });
    $("colours").append(b);
    return b;
  });
  select(colourButtons, 0);

  function cutRow(i) {
    if (player.manual) return;
    player.enqueueRows([i]);
    $("pause").disabled = false;
  }
  $("drag").addEventListener("click", () => {
    if (player.manual) return;
    player.enqueueDrag();
    $("pause").disabled = false;
  });
  let wantDrive = false;
  $("drive").addEventListener("click", () => {
    if (player.manual) {
      player.stopDriving();
      $("drive").textContent = "Drive it yourself";
      $("close-blades").disabled = true;
      return;
    }
    wantDrive = true;
    if (player.endsAt === "home") player.enqueueRows([]);
    $("drive").textContent = "Stop driving";
  });
  const closeBlades = $("close-blades");
  const setClose = (on) => { player.closeHeld = on; closeBlades.setAttribute("aria-pressed", String(on)); };
  closeBlades.addEventListener("pointerdown", () => setClose(true));
  addEventListener("pointerup", () => setClose(false));
  addEventListener("keydown", (e) => {
    if (e.code === "Space" && player.manual && e.target === document.body) { e.preventDefault(); setClose(true); }
  });
  addEventListener("keyup", (e) => { if (e.code === "Space") setClose(false); });
  // While driving, dragging near the scissors moves them, in the plane facing the camera.
  const handle = new THREE.Mesh(
    new THREE.SphereGeometry(0.009, 16, 12),
    new THREE.MeshBasicMaterial({ color: "#7fe0c8", depthTest: false, transparent: true, opacity: 0.8 }),
  );
  handle.renderOrder = 13;
  handle.visible = false;
  scene.add(handle);
  let dragging = false;
  const dragPlane = new THREE.Plane();
  const pointerRay = new THREE.Raycaster();
  const pointerOn = (e) => new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  canvas.addEventListener("pointerdown", (e) => {
    if (!player.manual) return;
    const v = player.blades.clone().project(camera);
    const px = ((v.x + 1) / 2) * innerWidth;
    const py = ((1 - v.y) / 2) * innerHeight;
    if (Math.hypot(px - e.clientX, py - e.clientY) > 70) return;
    dragging = true;
    controls.enabled = false;
    canvas.setPointerCapture(e.pointerId);
    const facing = new THREE.Vector3();
    camera.getWorldDirection(facing);
    dragPlane.setFromNormalAndCoplanarPoint(facing, player.blades);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    pointerRay.setFromCamera(pointerOn(e), camera);
    const hit = new THREE.Vector3();
    if (pointerRay.ray.intersectPlane(dragPlane, hit)) player.goal.copy(hit);
  });
  addEventListener("pointerup", () => {
    dragging = false;
    controls.enabled = true;
  });
  window.__drive = { goal: (x, y, z) => player.goal.set(x, y, z) };
  $("cut-all").addEventListener("click", () => {
    if (player.manual) return;
    player.enqueueRows([...Array(nRows).keys()], true);
    $("pause").disabled = false;
  });
  $("pause").addEventListener("click", () => {
    player.paused = !player.paused;
    $("pause").textContent = player.paused ? "Resume" : "Pause";
  });
  $("reset").addEventListener("click", () => {
    hair.reset();
    player.note("reset: full head of hair");
  });
  let closeUp = false;
  $("view").addEventListener("click", () => {
    closeUp = !closeUp;
    cameraGoal = closeUp ? CLOSE : WIDE;
    $("view").textContent = closeUp ? "Wide" : "Close-up";
  });
  canvas.addEventListener("pointerdown", () => { cameraGoal = null; });

  // Click the hair to cut the nearest row. Rows run front to back, spaced along x.
  const ray = new THREE.Raycaster();
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", (e) => {
    if (player.manual || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
    const pointer = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(pointer, camera);
    // A sphere as big as the head with its hair, so clicks on the hair count.
    const hit = ray.ray.intersectSphere(new THREE.Sphere(hair.center, 0.125), new THREE.Vector3());
    if (!hit || hit.z < hair.center.z) return;
    const offset = hit.x - hair.center.x;
    let row = 0;
    plans.row_offsets.forEach((o, i) => {
      if (Math.abs(o - offset) < Math.abs(plans.row_offsets[row] - offset)) row = i;
    });
    cutRow(row);
  });

  const placeInside = () => {
    const bottom = document.querySelector(".sign").getBoundingClientRect().bottom;
    document.documentElement.style.setProperty("--inside-top", `${Math.ceil(bottom) + 8}px`);
  };
  placeInside();
  addEventListener("resize", placeInside);
  $("inside").open = !MOBILE;
  const inside = buildInside(kinematics, robot);
  const overlays = new THREE.Group();
  scene.add(overlays);
  const dots = (colour, size) => {
    const geometry = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(new Float32Array(MAX_CONTACT_POINTS * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", attribute);
    geometry.setDrawRange(0, 0);
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({ color: colour, size, depthTest: false, transparent: true }));
    points.frustumCulled = false;
    points.renderOrder = 11;
    overlays.add(points);
    return { points, attribute, geometry };
  };
  const contactDots = dots("#ffb000", 0.0022); // hair resting on the comb
  const axes = new THREE.AxesHelper(0.05);
  axes.material.depthTest = false;
  axes.renderOrder = 12;
  robot.tcp.add(axes);
  const planeOutline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(
      [[0.05, -0.03], [0.125, -0.03], [0.125, 0.03], [0.05, 0.03]].map(([x, y]) => new THREE.Vector3(x, y, PLANE_Z)),
    ),
    new THREE.LineBasicMaterial({ color: "#7fe0c8", depthTest: false, transparent: true }),
  );
  planeOutline.renderOrder = 12;
  robot.tcp.add(planeOutline);
  let overlaysOn = true;
  const setOverlays = (on) => {
    overlaysOn = on;
    overlays.visible = axes.visible = planeOutline.visible = on;
    $("overlays").setAttribute("aria-pressed", String(on));
  };
  setOverlays(!MOBILE);
  $("overlays").addEventListener("click", () => setOverlays(!overlaysOn));
  const updateOverlays = () => {
    if (!overlaysOn) return;
    const put = (target, source, count) => {
      const n = Math.min(count, MAX_CONTACT_POINTS);
      target.attribute.array.set(source.subarray(0, n * 3));
      target.attribute.clearUpdateRanges();
      target.attribute.addUpdateRange(0, n * 3);
      target.attribute.needsUpdate = true;
      target.geometry.setDrawRange(0, n);
    };
    put(contactDots, hair.contactPoints, hair.contactCount);
  };

  function updateUi() {
    if (wantDrive && !player.busy && !player.manual) {
      wantDrive = false;
      player.startDriving();
      closeBlades.disabled = false;
    }
    handle.visible = player.manual;
    if (player.manual) handle.position.copy(player.blades);
    for (const id of ["cut-all", "drag"]) $(id).disabled = player.manual || wantDrive;
    rowButtons.forEach((b) => { b.disabled = player.manual || wantDrive; });
    styleButtons.forEach((b) => { b.disabled = player.busy || player.manual || wantDrive; });
    inside.update(player, hair, robot, tool, fps);
    rowButtons.forEach((b, i) => b.classList.toggle("active", player.activeRow === i));
    $("pause").disabled = !player.busy;
    const pct = hair.percentCut();
    $("pole-fill").style.width = `${pct}%`;
    let text;
    const seg = player.segment;
    if (player.paused) text = "Paused.";
    else if (player.held) text = `Hair's snagged and pulling ${hair.toolLoad.toFixed(0)} N. The arm is holding.`;
    else if (player.manual) text = hair.hookedCount ? `You've snagged ${hair.hookedCount.toLocaleString("en-US")} strands.` : "You're driving. Drag the green dot; hold Close blades to cut.";
    else if (seg?.dragging && seg.label === "drag") text = `Dragging through the hair, which you'd never do: ${hair.hookedCount.toLocaleString("en-US")} strands caught.`;
    else if (seg?.label === "return to ready") text = "Going back to the ready position.";
    else if (seg?.station !== undefined) {
      const where = `Row ${player.activeRow + 1} of ${nRows}, snip ${seg.station + 1} of ${seg.stations}`;
      text = `${where}: ${phaseOf(player).text}.`;
    } else if (player.activeRow !== null) text = `Row ${player.activeRow + 1} of ${nRows}: ${seg?.label === "approach" ? "moving into place" : "lifting clear"}.`;
    else if (player.busy) text = "Moving into position.";
    else if (pct > 0) text = DEMO ? "All done." : "All done. Try another row or reset.";
    else text = "Ready. Pull up a chair.";
    const strands = hair.cutCount.toLocaleString("en-US");
    setStatus(pct > 0 ? `${text} ${strands} strands cut (${pct.toFixed(0)}%).` : text);
  }

  // Video mode: hands-free full haircut, moving in close while it cuts.
  let demoStage = 0;
  let demoClock = 0;
  if (DEMO) {
    document.body.classList.add("demo");
    camera.position.copy(WIDE.pos);
    controls.target.copy(WIDE.target);
    cameraGoal = null;
  }
  function updateDemo(dt) {
    demoClock += dt;
    if (demoStage === 0 && demoClock > 5) {
      player.enqueueRows([0, 1, 2, 3, 4], true);
      demoStage = 1;
    }
    if (demoStage === 1 && demoClock > 6.5) {
      cameraGoal = { ...CLOSE, rate: 0.5 };
      demoStage = 2;
    }
    if (demoStage === 2 && !player.busy) {
      cameraGoal = { ...WIDE, rate: 0.7 };
      demoStage = 3;
      demoClock = 0;
    }
    if (demoStage === 3 && demoClock > 4) window.__demo.finished = true;
  }

  // Exposed for automated checks and for recording.
  window.__demo = { HAIRSTYLES, player, hair, robot, tool, camera, controls, finished: false, cutRow };

  let last = performance.now();
  let uiClock = 0;
  let owed = 0; // simulated seconds still to run
  let fps = 60;
  renderer.setAnimationLoop((now) => {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (dt > 0) fps += (1 / dt - fps) * 0.05;
    if (DEMO) updateDemo(dt);
    if (!player.paused) {
      owed += dt * player.speed;
      let steps = 0;
      while (owed >= SIM_DT && steps < MAX_STEPS_PER_FRAME) {
        player.update(SIM_DT);
        owed -= SIM_DT;
        steps++;
      }
      if (owed >= SIM_DT) owed = 0; // too slow to keep up: run slower, not jumpier
      if (steps) {
        hair.flush();
        updateOverlays();
      }
    }
    poleTex.offset.y -= dt * 0.12;
    if (cameraGoal) {
      const rate = 1 - Math.exp(-dt * (cameraGoal.rate ?? 2.5));
      camera.position.lerp(cameraGoal.pos, rate);
      controls.target.lerp(cameraGoal.target, rate);
    }
    controls.update();
    uiClock += dt;
    if (uiClock > 0.12) {
      updateUi();
      uiClock = 0;
    }
    renderer.render(scene, camera);
  });
}

function need(response) {
  if (!response.ok) throw new Error(`${response.url}: HTTP ${response.status}`);
  return response;
}

main().catch((error) => {
  console.error(error);
  const box = $("error");
  box.hidden = false;
  box.textContent = /WebGL/i.test(String(error))
    ? "This demo needs WebGL, and your browser couldn't start it."
    : `The demo couldn't load: ${error.message}`;
  setStatus("Something went wrong.");
});
