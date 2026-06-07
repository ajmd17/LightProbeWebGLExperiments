import * as S from './shaders.js';
import { VoxelGrid, sceneBounds, voxelizeScene } from './voxel.js';
import { LightFieldProbeSystem } from './gi/lightFieldProbe.js';
import { TetrahedralProbeSystem } from './gi/tetrahedralProbe.js';

const RES = 128;
const OCT_RES = 256;
const LOW_RES = 64;
const NX = 4, NY = 4, NZ = 4;
const NUM_PROBES = NX * NY * NZ;

const $ = document.getElementById.bind(document);
const info = $('info');
const canvas = $('glcanvas');

// ─── WebGL 2.0 Context ───────────────────────────────────────────────
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const gl = canvas.getContext('webgl2', { alpha: false });
if (!gl) { info.textContent = 'ERROR: no WebGL 2.0'; throw Error(''); }
info.textContent = 'WebGL 2.0 OK';
gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');

// ─── Helpers ─────────────────────────────────────────────────────────
const cs = (s, t) => { const a = gl.createShader(t); gl.shaderSource(a, s); gl.compileShader(a); if (!gl.getShaderParameter(a, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(a)); gl.deleteShader(a); return null; } return a; };
const mkProg = (v, f) => { const a = cs(v, gl.VERTEX_SHADER), b = cs(f, gl.FRAGMENT_SHADER); if (!a||!b) return null; const p = gl.createProgram(); gl.attachShader(p, a); gl.attachShader(p, b); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(p)); return null; } return p; };
const ul = (p, n) => gl.getUniformLocation(p, n);

// ─── Textures ────────────────────────────────────────────────────────
function cmF16(w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  const p = (n, v) => gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  p(gl.TEXTURE_MIN_FILTER, gl.LINEAR); p(gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  p(gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return t;
}
function cmR16F(w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.R16F, w, h, 0, gl.RED, gl.HALF_FLOAT, null);
  const p = (n, v) => gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  p(gl.TEXTURE_MIN_FILTER, gl.NEAREST); p(gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  p(gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return t;
}
function t2D(w, h, ch) {
  const c = ch||4; const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, c===1?gl.R16F:gl.RGBA16F, w, h, 0, c===1?gl.RED:gl.RGBA, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
function t2DA(w, h, layers, ch, filter) {
  const f = filter || gl.LINEAR;
  const c = ch||4; const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
  const fmt = c===1 ? gl.R16F : c===2 ? gl.RG32F : gl.RGBA16F;
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, fmt, w, h, layers);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

// ─── Programs ────────────────────────────────────────────────────────
const pScene = mkProg(S.sceneGBufVert, S.sceneGBufFrag);
const pProbe = mkProg(S.probeGBufVert, S.probeGBufFrag);
const pOct   = mkProg(S.quadVert, S.octFrag);
const pIrr   = mkProg(S.quadVert, S.irrFrag);
const pVSM   = mkProg(S.quadVert, S.vsmFrag);
const pMarch = mkProg(S.quadVert, S.raymarchFrag);
const pSHSphere = mkProg(S.shSphereVert, S.shSphereFrag);
const pDirect = mkProg(S.quadVert, S.directFrag);
const pForward = mkProg(S.sceneGBufVert, S.forwardFrag);
const pProject = mkProg(S.quadVert, S.shProjectFrag);
const pTetDebug = mkProg(S.tetDebugVert, S.tetDebugFrag);
if (!pScene||!pProbe||!pOct||!pIrr||!pVSM||!pMarch||!pSHSphere||!pDirect||!pForward||!pProject||!pTetDebug) { info.textContent='Shader error'; throw Error('shaders'); }
info.textContent='Shaders OK';

// ─── Full-screen quad ────────────────────────────────────────────────
const qVAO = gl.createVertexArray(); gl.bindVertexArray(qVAO);
const qB = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qB);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// ─── Scene meshes ────────────────────────────────────────────────────
function box(cx,cy,cz, sx,sy,sz, r,g,b) {
  const hx=sx/2, hy=sy/2, hz=sz/2;
  const p = [
    cx-hx,cy-hy,cz+hz, cx+hx,cy-hy,cz+hz, cx+hx,cy+hy,cz+hz, cx-hx,cy+hy,cz+hz,
    cx+hx,cy-hy,cz-hz, cx-hx,cy-hy,cz-hz, cx-hx,cy+hy,cz-hz, cx+hx,cy+hy,cz-hz,
    cx-hx,cy-hy,cz-hz, cx-hx,cy-hy,cz+hz, cx-hx,cy+hy,cz+hz, cx-hx,cy+hy,cz-hz,
    cx+hx,cy-hy,cz+hz, cx+hx,cy-hy,cz-hz, cx+hx,cy+hy,cz-hz, cx+hx,cy+hy,cz+hz,
    cx-hx,cy+hy,cz-hz, cx+hx,cy+hy,cz-hz, cx+hx,cy+hy,cz+hz, cx-hx,cy+hy,cz+hz,
    cx-hx,cy-hy,cz+hz, cx+hx,cy-hy,cz+hz, cx+hx,cy-hy,cz-hz, cx-hx,cy-hy,cz-hz,
  ];
  const fn = [[0,0,1],[0,0,-1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0]];
  const n = []; for (const f of fn) for (let j=0;j<4;j++) n.push(f[0],f[1],f[2]);
  const idx = []; for (let f=0;f<6;f++) { const b=f*4; idx.push(b,b+1,b+2,b,b+2,b+3); }
  return {pos:p, nrm:n, col:[r,g,b], idx};
}
function sphere(cx,cy,cz, r, slats, staves, rcol,gcol,bcol) {
  const p = [], n = [];
  for (let j = 0; j <= slats; j++) {
    const theta = j * Math.PI / slats;
    for (let i = 0; i <= staves; i++) {
      const phi = i * 2 * Math.PI / staves;
      const x = r * Math.sin(theta) * Math.cos(phi);
      const y = r * Math.cos(theta);
      const z = r * Math.sin(theta) * Math.sin(phi);
      p.push(cx+x, cy+y, cz+z); n.push(x/r, y/r, z/r);
    }
  }
  const idx = [];
  for (let j = 0; j < slats; j++) {
    for (let i = 0; i < staves; i++) {
      const a = j*(staves+1)+i, b = a+staves+1;
      idx.push(a, b, a+1, a+1, b, b+1);
    }
  }
  return {pos:p, nrm:n, col:[rcol,gcol,bcol], idx};
}
function cone(cx,cy,cz, rad, height, segs, rcol,gcol,bcol) {
  const p = [], n = [];
  const tip = [cx, cy+height/2, cz], baseC = [cx, cy-height/2, cz];
  for (let i = 0; i <= segs; i++) {
    const a = i / segs * 2 * Math.PI;
    const x = rad * Math.cos(a), z = rad * Math.sin(a);
    p.push(baseC[0]+x, baseC[1], baseC[2]+z);
    const dx = x, dz = z, dl = Math.sqrt(dx*dx + dz*dz);
    const nx = dx/dl * height/2, ny = rad, nz = dz/dl * height/2;
    const nl = Math.sqrt(nx*nx + ny*ny + nz*nz);
    n.push(nx/nl, ny/nl, nz/nl);
  }
  p.push(tip[0], tip[1], tip[2]); n.push(0, 1, 0);
  p.push(baseC[0], baseC[1], baseC[2]); n.push(0, -1, 0);
  const tipIdx = p.length/3-2, baseIdx = p.length/3-1;
  const idx = [];
  for (let i = 0; i < segs; i++) idx.push(i, (i+1)%segs, tipIdx);
  for (let i = 1; i < segs-1; i++) idx.push(baseIdx, i, i+1);
  return {pos:p, nrm:n, col:[rcol,gcol,bcol], idx};
}
function torus(cx,cy,cz, R, r, us, vs, rcol,gcol,bcol) {
  const p = [], n = [];
  for (let j = 0; j <= vs; j++) {
    const v = j / vs * 2 * Math.PI;
    for (let i = 0; i <= us; i++) {
      const u = i / us * 2 * Math.PI;
      const x = (R + r*Math.cos(v)) * Math.cos(u);
      const y = r * Math.sin(v);
      const z = (R + r*Math.cos(v)) * Math.sin(u);
      p.push(cx+x, cy+y, cz+z);
      n.push(Math.cos(v)*Math.cos(u), Math.sin(v), Math.cos(v)*Math.sin(u));
    }
  }
  const idx = [];
  for (let j = 0; j < vs; j++) {
    for (let i = 0; i < us; i++) {
      const a = j*(us+1)+i, b = a+us+1;
      idx.push(a, b, a+1, a+1, b, b+1);
    }
  }
  return {pos:p, nrm:n, col:[rcol,gcol,bcol], idx};
}

const M = [
  box(0,-4,0, 10,0.5,10, 0.9,0.9,0.9),
  box(-5,0,0, 0.5,8,10, 1.0,0.2,0.2),
  box(5,0,0, 0.5,8,10, 0.2,0.2,1.0),
  box(0,0,-5, 10,8,0.5, 0.8,0.8,0.8),
  box(0,4,0, 10,0.5,10, 0.6,0.6,0.6),
  box(-2,-3,0, 2,2,2, 0.1,0.9,0.1),
  box(2.5,-2.5,-2, 1.5,3,1.5, 1.0,0.6,0.1),
  box(0,3.5,0, 4,0.2,4, 5.0,5.0,5.0),
  sphere(-1, -0.5, 1.5, 0.8, 24, 16, 0.7, 0.2, 0.2),
  cone(1, 0, 1.5, 0.7, 1.4, 24, 0.2, 0.4, 0.8),
  torus(0, 0, -1.5, 1.0, 0.35, 24, 16, 0.9, 0.7, 0.1),
];

const sceneVAOs = (() => {
  const r = [];
  for (const m of M) {
    const vc = m.pos.length/3;
    const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.pos),gl.STATIC_DRAW);
    const nb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.nrm),gl.STATIC_DRAW);
    const cd=new Float32Array(vc*3); for(let i=0;i<vc;i++){cd[i*3]=m.col[0];cd[i*3+1]=m.col[1];cd[i*3+2]=m.col[2];}
    const cb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,cb); gl.bufferData(gl.ARRAY_BUFFER,cd,gl.STATIC_DRAW);
    const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(m.idx),gl.STATIC_DRAW);
    const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,cb); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
    gl.bindVertexArray(null);
    r.push({vao, count:m.idx.length});
  }
  return r;
})();

// ─── CPU voxelization ──────────────────────────────────────────────
const bounds = sceneBounds(M);
const VOX = 128;
console.log(`Voxelizing at ${VOX}^3...`);
const voxels = voxelizeScene(M, bounds, VOX);
const occ = voxels.occ.reduce((a,b)=>a+b, 0);
console.log(`${occ} / ${VOX**3} occupied (${(occ/(VOX**3)*100).toFixed(1)}%)`);
window.__v = voxels;

// Debug probe sphere VAO
const debugSphereMesh = sphere(0,0,0, 0.12, 12, 8, 5, 5, 5);
for (let i = 0; i < debugSphereMesh.nrm.length; i++) debugSphereMesh.nrm[i] = -debugSphereMesh.nrm[i];
const debugSphereVAO = (() => {
  const m = debugSphereMesh;
  const vc = m.pos.length/3;
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.pos),gl.STATIC_DRAW);
  const nb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.nrm),gl.STATIC_DRAW);
  const cd=new Float32Array(vc*3); for(let i=0;i<vc;i++){cd[i*3]=m.col[0];cd[i*3+1]=m.col[1];cd[i*3+2]=m.col[2];}
  const cb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,cb); gl.bufferData(gl.ARRAY_BUFFER,cd,gl.STATIC_DRAW);
  const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(m.idx),gl.STATIC_DRAW);
  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,cb); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
  gl.bindVertexArray(null);
  return {vao, count:m.idx.length};
})();

// SH test sphere VAO (rendered in tetrahedral mode)
const shSphereMesh = sphere(0,0,0, 0.25, 16, 12, 1, 1, 1);
const shSphereVAO = (() => {
  const m = shSphereMesh;
  const vc = m.pos.length/3;
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.pos),gl.STATIC_DRAW);
  const nb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(m.nrm),gl.STATIC_DRAW);
  const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(m.idx),gl.STATIC_DRAW);
  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,nb); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
  gl.bindVertexArray(null);
  return {vao, count:m.idx.length};
})();

// ═══════════════════════════════════════════════════════════════════════
//  GI SYSTEMS
// ═══════════════════════════════════════════════════════════════════════

const GMIN = [-4.0, -5.0, -3.5], GMAX = [4.0, 3.0, 2.0];
const MAX_PROBES = 256;

const sceneData = { sceneVAOs, M, bounds, voxels };
const programs = { pProbe, pOct, pIrr, pVSM };

const lfp = new LightFieldProbeSystem({
  res: RES, octRes: OCT_RES, lowRes: LOW_RES,
  nx: NX, ny: NY, nz: NZ,
  gMin: GMIN, gMax: GMAX,
  maxProbes: MAX_PROBES,
});
lfp.setRenderResources(qVAO, sceneVAOs);
lfp.init(gl, sceneData, programs);

const tetSys = new TetrahedralProbeSystem({
  gridX: 3, gridY: 3, gridZ: 3,
  bounds: { min: [-3.5, -4, -2.5], max: [3.5, 3, 2.5] },
});
tetSys.init(gl, sceneData);

// Render cubemaps from each tet probe position and project to SH
tetSys.precompute(gl, {
  pProbe, pProject, sceneVAOs, qVAO,
  lightPos: [0, 3.5, 0],
  lightCol: [10, 10, 10],
  res: 64, sampleCount: 512,
  info
});

// Pre-compute per-mesh SH for tetrahedral mode
const centerResult = tetSys.interpolateAt([0, 0, 0]);
const centerSH = centerResult.sh;
console.log(`[TET] center tetIndex=${centerResult.tetIndex} bary=${centerResult.bary ? centerResult.bary.map(v=>v.toFixed(3)).join(',') : 'null'}`);
console.log(`[TET] center SH R0=${centerSH[0].toFixed(3)} G0=${centerSH[1].toFixed(3)} B0=${centerSH[2].toFixed(3)}`);

const meshCentroids = M.map(m => {
  const pos = m.pos;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < pos.length; i += 3) { cx += pos[i]; cy += pos[i+1]; cz += pos[i+2]; }
  const n = pos.length / 3;
  return [cx/n, cy/n, cz/n];
});
const meshSH = M.map((_, mi) => {
  const r = tetSys.interpolateAt(meshCentroids[mi]);
  if (r.tetIndex < 0) {
    console.log(`[TET] mesh[${mi}] centroid (${meshCentroids[mi].map(v=>v.toFixed(1)).join(',')}) OUTSIDE — using fallback`);
  }
  return r.tetIndex >= 0 ? r.sh : centerSH;
});

// Build tetrahedron wireframe VAO
const tetWirePos = [];
for (const tet of tetSys.tets) {
  const v = tet.v;
  const edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
  for (const [a,b] of edges) {
    const p = tetSys.probes[v[a]].pos;
    const q = tetSys.probes[v[b]].pos;
    tetWirePos.push(p[0], p[1], p[2], q[0], q[1], q[2]);
  }
}
const tetWireVAO = gl.createVertexArray();
gl.bindVertexArray(tetWireVAO);
const twb = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, twb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tetWirePos), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// Build tetrahedral probe point VAO
const tetProbePos = tetSys.probes.map(p => p.pos).flat();
const tetProbeVAO = gl.createVertexArray();
gl.bindVertexArray(tetProbeVAO);
const tpb = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, tpb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tetProbePos), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// Which system is active (LFP disabled for now)
let activeSystem = 'tetrahedral';
let lfpReady = false;
let showIndirect = false;
let showDebug = true;

// ═══════════════════════════════════════════════════════════════════════
//  PRECOMPUTATION
// ═══════════════════════════════════════════════════════════════════════

function precompute() {
  info.textContent = `${tetSys.probes.length} tetrahedral probes ready.`;
}

// ═══════════════════════════════════════════════════════════════════════
//  RENDERING
// ═══════════════════════════════════════════════════════════════════════

let scenePosTex, sceneNrmTex, sceneAlbTex, sceneDepthRB, sceneFBO;

function resizeSceneBufs(w, h) {
  if (scenePosTex) {
    gl.deleteTexture(scenePosTex); gl.deleteTexture(sceneNrmTex); gl.deleteTexture(sceneAlbTex);
    gl.deleteRenderbuffer(sceneDepthRB); gl.deleteFramebuffer(sceneFBO);
  }
  scenePosTex = t2D(w, h);
  sceneNrmTex = t2D(w, h);
  sceneAlbTex = t2D(w, h);
  sceneDepthRB = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepthRB);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
  sceneFBO = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, scenePosTex, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, sceneNrmTex, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, sceneAlbTex, 0);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, sceneDepthRB);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) console.error('Scene FBO inc');
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ─── Camera controls ────────────────────────────────────────────────
let camPos = [0, 1, 8];
let yaw = 0, pitch = 0;
const keys = new Set();
const camSpeed = 4;

let singleProbe = -1;
let probeCycle = [0,0,0,0,0,0,0,0,0,0];

function updateInfo() {
  const sp = singleProbe >= 0 ? ` [probe:${singleProbe}]` : '';
  const gi = activeSystem === 'lightfield' ? 'LFP' : 'TET';
  const dbg = showIndirect ? ' [IRR]' : '';
  const dbg2 = showDebug ? ' [DBG]' : '';
  info.textContent = (document.pointerLockElement === canvas ? 'Click to release mouse' : `${gi} | ${tetSys.probes.length} probes ${tetSys.tets.length} tets`) + sp + dbg + dbg2;
}

canvas.addEventListener('click', () => canvas.requestPointerLock());
document.addEventListener('pointerlockchange', updateInfo);
document.addEventListener('mousemove', e => {
  if (document.pointerLockElement !== canvas) return;
  yaw += e.movementX * 0.002;
  pitch = Math.max(-1.5, Math.min(1.5, pitch - e.movementY * 0.002));
});
document.addEventListener('keydown', e => {
  keys.add(e.code);
  const digit = parseInt(e.code.replace('Digit', ''));
  const isDigit = !isNaN(digit) && digit >= 0 && digit <= 9;
  if (isDigit && activeSystem === 'lightfield') {
    const maxOff = Math.ceil((NUM_PROBES - 1 - digit) / 10);
    const off = probeCycle[digit];
    singleProbe = off * 10 + digit;
    lfp.singleProbe = singleProbe;
    probeCycle[digit] = (off + 1) > maxOff ? 0 : off + 1;
  } else if (e.code === 'KeyR') {
    singleProbe = -1;
    lfp.singleProbe = -1;
  } else if (e.code === 'KeyT') {
    activeSystem = activeSystem === 'lightfield' ? 'tetrahedral' : 'lightfield';
    singleProbe = -1;
    lfp.singleProbe = -1;
    console.log(`[GI] Switched to ${activeSystem}`);
  } else if (e.code === 'KeyI') {
    showIndirect = !showIndirect;
    console.log(`[TET] showIndirect=${showIndirect}`);
  } else if (e.code === 'KeyU') {
    showDebug = !showDebug;
    console.log(`[TET] showDebug=${showDebug}`);
  }
  updateInfo();
});
document.addEventListener('keyup', e => { keys.delete(e.code); });

function updateCamera(dt) {
  const cx = Math.cos(yaw), sx = Math.sin(yaw);
  const cy = Math.cos(pitch), sy = Math.sin(pitch);
  const fwd = [cx * cy, sy, sx * cy];
  const right = [-sx, 0, cx];
  let speed = camSpeed * dt;
  if (keys.has('ShiftLeft') || keys.has('ShiftRight')) speed *= 3;
  if (keys.has('KeyW') || keys.has('ArrowUp'))    camPos = add(camPos, scale(fwd, speed));
  if (keys.has('KeyS') || keys.has('ArrowDown'))  camPos = add(camPos, scale(fwd, -speed));
  if (keys.has('KeyA') || keys.has('ArrowLeft'))  camPos = add(camPos, scale(right, -speed));
  if (keys.has('KeyD') || keys.has('ArrowRight')) camPos = add(camPos, scale(right, speed));
  if (keys.has('Space'))                          camPos[1] += speed;
  if (keys.has('KeyZ'))                           camPos[1] -= speed;
}

function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }

// SH test sphere orbit
let testPhi = 0;
const ORBIT_RADIUS = 0.6;
const ORBIT_SPEED = 0.6;

let lastTime = 0;

function renderFrame(time) {
  const dt = Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;
  updateCamera(dt);

  const w = canvas.clientWidth * devicePixelRatio;
  const h = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; resizeSceneBufs(w, h); }

  const proj = persp(Math.PI/3, w/h, 0.1, 20);
  const view = lookAt(camPos, add(camPos, [Math.cos(yaw)*Math.cos(pitch), Math.sin(pitch), Math.sin(yaw)*Math.cos(pitch)]), [0,1,0]);
  const vp = mul4(proj, view);

  // ── 1. Forward render (tetrahedral mode) ──────────────────────
  if (activeSystem === 'tetrahedral') {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0,0,w,h);
    gl.clearColor(0.05,0.05,0.05,1); gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);

    gl.useProgram(pForward);
    gl.uniformMatrix4fv(ul(pForward,'uVP'), false, vp);
    gl.uniform3fv(ul(pForward,'uLightPos'), [0, 3.5, 0]);
    gl.uniform3fv(ul(pForward,'uLightCol'), [10, 10, 10]);
    gl.uniform1f(ul(pForward,'uShowIndirect'), showIndirect ? 1 : 0);

    for (let mi = 0; mi < M.length; mi++) {
      gl.uniform3fv(ul(pForward, 'uSH'), meshSH[mi]);
      gl.uniformMatrix4fv(ul(pForward,'uM'), false, ID);
      gl.bindVertexArray(sceneVAOs[mi].vao);
      gl.drawElements(gl.TRIANGLES, sceneVAOs[mi].count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    }

    // ── Debug viz (wireframe + probe spheres) ─────────────────────
    if (showDebug) {
      gl.useProgram(pTetDebug);
      gl.uniformMatrix4fv(ul(pTetDebug,'uVP'), false, vp);
      gl.uniformMatrix4fv(ul(pTetDebug,'uM'), false, ID);
      gl.uniform1f(ul(pTetDebug,'uPointSize'), 1);
      gl.uniform3fv(ul(pTetDebug,'uColor'), [0.3, 0.5, 1.0]);
      gl.bindVertexArray(tetWireVAO);
      gl.drawArrays(gl.LINES, 0, tetWirePos.length / 3);
      gl.bindVertexArray(null);

      gl.useProgram(pSHSphere);
      gl.uniformMatrix4fv(ul(pSHSphere,'uVP'), false, vp);
      gl.uniform3fv(ul(pSHSphere,'uLightPos'), [0, 3.5, 0]);
      gl.uniform3fv(ul(pSHSphere,'uLightCol'), [0, 0, 0]);
      gl.uniform3fv(ul(pSHSphere,'uBaseCol'), [1, 1, 1]);
      gl.depthMask(true);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CW);
      for (let pi = 0; pi < tetSys.probes.length; pi++) {
        const pp = tetSys.probes[pi].pos;
        const s = 0.4;
        const M = new Float32Array([s,0,0,0, 0,s,0,0, 0,0,s,0, pp[0],pp[1],pp[2],1]);
        gl.uniformMatrix4fv(ul(pSHSphere,'uM'), false, M);
        gl.uniform3fv(ul(pSHSphere,'uSH'), tetSys.probes[pi].sh);
        gl.bindVertexArray(shSphereVAO.vao);
        gl.drawElements(gl.TRIANGLES, shSphereVAO.count, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
      }
      gl.disable(gl.CULL_FACE);
    }

    gl.disable(gl.DEPTH_TEST);

  } else {
    // ── 1. Scene G-buffer (LFP mode) ──────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO);
    gl.viewport(0,0,w,h);
    gl.clearColor(0,0,0,1); gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
    gl.useProgram(pScene);
    gl.uniformMatrix4fv(ul(pScene,'uVP'), false, vp);

    for (const m of sceneVAOs) { gl.uniformMatrix4fv(ul(pScene,'uM'),false,ID); gl.bindVertexArray(m.vao); gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0); gl.bindVertexArray(null); }

    if (singleProbe >= 0) {
      const pp = lfp.adjProbePos[singleProbe];
      const M = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, pp[0],pp[1],pp[2],1]);
      gl.uniformMatrix4fv(ul(pScene,'uM'), false, M);
      gl.bindVertexArray(debugSphereVAO.vao);
      gl.drawElements(gl.TRIANGLES, debugSphereVAO.count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    } else {
      for (let pi = 0; pi < NUM_PROBES; pi++) {
        const pp = lfp.adjProbePos[pi];
        const M = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, pp[0],pp[1],pp[2],1]);
        gl.uniformMatrix4fv(ul(pScene,'uM'), false, M);
        gl.bindVertexArray(debugSphereVAO.vao);
        gl.drawElements(gl.TRIANGLES, debugSphereVAO.count, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
      }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── 2. Background pass (LFP mode) ──────────────────────────────
    gl.viewport(0,0,w,h);
    gl.clearColor(0.05,0.05,0.05,1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);

    if (lfpReady) {
      gl.useProgram(pMarch);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scenePosTex);
      gl.uniform1i(ul(pMarch,'uPos'),0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, sceneNrmTex);
      gl.uniform1i(ul(pMarch,'uNrm'),1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, sceneAlbTex);
      gl.uniform1i(ul(pMarch,'uAlb'),2);
      lfp.bindUniforms(gl, pMarch);
      {
        const cy = Math.cos(yaw), sy = Math.sin(yaw);
        const cp = Math.cos(pitch), sp = Math.sin(pitch);
        const fwd = [cy*cp, sp, sy*cp], up = [-cy*sp, cp, -sy*sp], right = [-sy, 0, cy];
        gl.uniform3fv(ul(pMarch, 'uCamFwd'), fwd);
        gl.uniform3fv(ul(pMarch, 'uCamUp'), up);
        gl.uniform3fv(ul(pMarch, 'uCamRight'), right);
      }
      gl.uniform1f(ul(pMarch, 'uTanHalfFov'), Math.tan(Math.PI/6));
      gl.uniform1f(ul(pMarch, 'uAspect'), w/h);
      gl.uniform1f(ul(pMarch, 'uNormalBias'), 0.03);
      gl.uniform1f(ul(pMarch, 'uDistBias'), 0.02);
      gl.uniform1i(ul(pMarch,'uDebug'), 0);
      gl.uniform1i(ul(pMarch,'uSingleProbe'), singleProbe);
      gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES, 0, 6); gl.bindVertexArray(null);
    } else {
      gl.useProgram(pDirect);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scenePosTex);
      gl.uniform1i(ul(pDirect,'uPos'),0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, sceneNrmTex);
      gl.uniform1i(ul(pDirect,'uNrm'),1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, sceneAlbTex);
      gl.uniform1i(ul(pDirect,'uAlb'),2);
      gl.uniform3fv(ul(pDirect,'uLightPos'), [0, 3.5, 0]);
      gl.uniform3fv(ul(pDirect,'uLightCol'), [10, 10, 10]);
      gl.uniform3fv(ul(pDirect,'uAmbient'), [0.03, 0.04, 0.06]);
      gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES, 0, 6); gl.bindVertexArray(null);
    }
  }

  // ── 3. Tetrahedral SH sphere overlay ────────────────────────────
  if (activeSystem === 'tetrahedral') {
    testPhi += dt * ORBIT_SPEED;
    const testPos = [ORBIT_RADIUS * Math.cos(testPhi), 0.0, ORBIT_RADIUS * Math.sin(testPhi)];

    const result = tetSys.interpolateAt(testPos);
    if (result.tetIndex >= 0) {
      if (Math.floor(testPhi / (Math.PI * 2) * 10) !== Math.floor((testPhi - dt * ORBIT_SPEED) / (Math.PI * 2) * 10))
        console.log(`[TET] tet=${result.tetIndex} bary=(${result.bary.map(v=>v.toFixed(2)).join(',')}) pos=(${testPos.map(v=>v.toFixed(2)).join(',')}) sh_R0=${result.sh[0].toFixed(3)} G0=${result.sh[1].toFixed(3)} B0=${result.sh[2].toFixed(3)}`);
      const M = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, testPos[0],testPos[1],testPos[2],1]);
      gl.useProgram(pSHSphere);
      gl.uniformMatrix4fv(ul(pSHSphere,'uVP'), false, vp);
      gl.uniformMatrix4fv(ul(pSHSphere,'uM'), false, M);
      gl.uniform3fv(ul(pSHSphere,'uSH'), result.sh);
      gl.uniform3fv(ul(pSHSphere,'uLightPos'), [0, 3.5, 0]);
      gl.uniform3fv(ul(pSHSphere,'uLightCol'), [10, 10, 10]);
      gl.uniform3fv(ul(pSHSphere,'uBaseCol'), [0.8, 0.7, 0.6]);
      gl.depthMask(true);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CW);
      gl.bindVertexArray(shSphereVAO.vao);
      gl.drawElements(gl.TRIANGLES, shSphereVAO.count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.DEPTH_TEST);
    }
  }

  requestAnimationFrame(renderFrame);
}

// ─── Math helpers ────────────────────────────────────────────────────
function persp(fov,a,near,far){const f=1/Math.tan(fov/2), nf=1/(near-far); return new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);}
function lookAt(e,t,u){const z=norm(sub(e,t)), x=norm(cross(u,z)), y=cross(z,x); return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]);}
function norm(v){const l=Math.hypot(v[0],v[1],v[2]);return[v[0]/l,v[1]/l,v[2]/l];}
function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function mul4(a,b){const r=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++)r[j*4+i]=a[i]*b[j*4]+a[4+i]*b[j*4+1]+a[8+i]*b[j*4+2]+a[12+i]*b[j*4+3];return r;}
const ID=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

// ═══════════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════════
resizeSceneBufs(canvas.width, canvas.height);
precompute();
renderFrame(performance.now());
