import * as S from './shaders.js';

const RES = 64;
const OCT_RES = 256;
const LOW_RES = 64;
const NUM_PROBES = 216; // total probes: 6×6×6
const PROBE_DIM = 6; // probes per axis: 8×8×8

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
if (!pScene||!pProbe||!pOct||!pIrr||!pVSM||!pMarch) { info.textContent='Shader error'; throw Error('shaders'); }
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
  // side vertices
  for (let i = 0; i <= segs; i++) {
    const a = i / segs * 2 * Math.PI;
    const x = rad * Math.cos(a), z = rad * Math.sin(a);
    p.push(baseC[0]+x, baseC[1], baseC[2]+z);
    // side normal: normalize(cross(tangent_u, tangent_v))
    const dx = x, dz = z, dl = Math.sqrt(dx*dx + dz*dz);
    const nx = dx/dl * height/2, ny = rad, nz = dz/dl * height/2;
    const nl = Math.sqrt(nx*nx + ny*ny + nz*nz);
    n.push(nx/nl, ny/nl, nz/nl);
  }
  // tip vertex
  p.push(tip[0], tip[1], tip[2]);
  n.push(0, 1, 0);
  // base center
  p.push(baseC[0], baseC[1], baseC[2]);
  n.push(0, -1, 0);
  const tipIdx = p.length/3-2, baseIdx = p.length/3-1;
  const idx = [];
  for (let i = 0; i < segs; i++) {
    idx.push(i, (i+1)%segs, tipIdx);
  }
  // base cap (triangle fan)
  for (let i = 1; i < segs-1; i++) {
    idx.push(baseIdx, i, i+1);
  }
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
  // Room
  box(0,-4,0, 10,0.5,10, 0.9,0.9,0.9),
  box(-5,0,0, 0.5,8,10, 1.0,0.2,0.2),
  box(5,0,0, 0.5,8,10, 0.2,0.2,1.0),
  box(0,0,-5, 10,8,0.5, 0.8,0.8,0.8),
  box(0,4,0, 10,0.5,10, 0.6,0.6,0.6),
  // Existing objects
  box(-2,-3,0, 2,2,2, 0.1,0.9,0.1),
  box(2.5,-2.5,-2, 1.5,3,1.5, 1.0,0.6,0.1),
  // Ceiling light
  box(0,3.5,0, 4,0.2,4, 5.0,5.0,5.0),
  // New shapes
  sphere(-1, -0.5, 1.5, 0.8, 24, 16, 0.7, 0.2, 0.2),
  cone(1, 0, 1.5, 0.7, 1.4, 24, 0.2, 0.6, 0.4),
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

// Debug probe sphere VAO (rendered into G-buffer for spatial visualization)
const debugSphereMesh = sphere(0,0,0, 0.12, 12, 8, 5, 5, 5);
// Negate normals so the probe center passes the backface test (inward-facing normals)
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

// ─── Probe grid ──────────────────────────────────────────────────────
const GMIN = [-4.0, -5.0, -3.5], GMAX = [4.0, 3.0, 2.0];
const probePos = Array.from({length:NUM_PROBES}, (_,i)=>{
  const nx=PROBE_DIM, ny=PROBE_DIM, nz=PROBE_DIM;
  const ix=i%nx, iy=Math.floor(i/nx)%ny, iz=Math.floor(i/(nx*ny));
  const tx=ix/(nx-1), ty=iy/(ny-1), tz=iz/(nz-1);
  return [GMIN[0]+tx*(GMAX[0]-GMIN[0]), GMIN[1]+ty*(GMAX[1]-GMIN[1]), GMIN[2]+tz*(GMAX[2]-GMIN[2])];
});
const probePosFlat = new Float32Array(probePos.flat());

// ─── Probe G-buffer cubemap allocation ────────────────────────────────
function allocProbeGB() {
  const depthCM = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, depthCM);
  for (let j = 0; j < 6; j++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+j, 0, gl.DEPTH_COMPONENT24, RES, RES, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return { fbo:gl.createFramebuffer(), rad:cmF16(RES,RES), nrm:cmF16(RES,RES), dist:cmR16F(RES,RES), depthCM };
}
function freeProbeGB(gb) {
  gl.deleteFramebuffer(gb.fbo);
  gl.deleteTexture(gb.rad); gl.deleteTexture(gb.nrm); gl.deleteTexture(gb.dist); gl.deleteTexture(gb.depthCM);
}
const db3 = [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2];
function bindGBFace(fbo, face, r, n, d, depth) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const t = gl.TEXTURE_CUBE_MAP_POSITIVE_X + face;
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, r, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, t, n, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, t, d, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, t, depth, 0);
  gl.drawBuffers(db3);
}

// ─── Texture arrays ──────────────────────────────────────────────────
const OW = OCT_RES*2, OH = OCT_RES;
const LW = LOW_RES*2, LH = LOW_RES;
const irrArr = t2DA(OW, OH, NUM_PROBES, 4, gl.NEAREST);
const vsmArr = t2DA(OW, OH, NUM_PROBES, 2, gl.LINEAR);
const distHArr = t2DA(OW, OH, NUM_PROBES, 1, gl.LINEAR);
const distLArr = t2DA(LW, LH, NUM_PROBES, 1, gl.LINEAR);

const layerFBO = gl.createFramebuffer();

// ─── Math ────────────────────────────────────────────────────────────
function persp(fov,a,near,far){const f=1/Math.tan(fov/2), nf=1/(near-far); return new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);}
function lookAt(e,t,u){const z=norm(sub(e,t)), x=norm(cross(u,z)), y=cross(z,x); return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]);}
function norm(v){const l=Math.hypot(v[0],v[1],v[2]);return[v[0]/l,v[1]/l,v[2]/l];}
function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function mul4(a,b){const r=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++)r[j*4+i]=a[i]*b[j*4]+a[4+i]*b[j*4+1]+a[8+i]*b[j*4+2]+a[12+i]*b[j*4+3];return r;}
const ID=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

const cT=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const cU=[[0,-1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,-1,0],[0,-1,0]];

// ═══════════════════════════════════════════════════════════════════════
//  PRECOMPUTATION
// ═══════════════════════════════════════════════════════════════════════

function renderProbeGB(gb, pp) {
  const proj = persp(Math.PI/2, 1, 0.1, 20);
  gl.useProgram(pProbe);
  gl.uniform3fv(ul(pProbe, 'uQ'), pp);
  gl.uniform3fv(ul(pProbe, 'uLightPos'), [0, 3.5, 0]);
  gl.uniform3fv(ul(pProbe, 'uLightCol'), [20, 20, 20]);
  gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
  const blk = new Float32Array([0, 0, 0, 1]);
  const farDist = new Float32Array([20, 0, 0, 0]);
  for (let f = 0; f < 6; f++) {
    bindGBFace(gb.fbo, f, gb.rad, gb.nrm, gb.dist, gb.depthCM);
    const vp = mul4(proj, lookAt(pp, [pp[0]+cT[f][0],pp[1]+cT[f][1],pp[2]+cT[f][2]], cU[f]));
    gl.uniformMatrix4fv(ul(pProbe, 'uVP'), false, vp);
    gl.viewport(0,0,RES,RES);
    gl.clearBufferfv(gl.COLOR, 0, blk);
    gl.clearBufferfv(gl.COLOR, 1, blk);
    gl.clearBufferfv(gl.COLOR, 2, farDist);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    for (const m of sceneVAOs) { gl.uniformMatrix4fv(ul(pProbe,'uM'),false,ID); gl.bindVertexArray(m.vao); gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0); gl.bindVertexArray(null); }
  }
}

function makeOctPass(layer, cm, arr, prog, nrmCM, distCM) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, layerFBO);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, arr, 0, layer);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.viewport(0,0,OW,OH); gl.disable(gl.DEPTH_TEST);
  gl.useProgram(prog);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, cm);
  gl.uniform1i(ul(prog, 'uRad'), 0);
  if (prog === pOct) {
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_CUBE_MAP, nrmCM);
    gl.uniform1i(ul(prog, 'uNrm'), 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_CUBE_MAP, distCM);
    gl.uniform1i(ul(prog, 'uDist'), 2);
  }
  if (prog === pIrr) gl.uniform1i(ul(prog, 'uN'), 1024);
  const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (st !== gl.FRAMEBUFFER_COMPLETE) console.error('FBO inc:', st);
  gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES,0,6); gl.bindVertexArray(null);
}

function precompute() {
  for (let i = 0; i < NUM_PROBES; i++) {
    info.textContent = `Precomputing ${i+1}/${NUM_PROBES}...`;
    const gb = allocProbeGB();
    renderProbeGB(gb, probePos[i]);

    // High-res octahedral distance → distHArr
    makeOctPass(i, gb.dist, distHArr, pOct, gb.nrm, gb.dist);

    // Low-res octahedral distance → distLArr
    gl.bindFramebuffer(gl.FRAMEBUFFER, layerFBO);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, distLArr, 0, i);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0,0,LW,LH); gl.disable(gl.DEPTH_TEST);
    gl.useProgram(pOct);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
    gl.uniform1i(ul(pOct,'uRad'),0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.nrm);
    gl.uniform1i(ul(pOct,'uNrm'),1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
    gl.uniform1i(ul(pOct,'uDist'),2);
    gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES,0,6); gl.bindVertexArray(null);

    // Irradiance filter → irrArr
    makeOctPass(i, gb.rad, irrArr, pIrr, null, null);

    // VSM filter → vsmArr
    gl.bindFramebuffer(gl.FRAMEBUFFER, layerFBO);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, vsmArr, 0, i);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0,0,OW,OH);
    gl.useProgram(pVSM);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
    gl.uniform1i(ul(pVSM,'uDist'),0);
    gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES,0,6); gl.bindVertexArray(null);

    freeProbeGB(gb);
  }
  info.textContent = `${NUM_PROBES} probes ready.`;
}

// ═══════════════════════════════════════════════════════════════════════
//  RENDERING
// ═══════════════════════════════════════════════════════════════════════

// Scene G-buffer textures + depth (sized dynamically)
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

// ─── Camera controls (WASD + mouse look) ────────────────────────────
let camPos = [0, 1, 8];
let yaw = 0, pitch = 0;
const keys = new Set();
const camSpeed = 4;

let debugMode = 0;      // 0=normal, 1=octUV, 2=distance, 3=steps, 4-6=texture views, 7=trilinear-only, 8=tw, 9=backface, 10=vis
let singleProbe = -1;   // -1=all, 0..N-1=single probe only
let lChord = false;
const debugLabels = ['normal', 'oct UV', 'dist', 'steps', 'tex:irr', 'tex:dist', 'tex:split', 'trilinear', 'weight', 'backface', 'visibility'];
function updateInfo() {
  const dbg = debugMode ? ` [debug:${debugLabels[debugMode]}]` : '';
  const sp = singleProbe >= 0 ? ` [probe:${singleProbe}]` : '';
  info.textContent = (document.pointerLockElement === canvas ? 'Click to release mouse' : `${NUM_PROBES} probes ready. Click for mouse look`) + dbg + sp;
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
  if (e.code === 'KeyL' && !e.repeat) lChord = false;

  const digit = parseInt(e.code.replace('Digit', ''));
  const isDigit = !isNaN(digit) && digit >= 0 && digit <= 9;

  if (isDigit && keys.has('KeyL')) {
    // L+digit chord → texture view mode
    lChord = true;
    if (debugMode < 4 || debugMode > 6) debugMode = 4;
    const probeMap = [0, 73, 146, 219, 292, 365, 438, 63, 448, 511];
    singleProbe = probeMap[digit] ?? 0;
  } else if (isDigit) {
    const tbl = [
      { d:0, p:6 }, { d:1, p:-1 }, { d:2, p:-1 }, { d:3, p:-1 },
      { d:0, p:0 }, { d:0, p:1 }, { d:0, p:2 }, { d:0, p:3 },
      { d:0, p:4 }, { d:0, p:5 },
    ];
    debugMode = tbl[digit].d;
    singleProbe = tbl[digit].p;
  } else if (e.code === 'KeyR') {
    debugMode = 0;
    singleProbe = -1;
  }
  updateInfo();
});
document.addEventListener('keyup', e => {
  if (e.code === 'KeyL' && !lChord) {
    if (debugMode >= 4 && debugMode <= 6) {
      debugMode = ((debugMode - 3) % 3) + 4;
    } else {
      debugMode = 4;
      if (singleProbe < 0) singleProbe = 0;
    }
    updateInfo();
  }
  keys.delete(e.code);
});

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

  // 1. Scene G-buffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO);
  gl.viewport(0,0,w,h);
  gl.clearColor(0,0,0,1); gl.clearDepth(1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
  gl.useProgram(pScene);
  gl.uniformMatrix4fv(ul(pScene,'uVP'), false, vp);
  for (const m of sceneVAOs) { gl.uniformMatrix4fv(ul(pScene,'uM'),false,ID); gl.bindVertexArray(m.vao); gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0); gl.bindVertexArray(null); }
  if (singleProbe >= 0) {
    const pp = probePos[singleProbe];
    const M = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, pp[0],pp[1],pp[2],1]);
    gl.uniformMatrix4fv(ul(pScene,'uM'), false, M);
    gl.bindVertexArray(debugSphereVAO.vao);
    gl.drawElements(gl.TRIANGLES, debugSphereVAO.count, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  } else {
    for (let pi = 0; pi < NUM_PROBES; pi++) {
      const pp = probePos[pi];
      const M = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, pp[0],pp[1],pp[2],1]);
      gl.uniformMatrix4fv(ul(pScene,'uM'), false, M);
      gl.bindVertexArray(debugSphereVAO.vao);
      gl.drawElements(gl.TRIANGLES, debugSphereVAO.count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    }
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // 2. Ray-march → screen (with tone mapping)
  gl.viewport(0,0,w,h);
  gl.clearColor(0.05,0.05,0.05,1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST);

  gl.useProgram(pMarch);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scenePosTex);
  gl.uniform1i(ul(pMarch,'uPos'),0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, sceneNrmTex);
  gl.uniform1i(ul(pMarch,'uNrm'),1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, sceneAlbTex);
  gl.uniform1i(ul(pMarch,'uAlb'),2);
  gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY, irrArr);
  gl.uniform1i(ul(pMarch,'uIrr'),3);
  gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D_ARRAY, vsmArr);
  gl.uniform1i(ul(pMarch,'uVSM'),4);
  gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, distHArr);
  gl.uniform1i(ul(pMarch,'uDistH'),5);
  gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D_ARRAY, distLArr);
  gl.uniform1i(ul(pMarch,'uDistL'),6);

  gl.uniform3fv(ul(pMarch,'uProbePos'), probePosFlat);
  gl.uniform3fv(ul(pMarch,'uGMin'), GMIN);
  gl.uniform3fv(ul(pMarch,'uGMax'), GMAX);
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
  gl.uniform1i(ul(pMarch,'uDebug'), debugMode);
  gl.uniform1i(ul(pMarch,'uSingleProbe'), singleProbe);

  gl.bindVertexArray(qVAO); gl.drawArrays(gl.TRIANGLES, 0, 6); gl.bindVertexArray(null);

  requestAnimationFrame(renderFrame);
}

// ═══════════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════════
resizeSceneBufs(canvas.width, canvas.height);
precompute();
renderFrame(performance.now());
