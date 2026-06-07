import * as shaderSources from './shaders.js';
import { VoxelGrid, sceneBounds, voxelizeScene } from './voxel.js';
import { LightFieldProbeSystem } from './gi/lightFieldProbe.js';
import { TetrahedralProbeSystem } from './gi/tetrahedralProbe.js';

const cubemapResolution = 128;
const octahedralResolution = 256;
const lowResDistance = 64;
const gridSizeX = 4;
const gridSizeY = 4;
const gridSizeZ = 4;
const totalProbeCount = gridSizeX * gridSizeY * gridSizeZ;

const getElementById = document.getElementById.bind(document);
const infoElement = getElementById('info');
const canvas = getElementById('glcanvas');

// ─── WebGL 2.0 Context ───────────────────────────────────────────────
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const gl = canvas.getContext('webgl2', { alpha: false });
if (!gl) {
  infoElement.textContent = 'ERROR: no WebGL 2.0';
  throw Error('');
}
infoElement.textContent = 'WebGL 2.0 OK';
gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');

// ─── Helpers ─────────────────────────────────────────────────────────
function compileShader(source, type) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(vertexSource, fragmentSource) {
  const vertexShader = compileShader(vertexSource, gl.VERTEX_SHADER);
  const fragmentShader = compileShader(fragmentSource, gl.FRAGMENT_SHADER);
  if (!vertexShader || !fragmentShader) {
    return null;
  }
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

function getUniformLocation(program, name) {
  return gl.getUniformLocation(program, name);
}

// ─── Textures ────────────────────────────────────────────────────────
function createCubemapFloat16(width, height) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);
  for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
    gl.texImage2D(
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + faceIndex,
      0,
      gl.RGBA16F,
      width,
      height,
      0,
      gl.RGBA,
      gl.HALF_FLOAT,
      null
    );
  }
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return texture;
}

function createCubemapRedFloat16(width, height) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);
  for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
    gl.texImage2D(
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + faceIndex,
      0,
      gl.R16F,
      width,
      height,
      0,
      gl.RED,
      gl.HALF_FLOAT,
      null
    );
  }
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return texture;
}

function createTexture2D(width, height, channels) {
  const channelCount = channels || 4;
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const internalFormat = channelCount === 1 ? gl.R16F : gl.RGBA16F;
  const format = channelCount === 1 ? gl.RED : gl.RGBA;
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

function createTexture2DArray(width, height, layers, channels, filter) {
  const filterMode = filter || gl.LINEAR;
  const channelCount = channels || 4;
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  let internalFormat;
  if (channelCount === 1) {
    internalFormat = gl.R16F;
  } else if (channelCount === 2) {
    internalFormat = gl.RG32F;
  } else {
    internalFormat = gl.RGBA16F;
  }
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internalFormat, width, height, layers);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, filterMode);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, filterMode);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

// ─── Programs ────────────────────────────────────────────────────────
const programScene = createProgram(shaderSources.sceneGBufVert, shaderSources.sceneGBufFrag);
const programProbe = createProgram(shaderSources.probeGBufVert, shaderSources.probeGBufFrag);
const programOct = createProgram(shaderSources.quadVert, shaderSources.octFrag);
const programIrr = createProgram(shaderSources.quadVert, shaderSources.irrFrag);
const programVSM = createProgram(shaderSources.quadVert, shaderSources.vsmFrag);
const programMarch = createProgram(shaderSources.quadVert, shaderSources.raymarchFrag);
const programSHSphere = createProgram(shaderSources.shSphereVert, shaderSources.shSphereFrag);
const programDirect = createProgram(shaderSources.quadVert, shaderSources.directFrag);
const programForward = createProgram(shaderSources.sceneGBufVert, shaderSources.forwardFrag);
const programProject = createProgram(shaderSources.quadVert, shaderSources.shProjectFrag);
const programTetDebug = createProgram(shaderSources.tetDebugVert, shaderSources.tetDebugFrag);

if (
  !programScene ||
  !programProbe ||
  !programOct ||
  !programIrr ||
  !programVSM ||
  !programMarch ||
  !programSHSphere ||
  !programDirect ||
  !programForward ||
  !programProject ||
  !programTetDebug
) {
  infoElement.textContent = 'Shader error';
  throw Error('shaders');
}
infoElement.textContent = 'Shaders OK';

// ─── Full-screen quad ────────────────────────────────────────────────
const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const quadBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
gl.bufferData(
  gl.ARRAY_BUFFER,
  new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
  gl.STATIC_DRAW
);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// ─── Scene mesh generation ──────────────────────────────────────────
function createBox(centerX, centerY, centerZ, sizeX, sizeY, sizeZ, colorR, colorG, colorB) {
  const halfX = sizeX / 2;
  const halfY = sizeY / 2;
  const halfZ = sizeZ / 2;
  const positions = [
    centerX - halfX, centerY - halfY, centerZ + halfZ,
    centerX + halfX, centerY - halfY, centerZ + halfZ,
    centerX + halfX, centerY + halfY, centerZ + halfZ,
    centerX - halfX, centerY + halfY, centerZ + halfZ,
    centerX + halfX, centerY - halfY, centerZ - halfZ,
    centerX - halfX, centerY - halfY, centerZ - halfZ,
    centerX - halfX, centerY + halfY, centerZ - halfZ,
    centerX + halfX, centerY + halfY, centerZ - halfZ,
    centerX - halfX, centerY - halfY, centerZ - halfZ,
    centerX - halfX, centerY - halfY, centerZ + halfZ,
    centerX - halfX, centerY + halfY, centerZ + halfZ,
    centerX - halfX, centerY + halfY, centerZ - halfZ,
    centerX + halfX, centerY - halfY, centerZ + halfZ,
    centerX + halfX, centerY - halfY, centerZ - halfZ,
    centerX + halfX, centerY + halfY, centerZ - halfZ,
    centerX + halfX, centerY + halfY, centerZ + halfZ,
    centerX - halfX, centerY + halfY, centerZ - halfZ,
    centerX + halfX, centerY + halfY, centerZ - halfZ,
    centerX + halfX, centerY + halfY, centerZ + halfZ,
    centerX - halfX, centerY + halfY, centerZ + halfZ,
    centerX - halfX, centerY - halfY, centerZ + halfZ,
    centerX + halfX, centerY - halfY, centerZ + halfZ,
    centerX + halfX, centerY - halfY, centerZ - halfZ,
    centerX - halfX, centerY - halfY, centerZ - halfZ,
  ];
  const faceNormals = [
    [0, 0, 1], [0, 0, -1], [-1, 0, 0],
    [1, 0, 0], [0, 1, 0], [0, -1, 0],
  ];
  const normals = [];
  for (const faceNormal of faceNormals) {
    for (let vertexIndex = 0; vertexIndex < 4; vertexIndex++) {
      normals.push(faceNormal[0], faceNormal[1], faceNormal[2]);
    }
  }
  const indices = [];
  for (let faceIndex = 0; faceIndex < 6; faceIndex++) {
    const base = faceIndex * 4;
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return {
    pos: positions,
    nrm: normals,
    col: [colorR, colorG, colorB],
    idx: indices,
  };
}

function createSphere(centerX, centerY, centerZ, radius, slats, staves, colorR, colorG, colorB) {
  const positions = [];
  const normals = [];
  for (let latIndex = 0; latIndex <= slats; latIndex++) {
    const theta = latIndex * Math.PI / slats;
    for (let lonIndex = 0; lonIndex <= staves; lonIndex++) {
      const phi = lonIndex * 2 * Math.PI / staves;
      const x = radius * Math.sin(theta) * Math.cos(phi);
      const y = radius * Math.cos(theta);
      const z = radius * Math.sin(theta) * Math.sin(phi);
      positions.push(centerX + x, centerY + y, centerZ + z);
      normals.push(x / radius, y / radius, z / radius);
    }
  }
  const indices = [];
  for (let latIndex = 0; latIndex < slats; latIndex++) {
    for (let lonIndex = 0; lonIndex < staves; lonIndex++) {
      const a = latIndex * (staves + 1) + lonIndex;
      const b = a + staves + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return {
    pos: positions,
    nrm: normals,
    col: [colorR, colorG, colorB],
    idx: indices,
  };
}

function createCone(centerX, centerY, centerZ, radius, height, segments, colorR, colorG, colorB) {
  const positions = [];
  const normals = [];
  const tipX = centerX;
  const tipY = centerY + height / 2;
  const tipZ = centerZ;
  const baseCenterX = centerX;
  const baseCenterY = centerY - height / 2;
  const baseCenterZ = centerZ;
  for (let segmentIndex = 0; segmentIndex <= segments; segmentIndex++) {
    const angle = segmentIndex / segments * 2 * Math.PI;
    const localX = radius * Math.cos(angle);
    const localZ = radius * Math.sin(angle);
    positions.push(baseCenterX + localX, baseCenterY, baseCenterZ + localZ);
    const distance = Math.sqrt(localX * localX + localZ * localZ);
    const normalX = (localX / distance) * height / 2;
    const normalY = radius;
    const normalZ = (localZ / distance) * height / 2;
    const normalLength = Math.sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
    normals.push(normalX / normalLength, normalY / normalLength, normalZ / normalLength);
  }
  positions.push(tipX, tipY, tipZ);
  normals.push(0, 1, 0);
  positions.push(baseCenterX, baseCenterY, baseCenterZ);
  normals.push(0, -1, 0);
  const tipVertexIndex = positions.length / 3 - 2;
  const baseVertexIndex = positions.length / 3 - 1;
  const indices = [];
  for (let segmentIndex = 0; segmentIndex < segments; segmentIndex++) {
    indices.push(segmentIndex, (segmentIndex + 1) % segments, tipVertexIndex);
  }
  for (let segmentIndex = 1; segmentIndex < segments - 1; segmentIndex++) {
    indices.push(baseVertexIndex, segmentIndex, segmentIndex + 1);
  }
  return {
    pos: positions,
    nrm: normals,
    col: [colorR, colorG, colorB],
    idx: indices,
  };
}

function createTorus(centerX, centerY, centerZ, majorRadius, minorRadius, uSegments, vSegments, colorR, colorG, colorB) {
  const positions = [];
  const normals = [];
  for (let vIndex = 0; vIndex <= vSegments; vIndex++) {
    const vAngle = vIndex / vSegments * 2 * Math.PI;
    for (let uIndex = 0; uIndex <= uSegments; uIndex++) {
      const uAngle = uIndex / uSegments * 2 * Math.PI;
      const x = (majorRadius + minorRadius * Math.cos(vAngle)) * Math.cos(uAngle);
      const y = minorRadius * Math.sin(vAngle);
      const z = (majorRadius + minorRadius * Math.cos(vAngle)) * Math.sin(uAngle);
      positions.push(centerX + x, centerY + y, centerZ + z);
      normals.push(
        Math.cos(vAngle) * Math.cos(uAngle),
        Math.sin(vAngle),
        Math.cos(vAngle) * Math.sin(uAngle)
      );
    }
  }
  const indices = [];
  for (let vIndex = 0; vIndex < vSegments; vIndex++) {
    for (let uIndex = 0; uIndex < uSegments; uIndex++) {
      const a = vIndex * (uSegments + 1) + uIndex;
      const b = a + uSegments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return {
    pos: positions,
    nrm: normals,
    col: [colorR, colorG, colorB],
    idx: indices,
  };
}

// ═══════════════════════════════════════════════════════════════════════
//  SCENE MESHES
// ═══════════════════════════════════════════════════════════════════════

const sceneMeshes = [
  // Room
  createBox(0, -4, 0, 10, 0.5, 10, 0.9, 0.9, 0.9),
  createBox(-5, 0, 0, 0.5, 8, 10, 1.0, 0.2, 0.2),
  createBox(5, 0, 0, 0.5, 8, 10, 0.2, 0.2, 1.0),
  createBox(0, 0, -5, 10, 8, 0.5, 0.8, 0.8, 0.8),
  createBox(0, 4, 0, 10, 0.5, 10, 0.6, 0.6, 0.6),
  // Objects
  createBox(-2, -3, 0, 2, 2, 2, 0.1, 0.9, 0.1),
  createBox(2.5, -2.5, -2, 1.5, 3, 1.5, 1.0, 0.6, 0.1),
  createBox(0, 3.5, 0, 4, 0.2, 4, 5.0, 5.0, 5.0),
  createSphere(-1, -0.5, 1.5, 0.8, 24, 16, 0.7, 0.2, 0.2),
  createCone(1, 0, 1.5, 0.7, 1.4, 24, 0.2, 0.4, 0.8),
  createTorus(0, 0, -1.5, 1.0, 0.35, 24, 16, 0.9, 0.7, 0.1),
];

const sceneVAOs = (() => {
  const result = [];
  for (const mesh of sceneMeshes) {
    const vertexCount = mesh.pos.length / 3;
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.pos), gl.STATIC_DRAW);
    const normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.nrm), gl.STATIC_DRAW);
    const colorData = new Float32Array(vertexCount * 3);
    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
      colorData[vertexIndex * 3] = mesh.col[0];
      colorData[vertexIndex * 3 + 1] = mesh.col[1];
      colorData[vertexIndex * 3 + 2] = mesh.col[2];
    }
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colorData, gl.STATIC_DRAW);
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.idx), gl.STATIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bindVertexArray(null);
    result.push({ vao, count: mesh.idx.length });
  }
  return result;
})();

// ─── CPU voxelization ──────────────────────────────────────────────
const sceneBoundsData = sceneBounds(sceneMeshes);
const voxelResolution = 128;
console.log(`Voxelizing at ${voxelResolution}^3...`);
const voxelGrid = voxelizeScene(sceneMeshes, sceneBoundsData, voxelResolution);
const occupiedVoxels = voxelGrid.occ.reduce((a, b) => a + b, 0);
console.log(
  `${occupiedVoxels} / ${voxelResolution ** 3} occupied ` +
  `(${(occupiedVoxels / (voxelResolution ** 3) * 100).toFixed(1)}%)`
);
window.__v = voxelGrid;

// Debug probe sphere VAO (for LFP probe visualization)
const debugSphereMesh = createSphere(0, 0, 0, 0.12, 12, 8, 5, 5, 5);
for (let i = 0; i < debugSphereMesh.nrm.length; i++) {
  debugSphereMesh.nrm[i] = -debugSphereMesh.nrm[i];
}
const debugSphereVAO = (() => {
  const mesh = debugSphereMesh;
  const vertexCount = mesh.pos.length / 3;
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.pos), gl.STATIC_DRAW);
  const normalBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.nrm), gl.STATIC_DRAW);
  const colorData = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    colorData[i * 3] = mesh.col[0];
    colorData[i * 3 + 1] = mesh.col[1];
    colorData[i * 3 + 2] = mesh.col[2];
  }
  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, colorData, gl.STATIC_DRAW);
  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.idx), gl.STATIC_DRAW);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bindVertexArray(null);
  return { vao, count: mesh.idx.length };
})();

// SH test sphere VAO (used in tetrahedral mode for the debug sphere and probe spheres)
const shSphereMesh = createSphere(0, 0, 0, 0.25, 16, 12, 1, 1, 1);
const shSphereVAO = (() => {
  const mesh = shSphereMesh;
  const vertexCount = mesh.pos.length / 3;
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.pos), gl.STATIC_DRAW);
  const normalBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.nrm), gl.STATIC_DRAW);
  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.idx), gl.STATIC_DRAW);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bindVertexArray(null);
  return { vao, count: mesh.idx.length };
})();

// ═══════════════════════════════════════════════════════════════════════
//  GI SYSTEMS
// ═══════════════════════════════════════════════════════════════════════

const gridMin = [-4.0, -5.0, -3.5];
const gridMax = [4.0, 3.0, 2.0];
const maxProbeCount = 256;

const sceneData = {
  sceneVAOs,
  M: sceneMeshes,
  bounds: sceneBoundsData,
  voxels: voxelGrid,
};
const lfpPrograms = {
  pProbe: programProbe,
  pOct: programOct,
  pIrr: programIrr,
  pVSM: programVSM,
};

const lfpSystem = new LightFieldProbeSystem({
  res: cubemapResolution,
  octRes: octahedralResolution,
  lowRes: lowResDistance,
  nx: gridSizeX,
  ny: gridSizeY,
  nz: gridSizeZ,
  gMin: gridMin,
  gMax: gridMax,
  maxProbes: maxProbeCount,
});
lfpSystem.setRenderResources(quadVAO, sceneVAOs);
lfpSystem.init(gl, sceneData, lfpPrograms);

const tetrahedralSystem = new TetrahedralProbeSystem({
  gridX: 3,
  gridY: 3,
  gridZ: 3,
  bounds: { min: [-3.5, -4, -2.5], max: [3.5, 3, 2.5] },
});
tetrahedralSystem.init(gl, sceneData);

// Render cubemaps from each tetrahedral probe position and project to SH
tetrahedralSystem.precompute(gl, {
  pProbe: programProbe,
  pProject: programProject,
  sceneVAOs: sceneVAOs,
  qVAO: quadVAO,
  lightPos: [0, 3.5, 0],
  lightCol: [10, 10, 10],
  cubemapResolution: 64,
  sampleCount: 512,
  info: infoElement,
});

// Pre-compute per-mesh SH for tetrahedral rendering
const centerSHResult = tetrahedralSystem.interpolateAt([0, 0, 0]);
const centerSH = centerSHResult.sh;
const barycentricString = centerSHResult.bary
  ? centerSHResult.bary.map(value => value.toFixed(3)).join(',')
  : 'null';
console.log(
  `[TET] center tetIndex=${centerSHResult.tetIndex} bary=${barycentricString}`
);
console.log(
  `[TET] center SH R0=${centerSH[0].toFixed(3)} ` +
  `G0=${centerSH[1].toFixed(3)} B0=${centerSH[2].toFixed(3)}`
);

const meshCentroids = sceneMeshes.map(mesh => {
  const positions = mesh.pos;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  for (let i = 0; i < positions.length; i += 3) {
    sumX += positions[i];
    sumY += positions[i + 1];
    sumZ += positions[i + 2];
  }
  const count = positions.length / 3;
  return [sumX / count, sumY / count, sumZ / count];
});

const meshSH = sceneMeshes.map((_, meshIndex) => {
  const result = tetrahedralSystem.interpolateAt(meshCentroids[meshIndex]);
  if (result.tetIndex < 0) {
    const centroidString = meshCentroids[meshIndex]
      .map(value => value.toFixed(1))
      .join(',');
    console.log(
      `[TET] mesh[${meshIndex}] centroid (${centroidString}) OUTSIDE — using fallback`
    );
  }
  return result.tetIndex >= 0 ? result.sh : centerSH;
});

// Build tetrahedron wireframe VAO
const wireframePositions = [];
for (const tetrahedron of tetrahedralSystem.tets) {
  const vertexIndices = tetrahedron.vertexIndices;
  const edges = [
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
  ];
  for (const [edgeA, edgeB] of edges) {
    const positionA = tetrahedralSystem.probes[vertexIndices[edgeA]].pos;
    const positionB = tetrahedralSystem.probes[vertexIndices[edgeB]].pos;
    wireframePositions.push(
      positionA[0], positionA[1], positionA[2],
      positionB[0], positionB[1], positionB[2]
    );
  }
}

const wireframeVAO = gl.createVertexArray();
gl.bindVertexArray(wireframeVAO);
const wireframeBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, wireframeBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(wireframePositions), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// Which system is active
let activeSystem = 'tetrahedral';
let lfpReady = false;
let showIndirect = false;
let showDebug = true;

// ═══════════════════════════════════════════════════════════════════════
//  PRECOMPUTATION
// ═══════════════════════════════════════════════════════════════════════

function precompute() {
  infoElement.textContent = `${tetrahedralSystem.probes.length} tetrahedral probes ready.`;
}

// ═══════════════════════════════════════════════════════════════════════
//  RENDERING
// ═══════════════════════════════════════════════════════════════════════

let scenePositionTexture;
let sceneNormalTexture;
let sceneAlbedoTexture;
let sceneDepthRenderbuffer;
let sceneFBO;

function resizeSceneBuffers(width, height) {
  if (scenePositionTexture) {
    gl.deleteTexture(scenePositionTexture);
    gl.deleteTexture(sceneNormalTexture);
    gl.deleteTexture(sceneAlbedoTexture);
    gl.deleteRenderbuffer(sceneDepthRenderbuffer);
    gl.deleteFramebuffer(sceneFBO);
  }
  scenePositionTexture = createTexture2D(width, height);
  sceneNormalTexture = createTexture2D(width, height);
  sceneAlbedoTexture = createTexture2D(width, height);
  sceneDepthRenderbuffer = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepthRenderbuffer);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
  sceneFBO = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, scenePositionTexture, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, sceneNormalTexture, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, sceneAlbedoTexture, 0);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, sceneDepthRenderbuffer);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    console.error('Scene FBO incomplete');
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ─── Camera controls ────────────────────────────────────────────────
let cameraPosition = [0, 1, 8];
let cameraYaw = 0;
let cameraPitch = 0;
const pressedKeys = new Set();
const cameraSpeed = 4;

let singleProbe = -1;
let probeCycle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

function updateInfoText() {
  const probeSuffix = singleProbe >= 0 ? ` [probe:${singleProbe}]` : '';
  const systemLabel = activeSystem === 'lightfield' ? 'LFP' : 'TET';
  const indirectFlag = showIndirect ? ' [IRR]' : '';
  const debugFlag = showDebug ? ' [DBG]' : '';
  const totalProbes = tetrahedralSystem.probes.length;
  const totalTets = tetrahedralSystem.tets.length;
  if (document.pointerLockElement === canvas) {
    infoElement.textContent = 'Click to release mouse';
  } else {
    infoElement.textContent =
      `${systemLabel} | ${totalProbes} probes ${totalTets} tets` +
      probeSuffix + indirectFlag + debugFlag;
  }
}

canvas.addEventListener('click', () => canvas.requestPointerLock());
document.addEventListener('pointerlockchange', updateInfoText);

document.addEventListener('mousemove', event => {
  if (document.pointerLockElement !== canvas) {
    return;
  }
  cameraYaw += event.movementX * 0.002;
  cameraPitch = Math.max(
    -1.5,
    Math.min(1.5, cameraPitch - event.movementY * 0.002)
  );
});

document.addEventListener('keydown', event => {
  pressedKeys.add(event.code);
  const digit = parseInt(event.code.replace('Digit', ''));
  const isDigit = !isNaN(digit) && digit >= 0 && digit <= 9;

  if (isDigit && activeSystem === 'lightfield') {
    const maxOffset = Math.ceil((totalProbeCount - 1 - digit) / 10);
    const currentOffset = probeCycle[digit];
    singleProbe = currentOffset * 10 + digit;
    lfpSystem.singleProbe = singleProbe;
    probeCycle[digit] = (currentOffset + 1) > maxOffset ? 0 : currentOffset + 1;
  } else if (event.code === 'KeyR') {
    singleProbe = -1;
    lfpSystem.singleProbe = -1;
  } else if (event.code === 'KeyT') {
    activeSystem = activeSystem === 'lightfield' ? 'tetrahedral' : 'lightfield';
    singleProbe = -1;
    lfpSystem.singleProbe = -1;
    console.log(`[GI] Switched to ${activeSystem}`);
  } else if (event.code === 'KeyI') {
    showIndirect = !showIndirect;
    console.log(`[TET] showIndirect=${showIndirect}`);
  } else if (event.code === 'KeyU') {
    showDebug = !showDebug;
    console.log(`[TET] showDebug=${showDebug}`);
  }
  updateInfoText();
});

document.addEventListener('keyup', event => {
  pressedKeys.delete(event.code);
});

function updateCamera(deltaTime) {
  const cosYaw = Math.cos(cameraYaw);
  const sinYaw = Math.sin(cameraYaw);
  const cosPitch = Math.cos(cameraPitch);
  const sinPitch = Math.sin(cameraPitch);
  const forward = [cosYaw * cosPitch, sinPitch, sinYaw * cosPitch];
  const right = [-sinYaw, 0, cosYaw];
  let speed = cameraSpeed * deltaTime;
  if (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) {
    speed *= 3;
  }
  if (pressedKeys.has('KeyW') || pressedKeys.has('ArrowUp')) {
    cameraPosition = addVectors(cameraPosition, scaleVector(forward, speed));
  }
  if (pressedKeys.has('KeyS') || pressedKeys.has('ArrowDown')) {
    cameraPosition = addVectors(cameraPosition, scaleVector(forward, -speed));
  }
  if (pressedKeys.has('KeyA') || pressedKeys.has('ArrowLeft')) {
    cameraPosition = addVectors(cameraPosition, scaleVector(right, -speed));
  }
  if (pressedKeys.has('KeyD') || pressedKeys.has('ArrowRight')) {
    cameraPosition = addVectors(cameraPosition, scaleVector(right, speed));
  }
  if (pressedKeys.has('Space')) {
    cameraPosition[1] += speed;
  }
  if (pressedKeys.has('KeyZ')) {
    cameraPosition[1] -= speed;
  }
}

function addVectors(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scaleVector(vector, scalar) {
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
}

// SH test sphere orbit
let orbitAngle = 0;
const orbitRadius = 0.6;
const orbitSpeed = 0.6;

let previousTime = 0;

function renderFrame(timestamp) {
  const deltaTime = Math.min((timestamp - previousTime) / 1000, 0.05);
  previousTime = timestamp;
  updateCamera(deltaTime);

  const pixelWidth = canvas.clientWidth * devicePixelRatio;
  const pixelHeight = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    resizeSceneBuffers(pixelWidth, pixelHeight);
  }

  const projectionMatrix = perspectiveMatrix(
    Math.PI / 3,
    pixelWidth / pixelHeight,
    0.1,
    20
  );
  const viewMatrix = lookAtMatrix(
    cameraPosition,
    addVectors(cameraPosition, [
      Math.cos(cameraYaw) * Math.cos(cameraPitch),
      Math.sin(cameraPitch),
      Math.sin(cameraYaw) * Math.cos(cameraPitch),
    ]),
    [0, 1, 0]
  );
  const viewProjectionMatrix = multiplyMatrices4(projectionMatrix, viewMatrix);

  // ── 1. Forward render (tetrahedral mode) ──────────────────────
  if (activeSystem === 'tetrahedral') {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, pixelWidth, pixelHeight);
    gl.clearColor(0.05, 0.05, 0.05, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    gl.useProgram(programForward);
    gl.uniformMatrix4fv(
      getUniformLocation(programForward, 'uVP'),
      false,
      viewProjectionMatrix
    );
    gl.uniform3fv(getUniformLocation(programForward, 'uLightPos'), [0, 3.5, 0]);
    gl.uniform3fv(getUniformLocation(programForward, 'uLightCol'), [10, 10, 10]);
    gl.uniform1f(
      getUniformLocation(programForward, 'uShowIndirect'),
      showIndirect ? 1 : 0
    );

    for (let meshIndex = 0; meshIndex < sceneMeshes.length; meshIndex++) {
      gl.uniform3fv(
        getUniformLocation(programForward, 'uSH'),
        meshSH[meshIndex]
      );
      gl.uniformMatrix4fv(
        getUniformLocation(programForward, 'uM'),
        false,
        identityMatrix
      );
      gl.bindVertexArray(sceneVAOs[meshIndex].vao);
      gl.drawElements(
        gl.TRIANGLES,
        sceneVAOs[meshIndex].count,
        gl.UNSIGNED_SHORT,
        0
      );
      gl.bindVertexArray(null);
    }

    // Debug visualization: wireframe + probe spheres
    if (showDebug) {
      gl.useProgram(programTetDebug);
      gl.uniformMatrix4fv(
        getUniformLocation(programTetDebug, 'uVP'),
        false,
        viewProjectionMatrix
      );
      gl.uniformMatrix4fv(
        getUniformLocation(programTetDebug, 'uM'),
        false,
        identityMatrix
      );
      gl.uniform1f(getUniformLocation(programTetDebug, 'uPointSize'), 1);
      gl.uniform3fv(
        getUniformLocation(programTetDebug, 'uColor'),
        [0.3, 0.5, 1.0]
      );
      gl.bindVertexArray(wireframeVAO);
      gl.drawArrays(gl.LINES, 0, wireframePositions.length / 3);
      gl.bindVertexArray(null);

      // Probe spheres (show SH indirect colour)
      gl.useProgram(programSHSphere);
      gl.uniformMatrix4fv(
        getUniformLocation(programSHSphere, 'uVP'),
        false,
        viewProjectionMatrix
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uLightPos'),
        [0, 3.5, 0]
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uLightCol'),
        [0, 0, 0]
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uBaseCol'),
        [1, 1, 1]
      );
      gl.depthMask(true);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CW);

      for (let probeIndex = 0; probeIndex < tetrahedralSystem.probes.length; probeIndex++) {
        const probePosition = tetrahedralSystem.probes[probeIndex].pos;
        const scaleFactor = 0.4;
        const modelMatrix = new Float32Array([
          scaleFactor, 0, 0, 0,
          0, scaleFactor, 0, 0,
          0, 0, scaleFactor, 0,
          probePosition[0], probePosition[1], probePosition[2], 1,
        ]);
        gl.uniformMatrix4fv(
          getUniformLocation(programSHSphere, 'uM'),
          false,
          modelMatrix
        );
        gl.uniform3fv(
          getUniformLocation(programSHSphere, 'uSH'),
          tetrahedralSystem.probes[probeIndex].sh
        );
        gl.bindVertexArray(shSphereVAO.vao);
        gl.drawElements(
          gl.TRIANGLES,
          shSphereVAO.count,
          gl.UNSIGNED_SHORT,
          0
        );
        gl.bindVertexArray(null);
      }
      gl.disable(gl.CULL_FACE);
    }

    gl.disable(gl.DEPTH_TEST);

  } else {
    // ── 1. Scene G-buffer (LFP mode) ──────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO);
    gl.viewport(0, 0, pixelWidth, pixelHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(programScene);
    gl.uniformMatrix4fv(
      getUniformLocation(programScene, 'uVP'),
      false,
      viewProjectionMatrix
    );

    for (const vaoEntry of sceneVAOs) {
      gl.uniformMatrix4fv(
        getUniformLocation(programScene, 'uM'),
        false,
        identityMatrix
      );
      gl.bindVertexArray(vaoEntry.vao);
      gl.drawElements(gl.TRIANGLES, vaoEntry.count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    }

    if (singleProbe >= 0) {
      const probePosition = lfpSystem.adjProbePos[singleProbe];
      const modelMatrix = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        probePosition[0], probePosition[1], probePosition[2], 1,
      ]);
      gl.uniformMatrix4fv(
        getUniformLocation(programScene, 'uM'),
        false,
        modelMatrix
      );
      gl.bindVertexArray(debugSphereVAO.vao);
      gl.drawElements(gl.TRIANGLES, debugSphereVAO.count, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    } else {
      for (let probeIndex = 0; probeIndex < totalProbeCount; probeIndex++) {
        const probePosition = lfpSystem.adjProbePos[probeIndex];
        const modelMatrix = new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          probePosition[0], probePosition[1], probePosition[2], 1,
        ]);
        gl.uniformMatrix4fv(
          getUniformLocation(programScene, 'uM'),
          false,
          modelMatrix
        );
        gl.bindVertexArray(debugSphereVAO.vao);
        gl.drawElements(
          gl.TRIANGLES,
          debugSphereVAO.count,
          gl.UNSIGNED_SHORT,
          0
        );
        gl.bindVertexArray(null);
      }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── 2. Background pass (LFP mode) ──────────────────────────────
    gl.viewport(0, 0, pixelWidth, pixelHeight);
    gl.clearColor(0.05, 0.05, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);

    if (lfpReady) {
      gl.useProgram(programMarch);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, scenePositionTexture);
      gl.uniform1i(getUniformLocation(programMarch, 'uPos'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, sceneNormalTexture);
      gl.uniform1i(getUniformLocation(programMarch, 'uNrm'), 1);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, sceneAlbedoTexture);
      gl.uniform1i(getUniformLocation(programMarch, 'uAlb'), 2);
      lfpSystem.bindUniforms(gl, programMarch);
      {
        const cosYaw = Math.cos(cameraYaw);
        const sinYaw = Math.sin(cameraYaw);
        const cosPitch = Math.cos(cameraPitch);
        const sinPitch = Math.sin(cameraPitch);
        const forward = [cosYaw * cosPitch, sinPitch, sinYaw * cosPitch];
        const up = [-cosYaw * sinPitch, cosPitch, -sinYaw * sinPitch];
        const right = [-sinYaw, 0, cosYaw];
        gl.uniform3fv(getUniformLocation(programMarch, 'uCamFwd'), forward);
        gl.uniform3fv(getUniformLocation(programMarch, 'uCamUp'), up);
        gl.uniform3fv(getUniformLocation(programMarch, 'uCamRight'), right);
      }
      gl.uniform1f(
        getUniformLocation(programMarch, 'uTanHalfFov'),
        Math.tan(Math.PI / 6)
      );
      gl.uniform1f(
        getUniformLocation(programMarch, 'uAspect'),
        pixelWidth / pixelHeight
      );
      gl.uniform1f(getUniformLocation(programMarch, 'uNormalBias'), 0.03);
      gl.uniform1f(getUniformLocation(programMarch, 'uDistBias'), 0.02);
      gl.uniform1i(getUniformLocation(programMarch, 'uDebug'), 0);
      gl.uniform1i(
        getUniformLocation(programMarch, 'uSingleProbe'),
        singleProbe
      );
      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    } else {
      gl.useProgram(programDirect);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, scenePositionTexture);
      gl.uniform1i(getUniformLocation(programDirect, 'uPos'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, sceneNormalTexture);
      gl.uniform1i(getUniformLocation(programDirect, 'uNrm'), 1);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, sceneAlbedoTexture);
      gl.uniform1i(getUniformLocation(programDirect, 'uAlb'), 2);
      gl.uniform3fv(
        getUniformLocation(programDirect, 'uLightPos'),
        [0, 3.5, 0]
      );
      gl.uniform3fv(
        getUniformLocation(programDirect, 'uLightCol'),
        [10, 10, 10]
      );
      gl.uniform3fv(
        getUniformLocation(programDirect, 'uAmbient'),
        [0.03, 0.04, 0.06]
      );
      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    }
  }

  // ── 2. Tetrahedral SH sphere overlay ─────────────────────────────
  if (activeSystem === 'tetrahedral') {
    orbitAngle += deltaTime * orbitSpeed;
    const testPosition = [
      orbitRadius * Math.cos(orbitAngle),
      0.0,
      orbitRadius * Math.sin(orbitAngle),
    ];

    const interpolationResult = tetrahedralSystem.interpolateAt(testPosition);
    if (interpolationResult.tetIndex >= 0) {
      // Log periodically (not every frame)
      const currentTick = Math.floor(orbitAngle / (Math.PI * 2) * 10);
      const previousTick = Math.floor(
        (orbitAngle - deltaTime * orbitSpeed) / (Math.PI * 2) * 10
      );
      if (currentTick !== previousTick) {
        const baryString = interpolationResult.bary
          .map(value => value.toFixed(2))
          .join(',');
        const posString = testPosition
          .map(value => value.toFixed(2))
          .join(',');
        console.log(
          `[TET] tet=${interpolationResult.tetIndex} ` +
          `bary=(${baryString}) pos=(${posString}) ` +
          `sh_R0=${interpolationResult.sh[0].toFixed(3)} ` +
          `G0=${interpolationResult.sh[1].toFixed(3)} ` +
          `B0=${interpolationResult.sh[2].toFixed(3)}`
        );
      }

      const modelMatrix = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        testPosition[0], testPosition[1], testPosition[2], 1,
      ]);
      gl.useProgram(programSHSphere);
      gl.uniformMatrix4fv(
        getUniformLocation(programSHSphere, 'uVP'),
        false,
        viewProjectionMatrix
      );
      gl.uniformMatrix4fv(
        getUniformLocation(programSHSphere, 'uM'),
        false,
        modelMatrix
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uSH'),
        interpolationResult.sh
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uLightPos'),
        [0, 3.5, 0]
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uLightCol'),
        [10, 10, 10]
      );
      gl.uniform3fv(
        getUniformLocation(programSHSphere, 'uBaseCol'),
        [0.8, 0.7, 0.6]
      );
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
function perspectiveMatrix(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAtMatrix(eye, target, up) {
  const zAxis = normalizeVector(subtractVectors(eye, target));
  const xAxis = normalizeVector(crossProduct(up, zAxis));
  const yAxis = crossProduct(zAxis, xAxis);
  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dotProduct(xAxis, eye), -dotProduct(yAxis, eye), -dotProduct(zAxis, eye), 1,
  ]);
}

function normalizeVector(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function subtractVectors(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function crossProduct(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dotProduct(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function multiplyMatrices4(a, b) {
  const result = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      result[col * 4 + row] =
        a[row] * b[col * 4] +
        a[4 + row] * b[col * 4 + 1] +
        a[8 + row] * b[col * 4 + 2] +
        a[12 + row] * b[col * 4 + 3];
    }
  }
  return result;
}

const identityMatrix = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

// Also reference identityMatrix as ID for compatibility with GI system internals
const ID = identityMatrix;

// ═══════════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════════
resizeSceneBuffers(canvas.width, canvas.height);
precompute();
renderFrame(performance.now());
