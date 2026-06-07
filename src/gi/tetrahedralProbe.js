import { IndirectLightingSystem } from './interface.js';

/**
 * TetrahedralProbeSystem — CPU tetrahedral search + SH interpolation
 * based on Robert Cupisz's "Light Probe Interpolation Using Tetrahedral
 * Tessellations" (GDC 2012).
 *
 * Each frame, for each dynamic object, finds the enclosing tetrahedron
 * via a neighbour-walk search, computes barycentric coordinates from a
 * precomputed 3x3 matrix, and interpolates the 4 corner probes' SH.
 *
 * Usage:
 *   const gi = new TetrahedralProbeSystem();
 *   gi.init(gl, sceneData);
 *   // ... each frame for each object ...
 *   gi.interpolateAt(objectPos);   // CPU — returns { sh, tetIndex, bary }
 *   gi.bindInterpolated(gl, program, 'uSH');  // upload as 9×vec3 uniform
 */

// ─── SH helpers ─────────────────────────────────────────────────────────
// 3-band RGB SH = 27 floats
const SH_FLOATS = 27;

function zeroSphericalHarmonics() {
  return new Float32Array(SH_FLOATS);
}

function copySphericalHarmonics(source) {
  const destination = new Float32Array(SH_FLOATS);
  destination.set(source);
  return destination;
}

/** Add source multiplied by weight into destination (destination += source * weight) */
function accumulateSphericalHarmonics(destination, source, weight) {
  for (let index = 0; index < SH_FLOATS; index++) {
    destination[index] += source[index] * weight;
  }
}

/** 3×3 matrix inverse (for precomputed tet matrices) */
function inverseMatrix3x3(matrix) {
  const [a, b, c, d, e, f, g, h, i] = matrix;
  const determinant =
    a * (e * i - f * h) -
    b * (d * i - f * g) +
    c * (d * h - e * g);
  if (Math.abs(determinant) < 1e-12) {
    return null;
  }
  const inverseDeterminant = 1 / determinant;
  return [
    (e * i - f * h) * inverseDeterminant,
    -(b * i - c * h) * inverseDeterminant,
    (b * f - c * e) * inverseDeterminant,
    -(d * i - f * g) * inverseDeterminant,
    (a * i - c * g) * inverseDeterminant,
    -(a * f - c * d) * inverseDeterminant,
    (d * h - e * g) * inverseDeterminant,
    -(a * h - b * g) * inverseDeterminant,
    (a * e - b * d) * inverseDeterminant,
  ];
}

/**
 * Precompute the tetrahedron matrix M such that:
 *   [λ₁, λ₂, λ₃] = M · (P - v0)
 *   λ₀ = 1 - λ₁ - λ₂ - λ₃
 *
 * M is the inverse of the 3×3 matrix whose columns are (v1 - v0, v2 - v0, v3 - v0).
 */
function computeTetrahedronMatrix(v0, v1, v2, v3) {
  const edge1 = [
    v1[0] - v0[0],
    v1[1] - v0[1],
    v1[2] - v0[2],
  ];
  const edge2 = [
    v2[0] - v0[0],
    v2[1] - v0[1],
    v2[2] - v0[2],
  ];
  const edge3 = [
    v3[0] - v0[0],
    v3[1] - v0[1],
    v3[2] - v0[2],
  ];
  // Matrix stored column-major: columns are edge1, edge2, edge3
  const columnMajorMatrix = [
    edge1[0], edge1[1], edge1[2],
    edge2[0], edge2[1], edge2[2],
    edge3[0], edge3[1], edge3[2],
  ];
  return inverseMatrix3x3(columnMajorMatrix);
}

/**
 * Compute barycentric coordinates using precomputed tetrahedron matrix.
 * Returns [λ₀, λ₁, λ₂, λ₃] or null if degenerate.
 */
function barycentricFromMatrix(worldPosition, vertex0, matrix) {
  const deltaX = worldPosition[0] - vertex0[0];
  const deltaY = worldPosition[1] - vertex0[1];
  const deltaZ = worldPosition[2] - vertex0[2];
  const lambda1 =
    matrix[0] * deltaX + matrix[3] * deltaY + matrix[6] * deltaZ;
  const lambda2 =
    matrix[1] * deltaX + matrix[4] * deltaY + matrix[7] * deltaZ;
  const lambda3 =
    matrix[2] * deltaX + matrix[5] * deltaY + matrix[8] * deltaZ;
  const lambda0 = 1 - lambda1 - lambda2 - lambda3;
  return [lambda0, lambda1, lambda2, lambda3];
}

// ═══════════════════════════════════════════════════════════════════════════
//  TETRAHEDRAL PROBE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

export class TetrahedralProbeSystem extends IndirectLightingSystem {
  constructor(cfg) {
    super();
    cfg = cfg || {};
    this.gridX = cfg.gridX || 2;
    this.gridY = cfg.gridY || 2;
    this.gridZ = cfg.gridZ || 2;
    this.bounds = cfg.bounds || { min: [-3.5, -4, -2.5], max: [3.5, 3, 2.5] };

    /** @type {{pos:number[], sh:Float32Array}[]} */
    this.probes = [];

    /**
     * @type {{
     *   vertexIndices:number[],       // 4 vertex indices
     *   neighbourIndices:number[],    // 4 neighbour indices
     *   precomputedMatrix:number[]|null,  // 3×3 matrix (9 floats)
     * }[]}
     */
    this.tets = [];

    // Cache for temporal coherence
    this._tetrahedronIndex = 0;
    this._tetrahedronIndexDirty = true;
    this.lastBarycentric = null;
    this.lastTetrahedronIndex = -1;
  }

  // ─── Initialisation ────────────────────────────────────────────────────

  init(gl, sceneData) {
    this._buildGrid();
    console.log(`[TET] ${this.probes.length} probes, ${this.tets.length} tetrahedra`);
  }

  /**
   * Build a regular grid of tetrahedra covering the scene volume.
   * Each grid cell is a box split into 6 tetrahedra (diagonal decomposition).
   */
  _buildGrid() {
    const { gridX, gridY, gridZ, bounds } = this;
    const nx = gridX + 1, ny = gridY + 1, nz = gridZ + 1;
    const bx = bounds.min, ex = bounds.max;
    const sx = (ex[0] - bx[0]) / gridX;
    const sy = (ex[1] - bx[1]) / gridY;
    const sz = (ex[2] - bx[2]) / gridZ;

    // Create probes at grid vertices
    const indexMap = (ix, iy, iz) => iz * (nx * ny) + iy * nx + ix;
    const numProbes = nx * ny * nz;
    this.probes = new Array(numProbes);
    for (let iz = 0; iz < nz; iz++)
      for (let iy = 0; iy < ny; iy++)
        for (let ix = 0; ix < nx; ix++)
          this.probes[indexMap(ix, iy, iz)] = {
            pos: [bx[0] + ix * sx, bx[1] + iy * sy, bx[2] + iz * sz],
            sh: zeroSphericalHarmonics(),
          };

    // Build all tetrahedron definitions (6 per cell)
    const tetrahedronDefs = [];
    const createCellTetrahedra = (vertexIndices) => [
      { vertexIndices: [vertexIndices[0], vertexIndices[1], vertexIndices[3], vertexIndices[7]] },
      { vertexIndices: [vertexIndices[0], vertexIndices[1], vertexIndices[5], vertexIndices[7]] },
      { vertexIndices: [vertexIndices[0], vertexIndices[2], vertexIndices[3], vertexIndices[7]] },
      { vertexIndices: [vertexIndices[0], vertexIndices[2], vertexIndices[6], vertexIndices[7]] },
      { vertexIndices: [vertexIndices[0], vertexIndices[4], vertexIndices[5], vertexIndices[7]] },
      { vertexIndices: [vertexIndices[0], vertexIndices[4], vertexIndices[6], vertexIndices[7]] },
    ];

    for (let iz = 0; iz < gridZ; iz++)
      for (let iy = 0; iy < gridY; iy++)
        for (let ix = 0; ix < gridX; ix++) {
          const cellVertexIndices = [
            indexMap(ix,   iy,   iz),
            indexMap(ix+1, iy,   iz),
            indexMap(ix,   iy+1, iz),
            indexMap(ix+1, iy+1, iz),
            indexMap(ix,   iy,   iz+1),
            indexMap(ix+1, iy,   iz+1),
            indexMap(ix,   iy+1, iz+1),
            indexMap(ix+1, iy+1, iz+1),
          ];
          tetrahedronDefs.push(...createCellTetrahedra(cellVertexIndices));
        }

    // Neighbour adjacency
    const findOpposite = (definitions, definitionIndex, vertexPosition) => {
      const faceVertices = definitions[definitionIndex].vertexIndices
        .filter((_, i) => i !== vertexPosition);
      for (let neighbourIndex = 0; neighbourIndex < definitions.length; neighbourIndex++) {
        if (neighbourIndex === definitionIndex) {
          continue;
        }
        const otherVertices = definitions[neighbourIndex].vertexIndices;
        if (faceVertices.every(vertex => otherVertices.includes(vertex))) {
          return neighbourIndex;
        }
      }
      return -1;
    };

    this.tets = tetrahedronDefs.map((definition, definitionIndex) => {
      const neighbourIndices = definition.vertexIndices
        .map((_, vertexPosition) => findOpposite(tetrahedronDefs, definitionIndex, vertexPosition));
      const probePositions = this.probes;
      const v0 = probePositions[definition.vertexIndices[0]].pos;
      const v1 = probePositions[definition.vertexIndices[1]].pos;
      const v2 = probePositions[definition.vertexIndices[2]].pos;
      const v3 = probePositions[definition.vertexIndices[3]].pos;
      const precomputedMatrix = computeTetrahedronMatrix(v0, v1, v2, v3);
      return {
        vertexIndices: [...definition.vertexIndices],
        neighbourIndices,
        precomputedMatrix,
      };
    });
  }

  // ─── Tetrahedral search (CPU, every frame) ────────────────────────────

  /**
   * Find the tetrahedron containing world-space position P.
   *
   * Uses a neighbour-walk starting from the cached tetrahedronIndex for
   * temporal coherence.  Returns the tetrahedron index, or -1 if outside
   * the entire mesh.
   *
   * Algorithm:
   *   1. Compute barycentric coords via precomputed matrix
   *   2. If all λ ≥ 0 → inside; return
   *   3. Otherwise, step through the neighbour of the most-negative λ
   *   4. Repeat (bounded by max iteration = num tetrahedra)
   */
  findTetrahedron(worldPosition) {
    const tetrahedra = this.tets;
    if (tetrahedra.length === 0) {
      return -1;
    }

    // Start from cached index (or 0 on first call)
    let tetrahedronIndex = this._tetrahedronIndexDirty
      ? 0
      : this._tetrahedronIndex;
    this._tetrahedronIndexDirty = false;

    const maxIterations = tetrahedra.length + 1;
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const tetrahedron = tetrahedra[tetrahedronIndex];
      if (!tetrahedron.precomputedMatrix) {
        tetrahedronIndex = (tetrahedronIndex + 1) % tetrahedra.length;
        continue;
      }

      const vertex0 = this.probes[tetrahedron.vertexIndices[0]].pos;
      const barycentric = barycentricFromMatrix(
        worldPosition,
        vertex0,
        tetrahedron.precomputedMatrix
      );

      let isInside = true;
      let mostNegativeValue = 0;
      let mostNegativeIndex = -1;

      for (let coordIndex = 0; coordIndex < 4; coordIndex++) {
        if (barycentric[coordIndex] < -1e-7) {
          isInside = false;
          if (barycentric[coordIndex] < mostNegativeValue) {
            mostNegativeValue = barycentric[coordIndex];
            mostNegativeIndex = coordIndex;
          }
        }
      }

      if (isInside) {
        this._tetrahedronIndex = tetrahedronIndex;
        this.lastTetrahedronIndex = tetrahedronIndex;
        this.lastBarycentric = barycentric;
        return tetrahedronIndex;
      }

      // Walk to neighbour across the face opposite the most negative coordinate
      const nextIndex = tetrahedron.neighbourIndices[mostNegativeIndex];
      if (nextIndex < 0 || nextIndex >= tetrahedra.length) {
        this.lastTetrahedronIndex = -1;
        this.lastBarycentric = null;
        return -1;
      }
      tetrahedronIndex = nextIndex;
    }

    console.warn('[TET] Search exceeded max iterations');
    this.lastTetrahedronIndex = -1;
    this.lastBarycentric = null;
    return -1;
  }

  // ─── SH interpolation ──────────────────────────────────────────────────

  /**
   * Interpolate SH at world-space position P.
   *
   * Finds the enclosing tetrahedron, computes barycentric weights,
   * and accumulates the 4 corner probes' SH.
   *
   * Returns { sh: Float32Array(27), tetIndex: number, bary: number[]|null }
   * On failure, sh is zeros and tetIndex is -1.
   */
  interpolateAt(worldPosition) {
    const tetrahedronIndex = this.findTetrahedron(worldPosition);

    if (tetrahedronIndex < 0) {
      return {
        sh: zeroSphericalHarmonics(),
        tetIndex: -1,
        bary: null,
      };
    }

    const tetrahedron = this.tets[tetrahedronIndex];
    const vertex0 = this.probes[tetrahedron.vertexIndices[0]].pos;
    const barycentric = barycentricFromMatrix(
      worldPosition,
      vertex0,
      tetrahedron.precomputedMatrix
    );

    const sphericalHarmonics = zeroSphericalHarmonics();
    for (let vertexIndex = 0; vertexIndex < 4; vertexIndex++) {
      accumulateSphericalHarmonics(
        sphericalHarmonics,
        this.probes[tetrahedron.vertexIndices[vertexIndex]].sh,
        barycentric[vertexIndex]
      );
    }

    return {
      sh: sphericalHarmonics,
      tetIndex: tetrahedronIndex,
      bary: barycentric,
    };
  }

  // ─── GL binding ────────────────────────────────────────────────────────

  /**
   * Upload a single interpolated SH array as two uniforms:
   *   uSH — vec3[9] (9 RGB coefficients)
   *
   * The companion shader can evaluate:
   *   vec3 evalSH(vec3 dir) { ... }
   */
  bindInterpolated(gl, program, shArray) {
    gl.uniform3fv(gl.getUniformLocation(program, 'uSH'), shArray);
  }

  // ─── Precomputation (render cubemaps + project to SH) ────────────────

  /**
   * Render cubemaps from each probe position and project radiance into
   * SH coefficients using the GPU.
   *
   * ctx: { gl, pProbe, pProject, sceneVAOs, qVAO, lightPos, lightCol, res }
   */
  precompute(gl, ctx) {
    const { pProbe, pProject, sceneVAOs, qVAO, cubemapResolution } = ctx;
    const lPos = ctx.lightPos || [0, 3.5, 0];
    const lCol = ctx.lightCol || [10, 10, 10];
    const sampleCount = ctx.sampleCount || 1024;

    const persp = _persp(Math.PI/2, 1, 0.1, 20);
    const blk = new Float32Array([0,0,0,1]);
    const farD = new Float32Array([20,0,0,0]);
    const cT = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    const cU = [[0,-1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,-1,0],[0,-1,0]];
    const ID = new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

    // Output texture: 9×1 RGBA32F (one pixel per SH band)
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 9, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const outFBO = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    const readBuf = new Float32Array(36);

    for (let pi = 0; pi < this.probes.length; pi++) {
      const pp = this.probes[pi].pos;

      if (ctx.info) ctx.info.textContent = `Tet probe ${pi+1}/${this.probes.length}...`;

      // Render cubemap from this probe position
      const gb = _allocCM(gl, cubemapResolution);
      gl.useProgram(pProbe);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uQ'), pp);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uLightPos'), lPos);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uLightCol'), lCol);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);

      for (let f = 0; f < 6; f++) {
        _bindCMFace(gl, gb, f);
        const vp = _mul4(persp, _lookAt(pp, [pp[0]+cT[f][0],pp[1]+cT[f][1],pp[2]+cT[f][2]], cU[f]));
        gl.uniformMatrix4fv(gl.getUniformLocation(pProbe, 'uVP'), false, vp);
        gl.viewport(0, 0, cubemapResolution, cubemapResolution);
        gl.clearBufferfv(gl.COLOR, 0, blk);
        gl.clearBufferfv(gl.COLOR, 1, blk);
        gl.clearBufferfv(gl.COLOR, 2, farD);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        for (const m of sceneVAOs) {
          gl.uniformMatrix4fv(gl.getUniformLocation(pProbe,'uM'), false, ID);
          gl.bindVertexArray(m.vao);
          gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
          gl.bindVertexArray(null);
        }
      }

      // Project cubemap to SH coefficients
      gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
      gl.viewport(0, 0, 9, 1);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(pProject);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.rad);
      gl.uniform1i(gl.getUniformLocation(pProject, 'uRad'), 0);
      gl.uniform1i(gl.getUniformLocation(pProject, 'uSampleCount'), sampleCount);
      gl.bindVertexArray(qVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);

      // Read back SH coefficients
      gl.readPixels(0, 0, 9, 1, gl.RGBA, gl.FLOAT, readBuf);

      // Convert readback to interleaved SH layout
      const sh = zeroSphericalHarmonics();
      for (let band = 0; band < 9; band++) {
        sh[band * 3]     = readBuf[band * 4];     // Red channel
        sh[band * 3 + 1] = readBuf[band * 4 + 1]; // Green channel
        sh[band * 3 + 2] = readBuf[band * 4 + 2]; // Blue channel
      }
      this.probes[pi].sh = sh;

      _freeCM(gl, gb);
    }

    // Cleanup
    gl.deleteFramebuffer(outFBO);
    gl.deleteTexture(outTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (ctx.info) ctx.info.textContent = `Tet SH done.`;
    console.log(`[TET] Precomputed SH for ${this.probes.length} probes`);
  }

  bindUniforms(gl, program, camera) {
    // Tetrahedral system doesn't bind global uniforms for raymarching;
    // per-object SH is bound via bindInterpolated() instead.
  }

  destroy(gl) {
    // No GPU resources
  }

  // ─── Debug ─────────────────────────────────────────────────────────────

  /**
   * Return the cached tet index for the last interpolateAt() call.
   * Useful for overlay / debug rendering.
   */
  getDebugInfo() {
    return {
      tetrahedronIndex: this.lastTetrahedronIndex,
      barycentric: this.lastBarycentric,
      numberOfTetrahedra: this.tets.length,
      numberOfProbes: this.probes.length,
    };
  }
}

// ─── Cubemap rendering helpers ──────────────────────────────────────────

function _allocCM(gl, res) {
  const depth = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, depth);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.DEPTH_COMPONENT24, res, res, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  const rad = _cmF16(gl, res, res);
  const nrm = _cmF16(gl, res, res);
  const dist = _cmR16F(gl, res, res);
  return { fbo, rad, nrm, dist, depth };
}
function _cmF16(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  for (const [n,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  return t;
}
function _cmR16F(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.R16F, w, h, 0, gl.RED, gl.HALF_FLOAT, null);
  for (const [n,v] of [[gl.TEXTURE_MIN_FILTER,gl.NEAREST],[gl.TEXTURE_MAG_FILTER,gl.NEAREST],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  return t;
}
function _bindCMFace(gl, gb, face) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
  const t = gl.TEXTURE_CUBE_MAP_POSITIVE_X+face;
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, gb.rad, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, t, gb.nrm, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, t, gb.dist, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, t, gb.depth, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
}
function _freeCM(gl, gb) {
  gl.deleteFramebuffer(gb.fbo);
  gl.deleteTexture(gb.rad); gl.deleteTexture(gb.nrm); gl.deleteTexture(gb.dist); gl.deleteTexture(gb.depth);
}
function _persp(fov,a,near,far) {
  const f=1/Math.tan(fov/2), nf=1/(near-far);
  return new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function _lookAt(e,t,u) {
  const z=_norm(_sub(e,t)), x=_norm(_cross(u,z)), y=_cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -_dot(x,e),-_dot(y,e),-_dot(z,e),1]);
}
function _norm(v) { const l=Math.hypot(v[0],v[1],v[2]); return [v[0]/l,v[1]/l,v[2]/l]; }
function _sub(a,b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function _cross(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
function _dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function _mul4(a,b) {
  const r=new Float32Array(16);
  for(let i=0;i<4;i++) for(let j=0;j<4;j++) r[j*4+i]=a[i]*b[j*4]+a[4+i]*b[j*4+1]+a[8+i]*b[j*4+2]+a[12+i]*b[j*4+3];
  return r;
}
