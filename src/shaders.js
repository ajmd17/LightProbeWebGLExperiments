// ─── Octahedral helpers (shared across all shaders) ──────────────────
export const octHelpers = `
vec3 octDecode(vec2 uv) {
  uv = uv * 2.0 - 1.0;
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) { vec2 s = sign(uv); v.xy = (1.0 - abs(v.yx)) * s; }
  return normalize(v);
}
vec2 octEncode(vec3 d) {
  vec3 p = d / (abs(d.x) + abs(d.y) + abs(d.z));
  if (p.z < 0.0) { vec2 s = sign(p.xy); p.xy = (1.0 - abs(p.yx)) * s; }
  return p.xy * 0.5 + 0.5;
}
mat3 buildBasis(vec3 N) {
  vec3 up = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
  vec3 T = normalize(cross(up, N));
  return mat3(T, cross(N, T), N);
}
`;

// ─── Fullscreen quad vertex ─────────────────────────────────────────
export const quadVert = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() { vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
`;

// ─── Scene G-buffer (main camera) ───────────────────────────────────
export const sceneGBufVert = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec3 aCol;
uniform mat4 uVP;
uniform mat4 uM;
out vec3 vN; out vec3 vC; out vec3 vW;
void main() { vec4 wp = uM * vec4(aPos,1); vW=wp.xyz; vN=mat3(uM)*aNrm; vC=aCol; gl_Position=uVP*wp; }
`;

export const sceneGBufFrag = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vC; in vec3 vW;
layout(location=0) out vec4 oPos;
layout(location=1) out vec4 oNrm;
layout(location=2) out vec4 oAlb;
void main() { oPos=vec4(vW,1); oNrm=vec4(normalize(vN)*0.5+0.5,1); oAlb=vec4(vC,1); }
`;

// ─── Probe G-buffer (cubemap MRT) ───────────────────────────────────
export const probeGBufVert = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec3 aCol;
uniform mat4 uVP;
uniform mat4 uM;
out vec3 vN; out vec3 vC; out vec3 vW;
void main() { vec4 wp=uM*vec4(aPos,1); vW=wp.xyz; vN=mat3(uM)*aNrm; vC=aCol; gl_Position=uVP*wp; }
`;

export const probeGBufFrag = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vC; in vec3 vW;
uniform vec3 uQ;
uniform vec3 uLightPos;
uniform vec3 uLightCol;
layout(location=0) out vec4 oRad;
layout(location=1) out vec4 oNrm;
layout(location=2) out float oD;
void main() {
  vec3 N = normalize(vN);
  if (dot(N, normalize(uQ - vW)) < 0.0) N = -N;
  vec3 L = normalize(uLightPos - vW);
  float NdotL = max(0.0, dot(N, L));
  float distL = length(uLightPos - vW);
  float atten = min(1.0 / (1.0 + distL * distL * 0.01), 1.0);
  vec3 direct = uLightCol * NdotL * atten;
  vec3 rad = vC * direct;
  oRad = vec4(rad, 1); oNrm = vec4(N*0.5+0.5, 1); oD = length(vW - uQ);
}
`;

// ─── Octahedral conversion (cubemap → 2D) ───────────────────────────
export const octFrag = `#version 300 es
precision highp float;
uniform highp samplerCube uRad;
uniform highp samplerCube uNrm;
uniform highp samplerCube uDist;
in vec2 vUV;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
layout(location=2) out float o2;
${octHelpers}
void main() {
  vec3 d = octDecode(vUV);
  o0 = texture(uRad, d); o1 = texture(uNrm, d); o2 = texture(uDist, d).r;
}
`;

// ─── Irradiance filter (Lambertian convolution) ─────────────────────
export const irrFrag = `#version 300 es
precision highp float;
#define PI 3.14159265
uniform highp samplerCube uRad;
uniform int uN;
in vec2 vUV;
layout(location=0) out vec4 oI;
${octHelpers}
vec3 fibSph(int i, int N) {
  float p = acos(1.0-2.0*float(i+1)/float(N+1)), t = PI*(1.0+sqrt(5.0))*float(i);
  return vec3(sin(p)*cos(t), sin(p)*sin(t), cos(p));
}
void main() {
  vec3 N = octDecode(vUV);
  mat3 B = buildBasis(N);
  vec3 irr = vec3(0); int Ns = uN;
  for (int i = 0; i < 1024; i++) {
    if (i >= Ns) break;
    vec3 L = fibSph(i, Ns);
    if (L.z <= 0.0) continue;
    irr += texture(uRad, B * L).rgb * L.z;
  }
  oI = vec4(irr * 4.0 * PI / float(Ns), 1);
}
`;

// ─── VSM filter ─────────────────────────────────────────────────────
export const vsmFrag = `#version 300 es
precision highp float;
uniform highp samplerCube uDist;
in vec2 vUV;
layout(location=0) out vec4 oV;
${octHelpers}
vec3 ss(int i) {
  float p = acos(1.0-2.0*float(i+1)/33.0), t = 2.399963*float(i);
  return vec3(sin(p)*cos(t), sin(p)*sin(t), cos(p));
}
void main() {
  vec3 d = octDecode(vUV);
  float dd = texture(uDist, d).r;
  float avg = dd, avgSq = dd*dd;
  for (int i = 0; i < 16; i++) {
    float x = texture(uDist, normalize(d + ss(i)*0.10)).r;
    avg += x; avgSq += x*x;
  }
  float iN = 1.0/17.0; avg *= iN; avgSq *= iN;
  oV = vec4(avg, max(abs(avgSq - avg*avg), 1e-6), 0.0, 1.0);
}
`;

// ─── Ray-marching shader (full-screen pass) ─────────────────────────
export const raymarchFrag = `#version 300 es
precision highp float;

#define NPROBES_PER_AXIS 6
#define NPROBES (NPROBES_PER_AXIS * NPROBES_PER_AXIS * NPROBES_PER_AXIS)

uniform highp sampler2D uPos;
uniform highp sampler2D uNrm;
uniform highp sampler2D uAlb;
uniform highp sampler2DArray uIrr;
uniform highp sampler2DArray uVSM;
uniform highp sampler2DArray uDistH;
uniform highp sampler2DArray uDistL;

uniform vec3 uProbePos[NPROBES];
uniform vec3 uGMin;
uniform vec3 uGMax;
uniform vec3 uCamFwd;
uniform vec3 uCamUp;
uniform vec3 uCamRight;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uNormalBias;
uniform float uDistBias;
uniform int uDebug;
uniform int uSingleProbe;

in vec2 vUV;
out vec4 fColor;

${octHelpers}

// Find 4x4x4 grid cell containing P, return 8 corner indices & weights
void getProbeSlice(vec3 P, out int idx[8], out float wt[8]) {
  vec3 t = clamp((P - uGMin) / (uGMax - uGMin), 0.0, 1.0);
  float step = 1.0 / float(NPROBES_PER_AXIS - 1);
  ivec3 cell = ivec3(floor(t / step));
  cell = min(cell, ivec3(NPROBES_PER_AXIS - 2));
  vec3 f = (t - vec3(cell) * step) / step;
  float x=f.x, y=f.y, z=f.z;
  float ox=1.0-x, oy=1.0-y, oz=1.0-z;
  int ci=cell.x, cj=cell.y, ck=cell.z;
  int s1=1, s2=NPROBES_PER_AXIS, s3=NPROBES_PER_AXIS*NPROBES_PER_AXIS;
  int base = ci + cj*s2 + ck*s3;
  idx[0]=base;                 wt[0]=ox*oy*oz;
  idx[1]=base+s1;              wt[1]=x *oy*oz;
  idx[2]=base+s2;              wt[2]=ox*y *oz;
  idx[3]=base+s1+s2;           wt[3]=x *y *oz;
  idx[4]=base+s3;              wt[4]=ox*oy*z;
  idx[5]=base+s1+s3;           wt[5]=x *oy*z;
  idx[6]=base+s2+s3;           wt[6]=ox*y *z;
  idx[7]=base+s1+s2+s3;        wt[7]=x *y *z;
}

// Chebyshev upper-bound visibility
float cheb(float d, float avg, float var) {
  if (d <= avg) return 1.0;
  float dd = d - avg;
  float p = var / (var + dd*dd);
  float bias = 0.05;
  return clamp((p - bias) / max(1.0 - bias, 1e-6), 0.0, 1.0);
}

// Probe visibility with step counter
float probeVis(vec3 P, vec3 Q, int pi, out int steps) {
  vec3 rdir = normalize(P - Q);
  float dist = max(length(P - Q) - uDistBias, 0.001);
  vec2 uv = octEncode(rdir);

  // VSM Chebyshev visibility
  vec2 vsm = texture(uVSM, vec3(uv, pi)).rg;
  float vis = cheb(dist, vsm.x, vsm.y);
  steps = 1;
  if (vis < 0.001) return 0.0;
  if (vis > 0.95) return vis;

  // Refinement: ray-march with resolution swap
  float t = 0.0;
  float marchStep = 0.5;
  bool hires = false;
  steps = 0;

  for (int i = 0; i < 256; i++) {
    if (t >= dist) break;
    steps++;
    float sd;
    if (hires) {
      sd = texture(uDistH, vec3(uv, pi)).r;
      if (t - sd + uDistBias > 0.10) return 0.0;
    } else {
      sd = texture(uDistL, vec3(uv, pi)).r;
      if (t - sd + uDistBias > -0.15) { hires = true; marchStep = 0.05; continue; }
    }
    t += marchStep;
  }
  return vis;
}

vec3 skyCol(vec3 dir) {
  float y = max(dir.y, 0.0);
  return mix(vec3(0.7, 0.8, 0.9), vec3(0.2, 0.4, 0.85), y);
}

void main() {
  vec3 P = texture(uPos, vUV).xyz;
  vec3 N = normalize(texture(uNrm, vUV).xyz * 2.0 - 1.0);
  vec3 albedo = texture(uAlb, vUV).rgb;

  if (length(P) < 0.001) {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec3 wDir = normalize(uCamFwd + ndc.x * uCamRight * uTanHalfFov * uAspect + ndc.y * uCamUp * uTanHalfFov);
    fColor = vec4(skyCol(wDir), 1.0);
    return;
  }

  // Phase 1: output raw octahedral UV of surface normal
  if (uDebug == 1) {
    vec2 nuv = octEncode(N);
    fColor = vec4(nuv, 0.0, 1.0);
    return;
  }

  // Phase 2: output distance sampled from first probe
  if (uDebug == 2) {
    vec3 dir = normalize(uProbePos[0] - P);
    vec2 duv = octEncode(dir);
    float d = texture(uDistH, vec3(duv, 0)).r;
    fColor = vec4(vec3(d / 10.0), 1.0);
    return;
  }

  // Modes 4-6: texture array inspection (sample at screen UV)
  if (uDebug >= 4 && uDebug <= 6) {
    int pi = max(uSingleProbe, 0);
    if (uDebug == 4) {
      vec3 irr = texture(uIrr, vec3(vUV, pi)).rgb;
      fColor = vec4(irr, 1.0);
    } else if (uDebug == 5) {
      float d = texture(uDistH, vec3(vUV, pi)).r;
      fColor = vec4(vec3(d / 10.0), 1.0);
    } else {
      vec2 uv2 = vUV;
      if (vUV.x < 0.5) { uv2.x = vUV.x * 2.0; vec3 irr = texture(uIrr, vec3(uv2, pi)).rgb; fColor = vec4(irr, 1.0); }
      else { uv2.x = (vUV.x - 0.5) * 2.0; float d = texture(uDistH, vec3(uv2, pi)).r; fColor = vec4(vec3(d / 10.0), 1.0); }
    }
    return;
  }

  vec3 indirect = vec3(0.0);
  float tw = 0.0;
  vec2 nOct = octEncode(N);
  int maxSteps = 0;

  // Phase 7: render with modifiers disabled (bf=1, vis=1) to isolate trilinear
  if (uDebug == 7) {
    int pi[8]; float pw[8];
    getProbeSlice(P, pi, pw);
    for (int j = 0; j < 8; j++) {
      float weight = pw[j];
      if (weight < 0.001) continue;
      int i = pi[j];
      vec3 irr = texture(uIrr, vec3(nOct, i)).rgb;
      indirect += irr * weight; tw += weight;
    }
    if (tw > 0.0) indirect /= tw;
    vec3 color = indirect * albedo;
    fColor = vec4(color / (1.0 + color), 1.0);
    return;
  }

  vec3 indirectTri = vec3(0.0);
  float twTri = 0.0;
  float dbgBF = 0.0, dbgVis = 0.0;

  if (uSingleProbe >= 0 && uSingleProbe < NPROBES) {
    int i = uSingleProbe;
    vec3 Q = uProbePos[i];
    vec3 dQ = normalize(Q - P);
    float bf = (dot(N, dQ) + 0.2) / 1.2;
    bf = max(0.001, bf);
    int steps;
    float vis = probeVis(P + N * uNormalBias, Q, i, steps);
    maxSteps = steps;
    vis = max(0.001, vis);
    vec3 irr = texture(uIrr, vec3(nOct, i)).rgb;
    float w = bf * vis;
    indirect += irr * w; tw += w;
    indirectTri = irr; twTri = 1.0;
    dbgBF = bf; dbgVis = vis;
  } else {
    int pi[8]; float pw[8];
    getProbeSlice(P, pi, pw);
    for (int j = 0; j < 8; j++) {
      float weight = pw[j];
      if (weight < 0.001) continue;
      int i = pi[j];
      vec3 Q = uProbePos[i];
      vec3 dQ = normalize(Q - P);
      float bf = (dot(N, dQ) + 0.2) / 1.2;
      bf = max(0.001, bf);
      int steps;
      float vis = probeVis(P + N * uNormalBias, Q, i, steps);
      if (steps > maxSteps) maxSteps = steps;
      vis = max(0.001, vis);
      vec3 irr = texture(uIrr, vec3(nOct, i)).rgb;
      float w = bf * vis * weight;
      indirect += irr * w; tw += w;
      indirectTri += irr * weight; twTri += weight;
      dbgBF += bf * weight; dbgVis += vis * weight;
    }
  }

  if (uDebug == 3) {
    float s = float(maxSteps) / 256.0;
    fColor = vec4(vec3(s), 1.0);
    return;
  }

  // Phase 8: debug total weight (tw as grayscale)
  if (uDebug == 8) {
    fColor = vec4(vec3(tw), 1.0);
    return;
  }

  // Phase 9: debug backface weight (bf weighted by trilinear)
  if (uDebug == 9) {
    fColor = vec4(vec3(dbgBF), 1.0);
    return;
  }

  // Phase 10: debug visibility weight (vis weighted by trilinear)
  if (uDebug == 10) {
    fColor = vec4(vec3(dbgVis), 1.0);
    return;
  }

  if (tw < 0.0001) {
    indirect = (twTri > 0.0) ? (indirectTri / twTri) : vec3(0.0);
  } else {
    indirect /= tw;
  }
  vec3 color = indirect * albedo;
  fColor = vec4(color / (1.0 + color), 1.0);
}
`;
