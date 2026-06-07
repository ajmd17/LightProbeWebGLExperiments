// CPU voxelization of triangle meshes into a regular 3D grid

function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// Triangle-AABB overlap via SAT (Separating Axis Theorem)
function triBoxOverlap(tri, bmin, bmax) {
  const tmin = [
    Math.min(tri[0][0], tri[1][0], tri[2][0]),
    Math.min(tri[0][1], tri[1][1], tri[2][1]),
    Math.min(tri[0][2], tri[1][2], tri[2][2]),
  ]
  const tmax = [
    Math.max(tri[0][0], tri[1][0], tri[2][0]),
    Math.max(tri[0][1], tri[1][1], tri[2][1]),
    Math.max(tri[0][2], tri[1][2], tri[2][2]),
  ]
  if (tmax[0] < bmin[0] || tmin[0] > bmax[0]) return false
  if (tmax[1] < bmin[1] || tmin[1] > bmax[1]) return false
  if (tmax[2] < bmin[2] || tmin[2] > bmax[2]) return false

  const e0 = [tri[1][0]-tri[0][0], tri[1][1]-tri[0][1], tri[1][2]-tri[0][2]]
  const e1 = [tri[2][0]-tri[1][0], tri[2][1]-tri[1][1], tri[2][2]-tri[1][2]]
  const e2 = [tri[0][0]-tri[2][0], tri[0][1]-tri[2][1], tri[0][2]-tri[2][2]]

  const n = [
    e0[1]*e1[2] - e0[2]*e1[1],
    e0[2]*e1[0] - e0[0]*e1[2],
    e0[0]*e1[1] - e0[1]*e1[0],
  ]
  if (n[0]*n[0] + n[1]*n[1] + n[2]*n[2] < 1e-20) return false

  const c = [(bmin[0]+bmax[0])*0.5, (bmin[1]+bmax[1])*0.5, (bmin[2]+bmax[2])*0.5]
  const h = [(bmax[0]-bmin[0])*0.5, (bmax[1]-bmin[1])*0.5, (bmax[2]-bmin[2])*0.5]

  const v0 = [tri[0][0]-c[0], tri[0][1]-c[1], tri[0][2]-c[2]]
  const v1 = [tri[1][0]-c[0], tri[1][1]-c[1], tri[1][2]-c[2]]
  const v2 = [tri[2][0]-c[0], tri[2][1]-c[1], tri[2][2]-c[2]]

  // AABB face normals (X, Y, Z)
  if (Math.max(v0[0],v1[0],v2[0]) < -h[0] || Math.min(v0[0],v1[0],v2[0]) > h[0]) return false
  if (Math.max(v0[1],v1[1],v2[1]) < -h[1] || Math.min(v0[1],v1[1],v2[1]) > h[1]) return false
  if (Math.max(v0[2],v1[2],v2[2]) < -h[2] || Math.min(v0[2],v1[2],v2[2]) > h[2]) return false

  // Triangle normal
  const np0 = dot(v0, n), np1 = dot(v1, n), np2 = dot(v2, n)
  const nr = h[0]*Math.abs(n[0]) + h[1]*Math.abs(n[1]) + h[2]*Math.abs(n[2])
  if (Math.max(np0,np1,np2) < -nr || Math.min(np0,np1,np2) > nr) return false

  // Edge × axis cross products (3 edges × 3 axes = 9 tests)
  for (let ei = 0; ei < 3; ei++) {
    const e = [e0, e1, e2][ei]
    for (let ai = 0; ai < 3; ai++) {
      const ax = [0, 0, 0]
      ax[(ai+1)%3] =  e[(ai+2)%3]
      ax[(ai+2)%3] = -e[(ai+1)%3]
      const verts = [v0, v1, v2]
      const p0 = dot(verts[0], ax), p1 = dot(verts[1], ax), p2 = dot(verts[2], ax)
      const pmin = Math.min(p0, p1, p2), pmax = Math.max(p0, p1, p2)
      const rad = h[0]*Math.abs(ax[0]) + h[1]*Math.abs(ax[1]) + h[2]*Math.abs(ax[2])
      if (pmin > rad || pmax < -rad) return false
    }
  }

  return true
}

export class VoxelGrid {
  constructor(min, max, res) {
    this.min = min
    this.max = max
    this.res = res
    this.size = [max[0]-min[0], max[1]-min[1], max[2]-min[2]]
    this.vs = [this.size[0]/res, this.size[1]/res, this.size[2]/res]
    const n = res * res * res
    this.occ = new Uint8Array(n)
    this.rgb = new Float32Array(n * 3)
  }

  idx(x, y, z) { return x + y * this.res + z * this.res * this.res }

  toGrid(wx, wy, wz) {
    return [
      Math.floor((wx - this.min[0]) / this.vs[0]),
      Math.floor((wy - this.min[1]) / this.vs[1]),
      Math.floor((wz - this.min[2]) / this.vs[2]),
    ]
  }

  set(x, y, z, r, g, b) {
    if (x < 0 || x >= this.res || y < 0 || y >= this.res || z < 0 || z >= this.res) return
    const i = this.idx(x, y, z)
    this.occ[i] = 1
    this.rgb[i*3] = r
    this.rgb[i*3+1] = g
    this.rgb[i*3+2] = b
  }
}

export function voxelizeScene(meshes, bounds, res) {
  const grid = new VoxelGrid(bounds[0], bounds[1], res)
  const bmin = bounds[0]

  for (const mesh of meshes) {
    const pos = mesh.pos, col = mesh.col, idx = mesh.idx
    for (let i = 0; i < idx.length; i += 3) {
      const i0 = idx[i] * 3, i1 = idx[i+1] * 3, i2 = idx[i+2] * 3
      const tri = [
        [pos[i0], pos[i0+1], pos[i0+2]],
        [pos[i1], pos[i1+1], pos[i1+2]],
        [pos[i2], pos[i2+1], pos[i2+2]],
      ]

      const tmin = [
        Math.min(tri[0][0], tri[1][0], tri[2][0]),
        Math.min(tri[0][1], tri[1][1], tri[2][1]),
        Math.min(tri[0][2], tri[1][2], tri[2][2]),
      ]
      const tmax = [
        Math.max(tri[0][0], tri[1][0], tri[2][0]),
        Math.max(tri[0][1], tri[1][1], tri[2][1]),
        Math.max(tri[0][2], tri[1][2], tri[2][2]),
      ]

      const gs = grid.toGrid(tmin[0], tmin[1], tmin[2])
      const ge = grid.toGrid(tmax[0], tmax[1], tmax[2])

      const sx = Math.max(0, gs[0]), sy = Math.max(0, gs[1]), sz = Math.max(0, gs[2])
      const ex = Math.min(res-1, ge[0]+1), ey = Math.min(res-1, ge[1]+1), ez = Math.min(res-1, ge[2]+1)

      for (let vz = sz; vz <= ez; vz++) {
        for (let vy = sy; vy <= ey; vy++) {
          for (let vx = sx; vx <= ex; vx++) {
            const vmin = [
              bmin[0] + vx * grid.vs[0],
              bmin[1] + vy * grid.vs[1],
              bmin[2] + vz * grid.vs[2],
            ]
            const vmax = [
              vmin[0] + grid.vs[0],
              vmin[1] + grid.vs[1],
              vmin[2] + grid.vs[2],
            ]
            if (triBoxOverlap(tri, vmin, vmax)) {
              grid.set(vx, vy, vz, col[0], col[1], col[2])
            }
          }
        }
      }
    }
  }

  return grid
}

export function sceneBounds(meshes, padding) {
  const p = padding || 0.5
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
  for (const m of meshes) {
    const pos = m.pos
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i]   < min[0]) min[0] = pos[i]
      if (pos[i+1] < min[1]) min[1] = pos[i+1]
      if (pos[i+2] < min[2]) min[2] = pos[i+2]
      if (pos[i]   > max[0]) max[0] = pos[i]
      if (pos[i+1] > max[1]) max[1] = pos[i+1]
      if (pos[i+2] > max[2]) max[2] = pos[i+2]
    }
  }
  return [[min[0]-p, min[1]-p, min[2]-p], [max[0]+p, max[1]+p, max[2]+p]]
}
