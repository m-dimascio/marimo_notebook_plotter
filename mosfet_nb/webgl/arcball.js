/**
 * Arcball camera for natural 3D interaction.
 * Adapted from Twinklebear/webgl-volume-raycaster
 *
 * Provides intuitive rotation via mouse/touch drag, zoom via scroll/pinch.
 */

// Quaternion operations (minimal implementation)
const quat = {
  create: () => [0, 0, 0, 1],

  setAxisAngle: (out, axis, angle) => {
    const halfAngle = angle / 2;
    const s = Math.sin(halfAngle);
    out[0] = axis[0] * s;
    out[1] = axis[1] * s;
    out[2] = axis[2] * s;
    out[3] = Math.cos(halfAngle);
    return out;
  },

  multiply: (out, a, b) => {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    const bx = b[0], by = b[1], bz = b[2], bw = b[3];
    out[0] = ax * bw + aw * bx + ay * bz - az * by;
    out[1] = ay * bw + aw * by + az * bx - ax * bz;
    out[2] = az * bw + aw * bz + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
  },

  normalize: (out, a) => {
    const len = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
    if (len > 0) {
      const invLen = 1 / len;
      out[0] = a[0] * invLen;
      out[1] = a[1] * invLen;
      out[2] = a[2] * invLen;
      out[3] = a[3] * invLen;
    }
    return out;
  }
};

// Vector3 operations
const vec3 = {
  create: () => [0, 0, 0],

  cross: (out, a, b) => {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
    return out;
  },

  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],

  length: (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]),

  normalize: (out, a) => {
    const len = vec3.length(a);
    if (len > 0) {
      const invLen = 1 / len;
      out[0] = a[0] * invLen;
      out[1] = a[1] * invLen;
      out[2] = a[2] * invLen;
    }
    return out;
  },

  add: (out, a, b) => {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
  },

  transformMat4: (out, a, m) => {
    const x = a[0], y = a[1], z = a[2];
    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    return out;
  }
};

// Matrix4 operations
const mat4 = {
  create: () => {
    const out = new Float32Array(16);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  },

  fromQuat: (out, q) => {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, yx = y * x2, yy = y * y2;
    const zx = z * x2, zy = z * y2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    out[0] = 1 - yy - zz;
    out[1] = yx + wz;
    out[2] = zx - wy;
    out[3] = 0;
    out[4] = yx - wz;
    out[5] = 1 - xx - zz;
    out[6] = zy + wx;
    out[7] = 0;
    out[8] = zx + wy;
    out[9] = zy - wx;
    out[10] = 1 - xx - yy;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  },

  lookAt: (out, eye, center, up) => {
    const zx = eye[0] - center[0];
    const zy = eye[1] - center[1];
    const zz = eye[2] - center[2];
    let len = 1 / Math.sqrt(zx * zx + zy * zy + zz * zz);
    const z0 = zx * len, z1 = zy * len, z2 = zz * len;

    const xx = up[1] * z2 - up[2] * z1;
    const xy = up[2] * z0 - up[0] * z2;
    const xz = up[0] * z1 - up[1] * z0;
    len = Math.sqrt(xx * xx + xy * xy + xz * xz);
    len = len ? 1 / len : 0;
    const x0 = xx * len, x1 = xy * len, x2 = xz * len;

    const y0 = z1 * x2 - z2 * x1;
    const y1 = z2 * x0 - z0 * x2;
    const y2 = z0 * x1 - z1 * x0;

    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
    out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
    out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
    out[15] = 1;
    return out;
  },

  perspective: (out, fovy, aspect, near, far) => {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
  }
};

/**
 * ArcballCamera - Intuitive 3D camera with mouse/touch support
 */
class ArcballCamera {
  constructor(center = [0, 0, 0], zoomSpeed = 0.1, rotationSpeed = 0.01) {
    this.center = center.slice();
    this.zoomSpeed = zoomSpeed;
    this.rotationSpeed = rotationSpeed;

    this.rotation = quat.create();
    this.distance = 3.0;

    this.viewMatrix = mat4.create();
    this.projectionMatrix = mat4.create();
    this.isDragging = false;
    this.lastMouse = [0, 0];

    // Pan support
    this.isPanning = false;
    this.panSpeed = 0.005;
  }

  /**
   * Project point onto arcball sphere
   */
  projectToSphere(x, y, width, height) {
    const px = (2 * x - width) / Math.min(width, height);
    const py = (height - 2 * y) / Math.min(width, height);
    const len2 = px * px + py * py;

    if (len2 <= 0.5) {
      return [px, py, Math.sqrt(1.0 - len2)];
    } else {
      const len = Math.sqrt(len2);
      return [px / len, py / len, 0];
    }
  }

  /**
   * Rotate camera based on mouse movement
   */
  rotate(prevMouse, curMouse, width, height) {
    const prev = this.projectToSphere(prevMouse[0], prevMouse[1], width, height);
    const cur = this.projectToSphere(curMouse[0], curMouse[1], width, height);

    // Compute rotation axis and angle
    const axis = vec3.cross([], prev, cur);
    const dot = vec3.dot(prev, cur);
    const angle = Math.acos(Math.min(1.0, Math.max(-1.0, dot))) * this.rotationSpeed * 10;

    if (vec3.length(axis) > 0.001) {
      vec3.normalize(axis, axis);
      const deltaQuat = quat.setAxisAngle([], axis, angle);
      quat.multiply(this.rotation, deltaQuat, this.rotation);
      quat.normalize(this.rotation, this.rotation);
    }
  }

  /**
   * Pan camera (shift + drag)
   */
  pan(dx, dy) {
    // Get right and up vectors from view matrix
    const right = [this.viewMatrix[0], this.viewMatrix[4], this.viewMatrix[8]];
    const up = [this.viewMatrix[1], this.viewMatrix[5], this.viewMatrix[9]];

    const panAmount = this.distance * this.panSpeed;
    this.center[0] -= (right[0] * dx + up[0] * dy) * panAmount;
    this.center[1] -= (right[1] * dx + up[1] * dy) * panAmount;
    this.center[2] -= (right[2] * dx + up[2] * dy) * panAmount;
  }

  /**
   * Zoom camera
   */
  zoom(delta) {
    this.distance *= 1 + delta * this.zoomSpeed;
    this.distance = Math.max(0.5, Math.min(20, this.distance));
  }

  /**
   * Get camera position in world space
   */
  getEyePosition() {
    const rotMat = mat4.fromQuat([], this.rotation);
    const eye = vec3.transformMat4([], [0, 0, this.distance], rotMat);
    vec3.add(eye, eye, this.center);
    return eye;
  }

  /**
   * Get view matrix
   */
  getViewMatrix() {
    const eye = this.getEyePosition();
    mat4.lookAt(this.viewMatrix, eye, this.center, [0, 1, 0]);
    return this.viewMatrix;
  }

  /**
   * Get projection matrix
   */
  getProjectionMatrix(aspect, fov = Math.PI / 4, near = 0.1, far = 100) {
    mat4.perspective(this.projectionMatrix, fov, aspect, near, far);
    return this.projectionMatrix;
  }

  /**
   * Attach event listeners to canvas
   */
  attachToCanvas(canvas, onUpdate = null) {
    // Mouse events
    canvas.addEventListener('mousedown', (e) => {
      if (e.shiftKey) {
        this.isPanning = true;
      } else {
        this.isDragging = true;
      }
      this.lastMouse = [e.clientX, e.clientY];
      e.preventDefault();
    });

    canvas.addEventListener('mousemove', (e) => {
      const cur = [e.clientX, e.clientY];

      if (this.isDragging) {
        this.rotate(this.lastMouse, cur, canvas.width, canvas.height);
        if (onUpdate) onUpdate();
      } else if (this.isPanning) {
        const dx = cur[0] - this.lastMouse[0];
        const dy = cur[1] - this.lastMouse[1];
        this.pan(dx, dy);
        if (onUpdate) onUpdate();
      }

      this.lastMouse = cur;
    });

    const stopDrag = () => {
      this.isDragging = false;
      this.isPanning = false;
    };
    canvas.addEventListener('mouseup', stopDrag);
    canvas.addEventListener('mouseleave', stopDrag);

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom(e.deltaY > 0 ? 1 : -1);
      if (onUpdate) onUpdate();
    }, { passive: false });

    // Touch support
    let lastTouchDist = 0;
    let lastTouchCenter = [0, 0];

    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
      } else if (e.touches.length === 2) {
        // Two-finger: compute distance and center for pinch/pan
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastTouchDist = Math.sqrt(dx * dx + dy * dy);
        lastTouchCenter = [
          (e.touches[0].clientX + e.touches[1].clientX) / 2,
          (e.touches[0].clientY + e.touches[1].clientY) / 2
        ];
      }
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && this.isDragging) {
        const cur = [e.touches[0].clientX, e.touches[0].clientY];
        this.rotate(this.lastMouse, cur, canvas.width, canvas.height);
        this.lastMouse = cur;
        if (onUpdate) onUpdate();
      } else if (e.touches.length === 2) {
        // Pinch to zoom
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        this.zoom((lastTouchDist - dist) * 0.02);
        lastTouchDist = dist;

        // Two-finger pan
        const center = [
          (e.touches[0].clientX + e.touches[1].clientX) / 2,
          (e.touches[0].clientY + e.touches[1].clientY) / 2
        ];
        const panDx = center[0] - lastTouchCenter[0];
        const panDy = center[1] - lastTouchCenter[1];
        this.pan(panDx, panDy);
        lastTouchCenter = center;

        if (onUpdate) onUpdate();
      }
    }, { passive: false });

    canvas.addEventListener('touchend', () => {
      this.isDragging = false;
    });
  }

  /**
   * Reset camera to initial state
   */
  reset() {
    this.rotation = quat.create();
    this.distance = 3.0;
    this.center = [0, 0, 0];
  }

  /**
   * Set camera to look at bounds
   */
  fitToBounds(boundsMin, boundsMax) {
    // Center on bounds
    this.center = [
      (boundsMin[0] + boundsMax[0]) / 2,
      (boundsMin[1] + boundsMax[1]) / 2,
      (boundsMin[2] + boundsMax[2]) / 2
    ];

    // Set distance based on bounds size
    const size = Math.max(
      boundsMax[0] - boundsMin[0],
      boundsMax[1] - boundsMin[1],
      boundsMax[2] - boundsMin[2]
    );
    this.distance = size * 2;
  }
}

// Export for use as module or inline
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ArcballCamera, mat4, vec3, quat };
}
