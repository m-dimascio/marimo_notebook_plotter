/**
 * MOSFET Volume Renderer
 * Combines techniques from:
 * - WebGL-Fluid-Simulation (smooth gradient rendering)
 * - webgl-volume-raycaster (scientific visualization)
 *
 * Renders 3D electron concentration with arcball camera control.
 */

// Viridis colormap data (256 RGB values)
const VIRIDIS_COLORS = [
  [68, 1, 84], [68, 2, 86], [69, 4, 87], [69, 5, 89], [70, 7, 90],
  [70, 8, 92], [70, 10, 93], [70, 11, 94], [71, 13, 96], [71, 14, 97],
  [71, 16, 99], [71, 17, 100], [71, 19, 101], [72, 20, 103], [72, 22, 104],
  [72, 23, 105], [72, 24, 106], [72, 26, 108], [72, 27, 109], [72, 28, 110],
  [72, 29, 111], [72, 31, 112], [72, 32, 113], [72, 33, 115], [72, 35, 116],
  [72, 36, 117], [72, 37, 118], [72, 38, 119], [72, 40, 120], [72, 41, 121],
  [71, 42, 122], [71, 44, 122], [71, 45, 123], [71, 46, 124], [71, 47, 125],
  [70, 48, 126], [70, 50, 126], [70, 51, 127], [69, 52, 128], [69, 53, 129],
  [69, 55, 129], [68, 56, 130], [68, 57, 131], [68, 58, 131], [67, 60, 132],
  [67, 61, 132], [66, 62, 133], [66, 63, 133], [66, 64, 134], [65, 66, 134],
  [65, 67, 135], [64, 68, 135], [64, 69, 136], [63, 71, 136], [63, 72, 137],
  [62, 73, 137], [62, 74, 137], [62, 76, 138], [61, 77, 138], [61, 78, 138],
  [60, 79, 139], [60, 80, 139], [59, 82, 139], [59, 83, 140], [58, 84, 140],
  [58, 85, 140], [57, 86, 141], [57, 88, 141], [56, 89, 141], [56, 90, 141],
  [55, 91, 142], [55, 92, 142], [54, 93, 142], [54, 95, 142], [53, 96, 142],
  [53, 97, 143], [52, 98, 143], [52, 99, 143], [51, 100, 143], [51, 102, 143],
  [50, 103, 143], [50, 104, 144], [49, 105, 144], [49, 106, 144], [49, 107, 144],
  [48, 108, 144], [48, 110, 144], [47, 111, 144], [47, 112, 144], [46, 113, 144],
  [46, 114, 144], [46, 115, 144], [45, 116, 144], [45, 117, 144], [44, 118, 144],
  [44, 120, 144], [44, 121, 144], [43, 122, 144], [43, 123, 144], [42, 124, 144],
  [42, 125, 144], [42, 126, 143], [41, 127, 143], [41, 128, 143], [41, 129, 143],
  [40, 130, 143], [40, 131, 143], [40, 132, 142], [39, 133, 142], [39, 134, 142],
  [39, 135, 142], [38, 137, 141], [38, 138, 141], [38, 139, 141], [37, 140, 140],
  [37, 141, 140], [37, 142, 140], [37, 143, 139], [36, 144, 139], [36, 145, 138],
  [36, 146, 138], [36, 147, 137], [35, 148, 137], [35, 149, 136], [35, 150, 136],
  [35, 151, 135], [35, 152, 135], [35, 153, 134], [35, 154, 134], [35, 155, 133],
  [35, 156, 132], [35, 157, 132], [35, 158, 131], [35, 159, 130], [36, 160, 130],
  [36, 161, 129], [36, 162, 128], [37, 163, 127], [37, 164, 127], [38, 165, 126],
  [38, 166, 125], [39, 167, 124], [40, 168, 123], [40, 169, 122], [41, 170, 121],
  [42, 171, 120], [43, 172, 119], [44, 173, 118], [45, 174, 117], [46, 175, 116],
  [47, 175, 115], [48, 176, 114], [50, 177, 113], [51, 178, 112], [52, 179, 111],
  [54, 180, 109], [55, 181, 108], [57, 182, 107], [58, 182, 106], [60, 183, 104],
  [61, 184, 103], [63, 185, 102], [65, 186, 100], [67, 186, 99], [68, 187, 97],
  [70, 188, 96], [72, 189, 94], [74, 189, 93], [76, 190, 91], [78, 191, 90],
  [80, 191, 88], [82, 192, 87], [84, 193, 85], [86, 193, 84], [88, 194, 82],
  [90, 195, 80], [92, 195, 79], [94, 196, 77], [97, 197, 75], [99, 197, 74],
  [101, 198, 72], [103, 198, 70], [106, 199, 68], [108, 199, 67], [110, 200, 65],
  [113, 200, 63], [115, 201, 61], [117, 201, 59], [120, 202, 57], [122, 202, 56],
  [125, 203, 54], [127, 203, 52], [130, 204, 50], [132, 204, 48], [135, 205, 46],
  [137, 205, 44], [140, 206, 42], [142, 206, 40], [145, 206, 38], [148, 207, 36],
  [150, 207, 34], [153, 208, 32], [156, 208, 30], [158, 208, 28], [161, 209, 26],
  [164, 209, 24], [166, 209, 22], [169, 210, 20], [172, 210, 18], [175, 210, 17],
  [177, 211, 15], [180, 211, 13], [183, 211, 12], [186, 211, 10], [189, 212, 9],
  [191, 212, 8], [194, 212, 7], [197, 212, 7], [200, 213, 7], [202, 213, 7],
  [205, 213, 8], [208, 213, 9], [210, 213, 10], [213, 214, 12], [216, 214, 13],
  [218, 214, 15], [221, 214, 17], [223, 214, 19], [226, 214, 22], [228, 215, 24],
  [231, 215, 27], [233, 215, 30], [235, 215, 33], [237, 215, 36], [239, 216, 39],
  [241, 216, 43], [243, 216, 47], [245, 216, 50], [246, 217, 54], [248, 217, 58],
  [249, 217, 62], [250, 218, 66], [251, 218, 71], [252, 219, 75], [253, 219, 79],
  [253, 220, 84], [254, 220, 88], [254, 221, 93], [254, 222, 97], [254, 222, 102],
  [254, 223, 107], [254, 224, 111], [254, 225, 116], [253, 225, 121], [253, 226, 125],
  [253, 227, 130]
];

/**
 * Create viridis colormap texture
 */
function createColormapTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const data = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const color = VIRIDIS_COLORS[i] || VIRIDIS_COLORS[255];
    data[i * 4] = color[0];
    data[i * 4 + 1] = color[1];
    data[i * 4 + 2] = color[2];
    data[i * 4 + 3] = 255;
  }

  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  return texture;
}

/**
 * Compile shader
 */
function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/**
 * Create shader program
 */
function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

/**
 * MOSFETRenderer - WebGL2 volume renderer for MOSFET visualization
 */
class MOSFETRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true
    });

    if (!this.gl) {
      throw new Error('WebGL2 not supported');
    }

    // Import camera if available, otherwise inline
    if (typeof ArcballCamera !== 'undefined') {
      this.camera = new ArcballCamera([0, 0, 0], 0.15, 0.02);
    } else {
      // Minimal inline camera
      this.camera = this.createInlineCamera();
    }

    this.animationFrame = 0;
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.frameTextures = [];
    this.parameterValues = [];
    this.bounds = null;
    this.dimensions = null;
    this.needsRender = true;
    this.animationId = null;

    this.initShaders();
    this.initGeometry();
  }

  createInlineCamera() {
    // Minimal arcball camera implementation
    return {
      center: [0, 0, 0],
      distance: 3.0,
      rotation: [0, 0, 0, 1],
      viewMatrix: new Float32Array(16),
      projectionMatrix: new Float32Array(16),
      lastMouse: [0, 0],
      isDragging: false,
      rotationSpeed: 0.02,
      zoomSpeed: 0.15,

      getViewMatrix() {
        // Build view matrix from quaternion
        const q = this.rotation;
        const d = this.distance;
        const c = this.center;

        // Convert quaternion to rotation matrix
        const x = q[0], y = q[1], z = q[2], w = q[3];
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, yx = y * x2, yy = y * y2;
        const zx = z * x2, zy = z * y2, zz = z * z2;
        const wx = w * x2, wy = w * y2, wz = w * z2;

        const rotMat = [
          1 - yy - zz, yx + wz, zx - wy, 0,
          yx - wz, 1 - xx - zz, zy + wx, 0,
          zx + wy, zy - wx, 1 - xx - yy, 0,
          0, 0, 0, 1
        ];

        // Eye position
        const eye = [
          rotMat[8] * d + c[0],
          rotMat[9] * d + c[1],
          rotMat[10] * d + c[2]
        ];

        // Build lookAt matrix
        const zAxis = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
        const zLen = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
        zAxis[0] /= zLen; zAxis[1] /= zLen; zAxis[2] /= zLen;

        const up = [0, 1, 0];
        const xAxis = [
          up[1] * zAxis[2] - up[2] * zAxis[1],
          up[2] * zAxis[0] - up[0] * zAxis[2],
          up[0] * zAxis[1] - up[1] * zAxis[0]
        ];
        const xLen = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
        if (xLen > 0.0001) {
          xAxis[0] /= xLen; xAxis[1] /= xLen; xAxis[2] /= xLen;
        }

        const yAxis = [
          zAxis[1] * xAxis[2] - zAxis[2] * xAxis[1],
          zAxis[2] * xAxis[0] - zAxis[0] * xAxis[2],
          zAxis[0] * xAxis[1] - zAxis[1] * xAxis[0]
        ];

        this.viewMatrix[0] = xAxis[0]; this.viewMatrix[1] = yAxis[0]; this.viewMatrix[2] = zAxis[0]; this.viewMatrix[3] = 0;
        this.viewMatrix[4] = xAxis[1]; this.viewMatrix[5] = yAxis[1]; this.viewMatrix[6] = zAxis[1]; this.viewMatrix[7] = 0;
        this.viewMatrix[8] = xAxis[2]; this.viewMatrix[9] = yAxis[2]; this.viewMatrix[10] = zAxis[2]; this.viewMatrix[11] = 0;
        this.viewMatrix[12] = -(xAxis[0]*eye[0] + xAxis[1]*eye[1] + xAxis[2]*eye[2]);
        this.viewMatrix[13] = -(yAxis[0]*eye[0] + yAxis[1]*eye[1] + yAxis[2]*eye[2]);
        this.viewMatrix[14] = -(zAxis[0]*eye[0] + zAxis[1]*eye[1] + zAxis[2]*eye[2]);
        this.viewMatrix[15] = 1;

        return this.viewMatrix;
      },

      getProjectionMatrix(aspect) {
        const fov = Math.PI / 4;
        const near = 0.1, far = 100;
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);

        this.projectionMatrix[0] = f / aspect;
        this.projectionMatrix[5] = f;
        this.projectionMatrix[10] = (far + near) * nf;
        this.projectionMatrix[11] = -1;
        this.projectionMatrix[14] = 2 * far * near * nf;
        this.projectionMatrix[1] = this.projectionMatrix[2] = this.projectionMatrix[3] = 0;
        this.projectionMatrix[4] = this.projectionMatrix[6] = this.projectionMatrix[7] = 0;
        this.projectionMatrix[8] = this.projectionMatrix[9] = this.projectionMatrix[12] = this.projectionMatrix[13] = 0;
        this.projectionMatrix[15] = 0;

        return this.projectionMatrix;
      },

      getEyePosition() {
        const q = this.rotation;
        const x = q[0], y = q[1], z = q[2], w = q[3];
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, yy = y * y2, zz = z * z2;
        const wx = w * x2, wy = w * y2;
        const zx = z * x2, zy = z * y2;

        return [
          (z * x2 + w * y2) * this.distance + this.center[0],
          (z * y2 - w * x2) * this.distance + this.center[1],
          (1 - x * x2 - y * y2) * this.distance + this.center[2]
        ];
      },

      rotate(dx, dy) {
        const speed = this.rotationSpeed;
        // Simplified rotation update
        const ax = dy * speed;
        const ay = dx * speed;

        // Apply rotation (simplified)
        const ca = Math.cos(ax / 2), sa = Math.sin(ax / 2);
        const cb = Math.cos(ay / 2), sb = Math.sin(ay / 2);

        const q = this.rotation;
        const qx = [sa, 0, 0, ca];
        const qy = [0, sb, 0, cb];

        // Multiply quaternions: qy * qx * q
        const temp = [
          qx[3]*qy[0] + qx[0]*qy[3] + qx[1]*qy[2] - qx[2]*qy[1],
          qx[3]*qy[1] - qx[0]*qy[2] + qx[1]*qy[3] + qx[2]*qy[0],
          qx[3]*qy[2] + qx[0]*qy[1] - qx[1]*qy[0] + qx[2]*qy[3],
          qx[3]*qy[3] - qx[0]*qy[0] - qx[1]*qy[1] - qx[2]*qy[2]
        ];

        this.rotation = [
          temp[3]*q[0] + temp[0]*q[3] + temp[1]*q[2] - temp[2]*q[1],
          temp[3]*q[1] - temp[0]*q[2] + temp[1]*q[3] + temp[2]*q[0],
          temp[3]*q[2] + temp[0]*q[1] - temp[1]*q[0] + temp[2]*q[3],
          temp[3]*q[3] - temp[0]*q[0] - temp[1]*q[1] - temp[2]*q[2]
        ];

        // Normalize
        const len = Math.sqrt(
          this.rotation[0]*this.rotation[0] +
          this.rotation[1]*this.rotation[1] +
          this.rotation[2]*this.rotation[2] +
          this.rotation[3]*this.rotation[3]
        );
        this.rotation[0] /= len;
        this.rotation[1] /= len;
        this.rotation[2] /= len;
        this.rotation[3] /= len;
      },

      zoom(delta) {
        this.distance *= 1 + delta * this.zoomSpeed;
        this.distance = Math.max(0.5, Math.min(20, this.distance));
      },

      fitToBounds(boundsMin, boundsMax) {
        this.center = [
          (boundsMin[0] + boundsMax[0]) / 2,
          (boundsMin[1] + boundsMax[1]) / 2,
          (boundsMin[2] + boundsMax[2]) / 2
        ];
        const size = Math.max(
          boundsMax[0] - boundsMin[0],
          boundsMax[1] - boundsMin[1],
          boundsMax[2] - boundsMin[2]
        );
        this.distance = size * 2.5;
      },

      attachToCanvas(canvas, onUpdate) {
        canvas.addEventListener('mousedown', (e) => {
          this.isDragging = true;
          this.lastMouse = [e.clientX, e.clientY];
          e.preventDefault();
        });

        canvas.addEventListener('mousemove', (e) => {
          if (this.isDragging) {
            const dx = e.clientX - this.lastMouse[0];
            const dy = e.clientY - this.lastMouse[1];
            this.rotate(dx, dy);
            this.lastMouse = [e.clientX, e.clientY];
            if (onUpdate) onUpdate();
          }
        });

        canvas.addEventListener('mouseup', () => { this.isDragging = false; });
        canvas.addEventListener('mouseleave', () => { this.isDragging = false; });

        canvas.addEventListener('wheel', (e) => {
          e.preventDefault();
          this.zoom(e.deltaY > 0 ? 1 : -1);
          if (onUpdate) onUpdate();
        }, { passive: false });

        // Touch support
        let lastTouchDist = 0;
        canvas.addEventListener('touchstart', (e) => {
          e.preventDefault();
          if (e.touches.length === 1) {
            this.isDragging = true;
            this.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
          } else if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            lastTouchDist = Math.sqrt(dx*dx + dy*dy);
          }
        }, { passive: false });

        canvas.addEventListener('touchmove', (e) => {
          e.preventDefault();
          if (e.touches.length === 1 && this.isDragging) {
            const dx = e.touches[0].clientX - this.lastMouse[0];
            const dy = e.touches[0].clientY - this.lastMouse[1];
            this.rotate(dx, dy);
            this.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
            if (onUpdate) onUpdate();
          } else if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            this.zoom((lastTouchDist - dist) * 0.02);
            lastTouchDist = dist;
            if (onUpdate) onUpdate();
          }
        }, { passive: false });

        canvas.addEventListener('touchend', () => { this.isDragging = false; });
      }
    };
  }

  initShaders() {
    const gl = this.gl;

    // Simplified volume rendering shader (ray marching)
    const vertexSource = `#version 300 es
      in vec2 aPosition;
      out vec2 vUV;
      void main() {
        vUV = aPosition * 0.5 + 0.5;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    const fragmentSource = `#version 300 es
      precision highp float;
      precision highp sampler3D;

      uniform sampler3D uVolume;
      uniform sampler2D uColormap;
      uniform sampler3D uRegions;

      uniform vec3 uBoundsMin;
      uniform vec3 uBoundsMax;
      uniform vec3 uCameraPos;
      uniform mat4 uInvViewProj;

      uniform float uOpacityScale;
      uniform float uDensityThreshold;
      uniform int uMaxSteps;

      in vec2 vUV;
      out vec4 fragColor;

      vec4 applyColormap(float value) {
        return texture(uColormap, vec2(clamp(value, 0.0, 1.0), 0.5));
      }

      float getRegionOpacity(float region) {
        int r = int(region * 255.0);
        if (r == 4 || r == 5) return 0.15;  // Oxide/gate
        if (r == 1 || r == 2) return 0.6;   // Source/drain
        if (r == 3) return 1.0;              // Channel
        return 0.2;                          // Body
      }

      void main() {
        // Compute ray direction from screen coordinates
        vec4 nearPoint = uInvViewProj * vec4(vUV * 2.0 - 1.0, -1.0, 1.0);
        vec4 farPoint = uInvViewProj * vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;

        vec3 rayOrigin = nearPoint.xyz;
        vec3 rayDir = normalize(farPoint.xyz - nearPoint.xyz);

        // Ray-box intersection
        vec3 invDir = 1.0 / rayDir;
        vec3 t0 = (uBoundsMin - rayOrigin) * invDir;
        vec3 t1 = (uBoundsMax - rayOrigin) * invDir;
        vec3 tmin = min(t0, t1);
        vec3 tmax = max(t0, t1);

        float tNear = max(max(tmin.x, tmin.y), tmin.z);
        float tFar = min(min(tmax.x, tmax.y), tmax.z);

        if (tNear > tFar || tFar < 0.0) {
          fragColor = vec4(0.08, 0.08, 0.12, 1.0);
          return;
        }

        tNear = max(tNear, 0.0);

        // Ray marching
        float stepSize = (tFar - tNear) / float(uMaxSteps);
        vec3 pos = rayOrigin + rayDir * tNear;
        vec3 step = rayDir * stepSize;

        vec4 accumulatedColor = vec4(0.0);

        for (int i = 0; i < 256; i++) {
          if (i >= uMaxSteps) break;

          vec3 texCoord = (pos - uBoundsMin) / (uBoundsMax - uBoundsMin);

          if (all(greaterThanEqual(texCoord, vec3(0.0))) &&
              all(lessThanEqual(texCoord, vec3(1.0)))) {

            float concentration = texture(uVolume, texCoord).r;
            float region = texture(uRegions, texCoord).r;

            if (concentration > uDensityThreshold) {
              vec4 sampleColor = applyColormap(concentration);
              float opacity = sampleColor.a * uOpacityScale *
                             getRegionOpacity(region) *
                             (0.3 + concentration * 0.7);

              sampleColor.a = opacity;
              sampleColor.rgb *= sampleColor.a;
              accumulatedColor += sampleColor * (1.0 - accumulatedColor.a);

              if (accumulatedColor.a > 0.95) break;
            }
          }

          pos += step;
        }

        // Background color
        vec3 bgColor = vec3(0.08, 0.08, 0.12);
        fragColor = vec4(
          accumulatedColor.rgb + bgColor * (1.0 - accumulatedColor.a),
          1.0
        );
      }
    `;

    this.program = createProgram(gl, vertexSource, fragmentSource);

    // Get uniform locations
    this.uniforms = {
      uVolume: gl.getUniformLocation(this.program, 'uVolume'),
      uColormap: gl.getUniformLocation(this.program, 'uColormap'),
      uRegions: gl.getUniformLocation(this.program, 'uRegions'),
      uBoundsMin: gl.getUniformLocation(this.program, 'uBoundsMin'),
      uBoundsMax: gl.getUniformLocation(this.program, 'uBoundsMax'),
      uCameraPos: gl.getUniformLocation(this.program, 'uCameraPos'),
      uInvViewProj: gl.getUniformLocation(this.program, 'uInvViewProj'),
      uOpacityScale: gl.getUniformLocation(this.program, 'uOpacityScale'),
      uDensityThreshold: gl.getUniformLocation(this.program, 'uDensityThreshold'),
      uMaxSteps: gl.getUniformLocation(this.program, 'uMaxSteps'),
    };

    // Create colormap texture
    this.colormapTexture = createColormapTexture(gl);
  }

  initGeometry() {
    const gl = this.gl;

    // Fullscreen quad
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);

    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Vertex attribute
    const aPosition = gl.getAttribLocation(this.program, 'aPosition');
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  create3DTexture(dims, data, isFloat = true) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, texture);

    if (isFloat) {
      gl.texImage3D(
        gl.TEXTURE_3D, 0, gl.R32F,
        dims[0], dims[1], dims[2], 0,
        gl.RED, gl.FLOAT, data
      );
    } else {
      gl.texImage3D(
        gl.TEXTURE_3D, 0, gl.R8,
        dims[0], dims[1], dims[2], 0,
        gl.RED, gl.UNSIGNED_BYTE, data
      );
    }

    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

    return texture;
  }

  loadVisualizationData(data) {
    const dims = data.mesh.dimensions;
    this.dimensions = dims;

    // Decode and upload region texture
    const regionBytes = atob(data.mesh.regions);
    const regionData = new Uint8Array(regionBytes.length);
    for (let i = 0; i < regionBytes.length; i++) {
      regionData[i] = regionBytes.charCodeAt(i);
    }
    this.regionTexture = this.create3DTexture(dims, regionData, false);

    // Pre-load all animation frames as 3D textures
    this.frameTextures = data.animation.frames.map(frameB64 => {
      const frameBytes = atob(frameB64);
      const byteArray = new Uint8Array(frameBytes.length);
      for (let i = 0; i < frameBytes.length; i++) {
        byteArray[i] = frameBytes.charCodeAt(i);
      }
      const frameData = new Float32Array(byteArray.buffer);
      return this.create3DTexture(dims, frameData, true);
    });

    this.parameterValues = data.animation.parameter_values;

    // Set bounds
    const b = data.mesh.bounds;
    this.bounds = {
      min: [b[0][0], b[1][0], b[2][0]],
      max: [b[0][1], b[1][1], b[2][1]]
    };

    // Fit camera to bounds
    this.camera.fitToBounds(this.bounds.min, this.bounds.max);
  }

  setFrame(frameIndex) {
    this.animationFrame = Math.max(0,
      Math.min(Math.floor(frameIndex), this.frameTextures.length - 1)
    );
    this.needsRender = true;
  }

  play() {
    this.isPlaying = true;
  }

  pause() {
    this.isPlaying = false;
  }

  getCurrentParameterValue() {
    return this.parameterValues[Math.floor(this.animationFrame)] || 0;
  }

  getNumFrames() {
    return this.frameTextures.length;
  }

  render() {
    const gl = this.gl;

    if (!this.bounds || this.frameTextures.length === 0) {
      return;
    }

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.08, 0.08, 0.12, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    // Update camera matrices
    const aspect = this.canvas.width / this.canvas.height;
    const viewMatrix = this.camera.getViewMatrix();
    const projMatrix = this.camera.getProjectionMatrix(aspect);

    // Compute inverse view-projection matrix
    const viewProj = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        viewProj[i * 4 + j] = 0;
        for (let k = 0; k < 4; k++) {
          viewProj[i * 4 + j] += projMatrix[k * 4 + j] * viewMatrix[i * 4 + k];
        }
      }
    }

    // Invert the matrix (simplified for this use case)
    const invViewProj = this.invertMatrix4(viewProj);

    // Set uniforms
    gl.uniform3fv(this.uniforms.uBoundsMin, this.bounds.min);
    gl.uniform3fv(this.uniforms.uBoundsMax, this.bounds.max);
    gl.uniform3fv(this.uniforms.uCameraPos, this.camera.getEyePosition());
    gl.uniformMatrix4fv(this.uniforms.uInvViewProj, false, invViewProj);
    gl.uniform1f(this.uniforms.uOpacityScale, 1.5);
    gl.uniform1f(this.uniforms.uDensityThreshold, 0.05);
    gl.uniform1i(this.uniforms.uMaxSteps, 128);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.frameTextures[Math.floor(this.animationFrame)]);
    gl.uniform1i(this.uniforms.uVolume, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colormapTexture);
    gl.uniform1i(this.uniforms.uColormap, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, this.regionTexture);
    gl.uniform1i(this.uniforms.uRegions, 2);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.needsRender = false;
  }

  invertMatrix4(m) {
    const inv = new Float32Array(16);

    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (Math.abs(det) < 1e-10) det = 1;
    det = 1.0 / det;

    for (let i = 0; i < 16; i++) {
      inv[i] *= det;
    }

    return inv;
  }

  startRenderLoop() {
    const animate = () => {
      if (this.isPlaying && this.frameTextures.length > 0) {
        this.animationFrame = (this.animationFrame + this.playbackSpeed * 0.05) %
                              this.frameTextures.length;
        this.needsRender = true;
      }

      if (this.needsRender) {
        this.render();
      }

      this.animationId = requestAnimationFrame(animate);
    };
    animate();
  }

  stopRenderLoop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  attachCamera() {
    this.camera.attachToCanvas(this.canvas, () => {
      this.needsRender = true;
    });
  }

  destroy() {
    this.stopRenderLoop();
    const gl = this.gl;

    // Clean up textures
    this.frameTextures.forEach(tex => gl.deleteTexture(tex));
    if (this.regionTexture) gl.deleteTexture(this.regionTexture);
    if (this.colormapTexture) gl.deleteTexture(this.colormapTexture);

    // Clean up buffers
    if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.program) gl.deleteProgram(this.program);
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MOSFETRenderer };
}
