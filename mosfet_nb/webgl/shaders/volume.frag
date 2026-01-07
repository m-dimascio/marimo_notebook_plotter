#version 300 es
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;        // 3D concentration texture
uniform sampler2D uColormap;      // 1D colormap as 2D texture (Nx1)
uniform sampler3D uRegions;       // Device region IDs

uniform vec3 uVolumeSize;         // Volume dimensions
uniform vec3 uBoundsMin;          // World-space bounds
uniform vec3 uBoundsMax;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uCameraPos;

uniform float uOpacityScale;      // Overall opacity multiplier
uniform float uDensityThreshold;  // Minimum density to render
uniform int uMaxSteps;            // Ray marching steps

in vec3 vRayDir;
out vec4 fragColor;

// Colormap lookup with smooth interpolation
vec4 applyColormap(float value) {
    return texture(uColormap, vec2(clamp(value, 0.0, 1.0), 0.5));
}

// Region-based opacity modulation
float getRegionOpacity(int region) {
    // 0=body, 1=source, 2=drain, 3=channel, 4=oxide, 5=gate
    if (region == 4 || region == 5) return 0.2;  // Oxide/gate semi-transparent
    if (region == 1 || region == 2) return 0.8;  // Source/drain
    if (region == 3) return 1.0;                  // Channel - full opacity
    return 0.3;                                   // Body - subtle
}

void main() {
    vec3 rayDir = normalize(vRayDir);

    // Ray-box intersection
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (uBoundsMin - uCameraPos) * invDir;
    vec3 t1 = (uBoundsMax - uCameraPos) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);

    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);

    if (tNear > tFar || tFar < 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    tNear = max(tNear, 0.0);

    // Ray marching
    float stepSize = (tFar - tNear) / float(uMaxSteps);
    vec3 pos = uCameraPos + rayDir * tNear;
    vec3 step = rayDir * stepSize;

    vec4 accumulatedColor = vec4(0.0);

    for (int i = 0; i < uMaxSteps; i++) {
        // Convert world position to texture coordinates
        vec3 texCoord = (pos - uBoundsMin) / (uBoundsMax - uBoundsMin);

        if (all(greaterThanEqual(texCoord, vec3(0.0))) &&
            all(lessThanEqual(texCoord, vec3(1.0)))) {

            // Sample concentration (normalized 0-1)
            float concentration = texture(uVolume, texCoord).r;

            // Sample region
            int region = int(texture(uRegions, texCoord).r * 255.0);

            if (concentration > uDensityThreshold) {
                // Apply colormap
                vec4 sampleColor = applyColormap(concentration);

                // Modulate opacity by region and concentration
                float opacity = sampleColor.a * uOpacityScale *
                               getRegionOpacity(region) *
                               concentration;

                // Front-to-back compositing (smooth blending)
                sampleColor.a = opacity;
                sampleColor.rgb *= sampleColor.a;
                accumulatedColor += sampleColor * (1.0 - accumulatedColor.a);

                // Early termination when nearly opaque
                if (accumulatedColor.a > 0.95) break;
            }
        }

        pos += step;
    }

    fragColor = accumulatedColor;
}
