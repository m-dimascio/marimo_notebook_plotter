#version 300 es
precision highp float;

in vec3 aPosition;
out vec3 vRayDir;
out vec3 vPosition;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uCameraPos;

void main() {
    vPosition = aPosition;
    vRayDir = aPosition - uCameraPos;
    gl_Position = uProjectionMatrix * uViewMatrix * vec4(aPosition, 1.0);
}
