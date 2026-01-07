// Colormap utilities for MOSFET visualization
// Provides Viridis, Plasma, and custom scientific colormaps

// Viridis colormap - perceptually uniform, colorblind-friendly
vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777, 0.0054, 0.3340);
    const vec3 c1 = vec3(0.1050, 0.6387, 0.7246);
    const vec3 c2 = vec3(-0.3308, 1.0041, 0.5996);
    const vec3 c3 = vec3(-4.6342, -5.7991, -19.3324);
    const vec3 c4 = vec3(6.2282, 14.1799, 56.6905);
    const vec3 c5 = vec3(4.7763, -13.7451, -65.3530);
    const vec3 c6 = vec3(-5.4354, 4.6456, 26.3124);

    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// Plasma colormap - good for scientific visualization
vec3 plasma(float t) {
    const vec3 c0 = vec3(0.0504, 0.0298, 0.5280);
    const vec3 c1 = vec3(2.0207, -0.1321, -0.4099);
    const vec3 c2 = vec3(-2.4066, 0.0667, 5.1089);
    const vec3 c3 = vec3(-18.1296, 14.5847, -24.4310);
    const vec3 c4 = vec3(51.3707, -44.0880, 46.9998);
    const vec3 c5 = vec3(-62.4545, 63.1055, -46.4934);
    const vec3 c6 = vec3(27.4660, -29.6267, 17.5024);

    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// Inferno colormap - high contrast
vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002, 0.0016, 0.0139);
    const vec3 c1 = vec3(0.1066, 0.5639, 3.9327);
    const vec3 c2 = vec3(11.6024, -3.9726, -15.9422);
    const vec3 c3 = vec3(-41.7036, 17.4386, 44.3556);
    const vec3 c4 = vec3(77.1629, -33.4023, -81.8080);
    const vec3 c5 = vec3(-71.3194, 32.6261, 73.2096);
    const vec3 c6 = vec3(25.1311, -12.2428, -23.0703);

    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// Turbo colormap - rainbow-like but perceptually better
vec3 turbo(float t) {
    const vec3 c0 = vec3(0.1140, 0.0624, 0.2252);
    const vec3 c1 = vec3(0.8770, 1.9722, 2.3589);
    const vec3 c2 = vec3(-4.0987, -7.6040, 5.6847);
    const vec3 c3 = vec3(25.0548, 28.7917, -32.4825);
    const vec3 c4 = vec3(-49.7036, -55.7890, 60.6566);
    const vec3 c5 = vec3(43.7970, 48.6619, -49.0907);
    const vec3 c6 = vec3(-13.5540, -15.2685, 14.7049);

    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// MOSFET-specific colormap: Blue (low) -> Cyan -> Yellow -> Red (high)
// Emphasizes channel formation visibility
vec3 mosfetColormap(float t) {
    if (t < 0.25) {
        // Deep blue to blue
        float s = t / 0.25;
        return mix(vec3(0.0, 0.0, 0.3), vec3(0.0, 0.2, 0.8), s);
    } else if (t < 0.5) {
        // Blue to cyan
        float s = (t - 0.25) / 0.25;
        return mix(vec3(0.0, 0.2, 0.8), vec3(0.0, 0.8, 0.9), s);
    } else if (t < 0.75) {
        // Cyan to yellow
        float s = (t - 0.5) / 0.25;
        return mix(vec3(0.0, 0.8, 0.9), vec3(0.9, 0.9, 0.0), s);
    } else {
        // Yellow to red
        float s = (t - 0.75) / 0.25;
        return mix(vec3(0.9, 0.9, 0.0), vec3(1.0, 0.2, 0.0), s);
    }
}
