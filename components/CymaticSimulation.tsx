import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle, useLayoutEffect } from 'react';
import { SimulationParams } from '../types.ts';

interface Props {
  params: SimulationParams;
  isPlaying: boolean;
}

export interface SimulationHandle {
  triggerDownload: () => void;
}

const vertexShaderSource = `
  attribute vec2 a_position;
  varying vec2 v_uv;
  void main() {
    v_uv = a_position * 0.5 + 0.5; 
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const fragmentShaderSource = `
  precision highp float;

  varying vec2 v_uv;

  uniform vec2 u_resolution;
  uniform float u_time;
  
  // Physics params
  uniform float u_frequency;
  uniform float u_amplitude;
  uniform float u_damping;
  uniform float u_depth;
  uniform float u_diameter;
  
  // Lighting params
  uniform float u_ledCount;
  uniform float u_ledBrightness;
  uniform float u_camHeight;
  uniform vec3 u_liquidColor;

  // Ring 1
  uniform vec3 u_ledColor;
  uniform float u_ledSize;
  uniform float u_ledHeight;
  uniform float u_ledRadius;

  // Ring 2
  uniform vec3 u_led2Color;
  uniform float u_led2Size;
  uniform float u_led2Height;
  uniform float u_led2Radius;

  #define PI 3.14159265359
  
  // --- PHYSICS ENGINE 5.5: STABLE RENDER ---

  // Pseudo-random hash
  float hash(float n) {
      return fract(sin(n) * 43758.5453123);
  }

  // Faraday Wave Mode Selection
  float getModeFromHash(float h, float freq) {
      float val = h * 10.0;
      
      // Low Freq (< 15Hz): Simple geometries dominate
      if (freq < 15.0) {
         if (val < 5.0) return 4.0; // Square grid
         return 3.0; // Triangular
      }

      // Medium/High Freq: Competition between Hexagonal and Square
      if (val < 4.0) return 6.0; // Hexagonal
      if (val < 7.0) return 4.0; // Square
      if (val < 8.0) return 12.0; // High complexity quasicrystal
      if (val < 9.0) return 8.0; // Octagonal
      return 3.0; // Triangular fallback
  }

  // CAPILLARY DISPERSION RELATION + DEPTH EFFECT
  float getWavenumber(float freq) {
      float k_phys = pow(freq, 0.6666); 
      
      // Depth effect: Shallow water = slower waves = shorter wavelength = HIGHER k
      float depthFactor = 1.0 + (1.5 / sqrt(u_depth + 0.1));

      return k_phys * u_diameter * 0.18 * depthFactor; 
  }

  // Calculate a Standing Wave field
  float calculateStandingWave(vec2 p, float k, float t, float N, float seed) {
      float h = 0.0;
      
      // Static rotation fixed to 0.0 to align patterns vertically
      float staticRot = 0.0; 

      for(float i = 0.0; i < 12.0; i++) {
          if(i >= N) break;
          
          float angle = staticRot + (i / N) * PI; 
          vec2 dir = vec2(cos(angle), sin(angle));
          
          // Standing Wave Function
          float spatial = dot(p, dir) * k;
          h += cos(spatial) * cos(t);
      }
      
      return h / N; 
  }

  float getSurfaceHeight(vec2 p) {
      float r = length(p);
      
      // Hard wall boundary (Physics: Radius = 1.0)
      if (r > 1.0) return 0.0;
      
      // IF FREQUENCY IS ZERO, LIQUID IS FLAT
      if (u_frequency < 0.1) return 0.0;

      // Mode Hopping Logic
      float effectiveFreq = u_frequency + (u_depth * 0.7); 
      float stabilityScale = 0.5; 
      float f_scaled = effectiveFreq * stabilityScale;
      
      float f_index = floor(f_scaled);
      float f_fract = smoothstep(0.4, 0.6, fract(f_scaled)); 
      
      float seedA = f_index * 12.34;
      float nA = getModeFromHash(hash(seedA), u_frequency);
      
      float seedB = (f_index + 1.0) * 12.34;
      float nB = getModeFromHash(hash(seedB), u_frequency);

      float k = getWavenumber(u_frequency);
      
      // Base vibration speed
      float w = u_time * u_frequency * 1.5;

      // 1. MAIN WAVE STRUCTURE
      float hA = calculateStandingWave(p, k, w, nA, seedA);
      float hB = calculateStandingWave(p, k, w, nB, seedB);
      float mainWave = mix(hA, hB, f_fract);

      // 2. MICRO-RESONANCE (Harmonics)
      float k_micro = k * 3.0; 
      float w_micro = w * 1.2; 
      
      float hMicroA = calculateStandingWave(p, k_micro, w_micro, 12.0, seedA + 33.1);
      float hMicroB = calculateStandingWave(p, k_micro, w_micro, 12.0, seedB + 33.1);
      float microWave = mix(hMicroA, hMicroB, f_fract);

      // Combine
      float rawHeight = mainWave + (microWave * 0.2);

      // Physical Damping (Reduced at edge to allow bounce)
      // We use a softer curve so energy hits the wall
      float bottomFriction = 1.0 + (1.0 / (u_depth + 0.1));
      float damping = 1.0 - (u_damping * 0.5 * r * r * bottomFriction);
      
      // Trochoidal Wave Shaping (Peaking)
      float sharp = exp(1.8 * (rawHeight - 0.2));
      
      // Static Meniscus (Surface Tension climbing the wall)
      // This adds a permanent curve at the very edge, simulating the contact angle
      float staticMeniscus = smoothstep(0.95, 1.0, r) * 0.2;

      return ((sharp - 0.5) * u_amplitude * damping) + staticMeniscus;
  }

  vec3 getNormal(vec2 p, float h) {
      vec2 e = vec2(0.001, 0.0);
      float hx = getSurfaceHeight(p + e.xy);
      float hy = getSurfaceHeight(p + e.yx);
      return normalize(vec3(h - hx, h - hy, e.x * 0.8));
  }

  // Generalized function for any ring with specific DOT SIZE
  float getLedRing(vec3 ro, vec3 rd, float ringRadius, float ringHeight, float dotSizeParam) {
      if (abs(rd.z) < 0.001) return 0.0; 
      float t = (ringHeight - ro.z) / rd.z;
      if (t < 0.0) return 0.0; 
      
      vec3 hit = ro + rd * t;
      float r = length(hit.xy);
      
      float distToRing = abs(r - ringRadius);
      float glow = exp(-distToRing * 40.0); 
      
      float angle = atan(hit.y, hit.x);
      float ledPhase = (angle / (2.0 * PI)) * u_ledCount;
      float ledLocal = fract(ledPhase);
      
      // LED Size control
      float dotSize = dotSizeParam * 0.5; 
      float dot = smoothstep(dotSize + 0.1, dotSize, abs(ledLocal - 0.5));
      
      float continuity = smoothstep(48.0, 120.0, u_ledCount);
      
      return glow * mix(dot, 1.0, continuity) * u_ledBrightness * step(distToRing, 0.15);
  }

  void main() {
      vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.y, u_resolution.x);
      uv *= 2.0; 

      // Hard Physical Clip at Container Edge
      if (length(uv) > 1.0) {
          gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
          return;
      }

      float h = getSurfaceHeight(uv);
      vec3 pos = vec3(uv, h * 0.15); 
      vec3 norm = getNormal(uv, h);

      vec3 camPos = vec3(0.0, 0.0, u_camHeight);
      vec3 viewDir = normalize(pos - camPos);
      vec3 reflDir = reflect(viewDir, norm);
      
      // Chromatic Aberration setup
      vec3 dirR = normalize(reflDir + vec3(0.004, 0.0, 0.0));
      vec3 dirG = reflDir;
      vec3 dirB = normalize(reflDir - vec3(0.004, 0.0, 0.0));

      // Ring 1 Calculations
      float r1 = getLedRing(pos, dirR, u_ledRadius, u_ledHeight, u_ledSize);
      float g1 = getLedRing(pos, dirG, u_ledRadius, u_ledHeight, u_ledSize);
      float b1 = getLedRing(pos, dirB, u_ledRadius, u_ledHeight, u_ledSize);
      vec3 col1 = vec3(r1, g1, b1) * u_ledColor;

      // Ring 2 Calculations
      float r2 = getLedRing(pos, dirR, u_led2Radius, u_led2Height, u_led2Size);
      float g2 = getLedRing(pos, dirG, u_led2Radius, u_led2Height, u_led2Size);
      float b2 = getLedRing(pos, dirB, u_led2Radius, u_led2Height, u_led2Size);
      vec3 col2 = vec3(r2, g2, b2) * u_led2Color;

      vec3 reflection = col1 + col2;
      
      // Mask reflections near edge (LEDs shouldn't touch wall)
      // But we KEEP the liquid surface visible
      float reflectionMask = smoothstep(0.96, 0.90, length(uv));
      reflection *= reflectionMask;

      // Opaque Liquid Shading (Black Mirror)
      float fresnel = pow(1.0 - max(0.0, dot(viewDir, norm)), 3.0);
      vec3 baseColor = u_liquidColor * 0.05; 

      vec3 finalColor = mix(baseColor, u_liquidColor, fresnel * 0.3) + reflection;

      gl_FragColor = vec4(finalColor, 1.0);
  }
`;

export const CymaticSimulation = forwardRef<SimulationHandle, Props>(({ params, isPlaying }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  
  const paramsRef = useRef(params);
  const isPlayingRef = useRef(isPlaying);
  const timeRef = useRef(0);
  const drawSceneRef = useRef<((time: number) => void) | null>(null);

  useLayoutEffect(() => {
    paramsRef.current = params;
  }, [params]);

  useLayoutEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useImperativeHandle(ref, () => ({
    triggerDownload: () => {
      const canvas = canvasRef.current;
      const drawScene = drawSceneRef.current;
      if (canvas && drawScene) {
        const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });
        if(!gl) return;

        const originalWidth = canvas.width;
        const originalHeight = canvas.height;
        
        const hdWidth = 3840;
        const aspect = originalWidth / originalHeight;
        const hdHeight = Math.round(hdWidth / aspect);
        
        canvas.width = hdWidth;
        canvas.height = hdHeight;
        gl.viewport(0, 0, hdWidth, hdHeight);
        
        drawScene(timeRef.current);
        
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `cymatics_HD_${timestamp}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
        
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        gl.viewport(0, 0, originalWidth, originalHeight);
        drawScene(timeRef.current);
      }
    }
  }));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });
    if (!gl) {
      setError("WebGL non supportato.");
      return;
    }

    const createShader = (gl: WebGLRenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader Error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    if (!vertexShader || !fragmentShader) {
      setError("Errore compilazione shader.");
      return;
    }

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      setError("Errore link programma.");
      return;
    }
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
      -1.0,  1.0,
      -1.0,  1.0,
       1.0, -1.0,
       1.0,  1.0,
    ]), gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const uLoc = {
      res: gl.getUniformLocation(program, "u_resolution"),
      time: gl.getUniformLocation(program, "u_time"),
      freq: gl.getUniformLocation(program, "u_frequency"),
      amp: gl.getUniformLocation(program, "u_amplitude"),
      damp: gl.getUniformLocation(program, "u_damping"),
      depth: gl.getUniformLocation(program, "u_depth"),
      diam: gl.getUniformLocation(program, "u_diameter"),
      lCount: gl.getUniformLocation(program, "u_ledCount"),
      lBright: gl.getUniformLocation(program, "u_ledBrightness"),
      
      // Ring 1
      lCol: gl.getUniformLocation(program, "u_ledColor"),
      lSize: gl.getUniformLocation(program, "u_ledSize"),
      lHeight: gl.getUniformLocation(program, "u_ledHeight"),
      lRad: gl.getUniformLocation(program, "u_ledRadius"),
      
      // Ring 2
      l2Col: gl.getUniformLocation(program, "u_led2Color"),
      l2Size: gl.getUniformLocation(program, "u_led2Size"),
      l2Height: gl.getUniformLocation(program, "u_led2Height"),
      l2Rad: gl.getUniformLocation(program, "u_led2Radius"),

      cHeight: gl.getUniformLocation(program, "u_camHeight"),
      wCol: gl.getUniformLocation(program, "u_liquidColor"),
    };

    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255
      ] : [1, 1, 1];
    };

    const drawScene = (currentTime: number) => {
      gl.viewport(0, 0, canvas.width, canvas.height);
      
      const p = paramsRef.current;

      gl.uniform2f(uLoc.res, canvas.width, canvas.height);
      gl.uniform1f(uLoc.time, currentTime);
      gl.uniform1f(uLoc.freq, p.frequency);
      gl.uniform1f(uLoc.amp, p.amplitude);
      gl.uniform1f(uLoc.damp, p.damping);
      gl.uniform1f(uLoc.depth, p.depth);
      gl.uniform1f(uLoc.diam, p.diameter);
      gl.uniform1f(uLoc.lCount, p.ledCount);
      gl.uniform1f(uLoc.lBright, p.ledBrightness);
      gl.uniform1f(uLoc.cHeight, p.cameraHeight);
      
      // Ring 1 uniforms
      gl.uniform1f(uLoc.lSize, p.ledSize);
      gl.uniform1f(uLoc.lHeight, p.ledHeight);
      const containerRadius = p.diameter / 2.0;
      gl.uniform1f(uLoc.lRad, p.ledRadius / containerRadius);
      const lC1 = hexToRgb(p.ledColor);
      gl.uniform3f(uLoc.lCol, lC1[0], lC1[1], lC1[2]);

      // Ring 2 uniforms
      gl.uniform1f(uLoc.l2Size, p.led2Size);
      gl.uniform1f(uLoc.l2Height, p.led2Height);
      gl.uniform1f(uLoc.l2Rad, p.led2Radius / containerRadius);
      const lC2 = hexToRgb(p.led2Color);
      gl.uniform3f(uLoc.l2Col, lC2[0], lC2[1], lC2[2]);
      
      const wC = hexToRgb(p.liquidColor);
      gl.uniform3f(uLoc.wCol, wC[0], wC[1], wC[2]);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };

    drawSceneRef.current = drawScene;

    let animId: number;
    let lastFrameTime = performance.now();
    
    // Flag to force-kill zombie loops in StrictMode
    let isMounted = true;

    const renderLoop = (now: number) => {
      if (!isMounted) return;
      if (!canvas) return;
      
      let dt = (now - lastFrameTime) * 0.001;
      if (dt > 0.1) dt = 0.1; 
      
      lastFrameTime = now;

      if (isPlayingRef.current) {
        timeRef.current += dt * paramsRef.current.simulationSpeed;
      }

      const displayWidth = canvas.clientWidth;
      const displayHeight = canvas.clientHeight;
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }

      drawScene(timeRef.current);
      animId = requestAnimationFrame(renderLoop);
    };

    renderLoop(lastFrameTime);

    return () => {
      isMounted = false;
      cancelAnimationFrame(animId);
      if (program) gl.deleteProgram(program);
    };
  }, []); 

  if (error) return <div className="text-red-500 p-4">{error}</div>;

  return <canvas ref={canvasRef} className="w-full h-full block" />;
});