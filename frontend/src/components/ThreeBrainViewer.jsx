import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { getZoneColor } from '../zonePalette'

const INFERNO_STOPS = [
  [0.0, '#000004'],
  [0.12, '#1b0c41'],
  [0.25, '#4a0c6b'],
  [0.38, '#781c6d'],
  [0.5, '#a52c60'],
  [0.62, '#cf4446'],
  [0.74, '#ed6925'],
  [0.86, '#fb9b06'],
  [1.0, '#fcfdbf'],
]

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function smoothstep(edge0, edge1, value) {
  if (edge1 <= edge0) {
    return value >= edge1 ? 1 : 0
  }
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return x * x * (3 - 2 * x)
}

function decodeBase64ToUint8(base64Value) {
  const binary = window.atob(base64Value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

function decodeNormalizedBytes(base64Value) {
  const bytes = decodeBase64ToUint8(base64Value)
  const values = new Float32Array(bytes.length)
  for (let index = 0; index < bytes.length; index += 1) {
    values[index] = bytes[index] / 255
  }
  return values
}

function percentile(values, ratio) {
  if (!values.length) {
    return 0
  }
  const sorted = Array.from(values).sort((left, right) => left - right)
  const position = clamp(ratio, 0, 1) * (sorted.length - 1)
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) {
    return sorted[lower]
  }
  const mix = position - lower
  return sorted[lower] * (1 - mix) + sorted[upper] * mix
}

function toDisplayCoordinates(coords) {
  const converted = new Float32Array(coords.length * 3)
  coords.forEach(([x, y, z], index) => {
    const offset = index * 3
    converted[offset] = x
    converted[offset + 1] = z
    converted[offset + 2] = -y
  })
  return converted
}

function buildContactShadowTexture() {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 256
  const context = canvas.getContext('2d')
  if (!context) {
    return null
  }

  const gradient = context.createRadialGradient(128, 128, 28, 128, 128, 120)
  gradient.addColorStop(0, 'rgba(2, 8, 12, 0.42)')
  gradient.addColorStop(0.45, 'rgba(2, 8, 12, 0.18)')
  gradient.addColorStop(0.76, 'rgba(2, 8, 12, 0.05)')
  gradient.addColorStop(1, 'rgba(2, 8, 12, 0)')

  context.fillStyle = gradient
  context.fillRect(0, 0, canvas.width, canvas.height)

  const texture = new THREE.CanvasTexture(canvas)
  texture.generateMipmaps = false
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter
  return texture
}

function buildRimMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      rimColor: { value: new THREE.Color('#7bdcff') },
      rimOpacity: { value: 0.12 },
      rimPower: { value: 3.6 },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 rimColor;
      uniform float rimOpacity;
      uniform float rimPower;
      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        vec3 viewDirection = normalize(vViewPosition);
        float rim = pow(1.0 - max(dot(normalize(vNormal), viewDirection), 0.0), rimPower);
        float alpha = rim * rimOpacity;
        if (alpha < 0.0025) {
          discard;
        }
        gl_FragColor = vec4(rimColor * rim, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
  })
}

function buildSignalMaterial({
  opacity,
  intensity,
  fresnelStrength,
  fresnelPower,
  floorStrength,
  displacementScale = 0,
  motionAmount = 0,
  blending,
  polygonOffset = false,
  side = THREE.DoubleSide,
}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: opacity },
      intensity: { value: intensity },
      fresnelStrength: { value: fresnelStrength },
      fresnelPower: { value: fresnelPower },
      floorStrength: { value: floorStrength },
      displacementScale: { value: displacementScale },
      time: { value: 0 },
      motionAmount: { value: motionAmount },
    },
    vertexShader: `
      attribute vec3 signalColor;
      attribute float signalStrength;

      varying vec3 vSignalColor;
      varying float vSignalStrength;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec3 vLocalPosition;

      uniform float displacementScale;
      uniform float time;
      uniform float motionAmount;

      void main() {
        vSignalColor = signalColor;
        vSignalStrength = signalStrength;
        float strength = clamp(signalStrength, 0.0, 1.0);
        float speed = 0.55 + pow(strength, 0.72) * 2.35;
        float motionWave =
          sin(position.x * 4.8 + time * (0.8 + speed * 1.05)) * 0.28 +
          sin(position.y * 5.6 - time * (0.6 + speed * 0.78)) * 0.22 +
          sin(position.z * 4.2 + time * (0.48 + speed * 0.56)) * 0.16;
        float displacement =
          pow(strength, 1.45) *
          displacementScale *
          (1.0 + motionWave * motionAmount * 1.6);
        vec3 displacedPosition = position + normal * displacement;
        vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewPosition = -mvPosition.xyz;
        vLocalPosition = displacedPosition;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float opacity;
      uniform float intensity;
      uniform float fresnelStrength;
      uniform float fresnelPower;
      uniform float floorStrength;
      uniform float time;
      uniform float motionAmount;

      varying vec3 vSignalColor;
      varying float vSignalStrength;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec3 vLocalPosition;

      void main() {
        float strength = clamp(vSignalStrength, 0.0, 1.0);
        if (strength <= 0.001) {
          discard;
        }

        vec3 viewDirection = normalize(vViewPosition);
        float facing = max(dot(normalize(vNormal), viewDirection), 0.0);
        float fresnel = pow(1.0 - facing, fresnelPower);
        float body = smoothstep(floorStrength, 1.0, strength);
        float speed = 0.5 + pow(strength, 0.72) * 2.0;
        float shimmerWave =
          sin(vLocalPosition.x * 5.6 + time * (1.1 + speed * 1.5)) * 0.5 +
          sin(vLocalPosition.y * 7.4 - time * (0.9 + speed * 1.18)) * 0.32 +
          sin(vLocalPosition.z * 4.1 + time * (0.68 + speed * 0.82)) * 0.18;
        float shimmer = 1.0 + shimmerWave * motionAmount;
        float alpha = (body * 0.76 + fresnel * fresnelStrength * strength) * opacity * shimmer;
        vec3 color = vSignalColor * (0.42 + strength * intensity + fresnel * fresnelStrength * 0.2) * shimmer;

        if (alpha < 0.0025) {
          discard;
        }

        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side,
    blending,
    toneMapped: false,
    polygonOffset,
    polygonOffsetFactor: polygonOffset ? -2 : 0,
    polygonOffsetUnits: polygonOffset ? -4 : 0,
  })
}

function buildParticleMaterial({
  opacity,
  intensity,
  size,
  lift,
}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: opacity },
      intensity: { value: intensity },
      size: { value: size },
      lift: { value: lift },
      time: { value: 0 },
    },
    vertexShader: `
      attribute vec3 signalColor;
      attribute float signalStrength;

      varying vec3 vSignalColor;
      varying float vSignalStrength;

      uniform float size;
      uniform float lift;
      uniform float time;

      void main() {
        float strength = clamp(signalStrength, 0.0, 1.0);
        float speed = 0.52 + pow(strength, 0.68) * 3.4;
        float pulse =
          0.82 +
          0.18 * sin(
            time * (1.2 + speed * 1.18) +
            position.x * 3.4 +
            position.y * 4.1 +
            position.z * 2.7
          );
        vec3 normalDirection = normalize(normal);
        vec3 referenceAxis =
          abs(normalDirection.y) > 0.84 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        vec3 tangent = normalize(cross(normalDirection, referenceAxis));
        vec3 bitangent = normalize(cross(normalDirection, tangent));
        float swirlA =
          sin(time * (0.95 + speed * 0.9) + position.x * 5.1 + position.z * 3.7);
        float swirlB =
          cos(time * (1.25 + speed * 1.08) + position.y * 4.2 - position.x * 2.9);
        vec3 displacedPosition =
          position +
          normalDirection * lift * (0.55 + strength * 1.9) +
          tangent * swirlA * lift * (0.045 + strength * 0.08) +
          bitangent * swirlB * lift * (0.03 + strength * 0.06);
        vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);

        vSignalColor = signalColor;
        vSignalStrength = strength;

        gl_PointSize =
          max(0.0, (1.8 + pow(strength, 0.82) * size * pulse) * (280.0 / max(1.0, -mvPosition.z)));
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float opacity;
      uniform float intensity;

      varying vec3 vSignalColor;
      varying float vSignalStrength;

      void main() {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float radius = dot(centered, centered);
        if (radius > 1.0) {
          discard;
        }

        float halo = smoothstep(1.0, 0.02, radius);
        float core = smoothstep(0.22, 0.0, sqrt(radius));
        float alpha = (halo * 0.48 + core * 0.92) * opacity * vSignalStrength;
        vec3 color =
          vSignalColor * (0.6 + vSignalStrength * intensity) +
          vec3(core * vSignalStrength * 0.42);

        if (alpha < 0.0025) {
          discard;
        }

        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  })
}

function buildConstellationMaterial({
  opacity,
  size,
}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: opacity },
      size: { value: size },
      time: { value: 0 },
    },
    vertexShader: `
      attribute vec3 color;

      varying vec3 vColor;

      uniform float size;
      uniform float time;

      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        float shimmer =
          0.94 +
          0.06 * sin(
            time * 0.72 +
            position.x * 3.2 +
            position.y * 4.6 +
            position.z * 2.8
          );
        vColor = color;
        gl_PointSize = max(0.0, size * shimmer * (220.0 / max(1.0, -mvPosition.z)));
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float opacity;

      varying vec3 vColor;

      void main() {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float radius = dot(centered, centered);
        if (radius > 1.0) {
          discard;
        }

        float halo = smoothstep(1.0, 0.04, radius);
        float core = smoothstep(0.18, 0.0, sqrt(radius));
        float alpha = (halo * 0.34 + core * 0.48) * opacity;
        vec3 color = vColor * (0.82 + core * 0.32);

        if (alpha < 0.0025) {
          discard;
        }

        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  })
}

function buildFilamentMaterial({ opacity }) {
  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: opacity },
    },
    vertexShader: `
      attribute vec3 color;
      attribute float signalStrength;

      varying vec3 vColor;
      varying float vSignalStrength;

      void main() {
        vColor = color;
        vSignalStrength = clamp(signalStrength, 0.0, 1.0);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float opacity;

      varying vec3 vColor;
      varying float vSignalStrength;

      void main() {
        float alpha = pow(vSignalStrength, 1.12) * opacity;
        if (alpha < 0.0025) {
          discard;
        }
        gl_FragColor = vec4(vColor * (0.45 + vSignalStrength * 1.35), alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  })
}

function buildSmoothingCache(faces, vertexCount) {
  const counts = new Float32Array(vertexCount)

  for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
    const face = faces[faceIndex]
    counts[face[0]] += 1
    counts[face[1]] += 1
    counts[face[2]] += 1
  }

  for (let index = 0; index < counts.length; index += 1) {
    if (counts[index] === 0) {
      counts[index] = 1
    }
  }

  return { faces, counts }
}

function smoothVertexValues(values, smoothingCache, passes = 1, blend = 0.32) {
  if (!smoothingCache || passes <= 0 || blend <= 0) {
    return Float32Array.from(values)
  }

  const { faces, counts } = smoothingCache
  let current = Float32Array.from(values)
  const vertexMeans = new Float32Array(current.length)

  for (let pass = 0; pass < passes; pass += 1) {
    vertexMeans.fill(0)
    for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
      const [left, center, right] = faces[faceIndex]
      const faceMean = (current[left] + current[center] + current[right]) / 3
      vertexMeans[left] += faceMean
      vertexMeans[center] += faceMean
      vertexMeans[right] += faceMean
    }
    for (let index = 0; index < current.length; index += 1) {
      const averaged = vertexMeans[index] / counts[index]
      current[index] = current[index] * (1 - blend) + averaged * blend
    }
  }

  return current
}

function sampleInferno(value) {
  const clamped = clamp(value, 0, 1)
  for (let index = 1; index < INFERNO_STOPS.length; index += 1) {
    const [rightStop, rightHex] = INFERNO_STOPS[index]
    if (clamped <= rightStop) {
      const [leftStop, leftHex] = INFERNO_STOPS[index - 1]
      const range = Math.max(rightStop - leftStop, 1e-6)
      const mix = (clamped - leftStop) / range
      const leftColor = new THREE.Color(leftHex)
      const rightColor = new THREE.Color(rightHex)
      return leftColor.lerp(rightColor, mix)
    }
  }
  return new THREE.Color(INFERNO_STOPS[INFERNO_STOPS.length - 1][1])
}

function buildAnatomyColorArray(bgValues, vertexCount) {
  const colors = new Float32Array(vertexCount * 3)
  const shadow = new THREE.Color('#111821')
  const midtone = new THREE.Color('#202b35')
  const highlight = new THREE.Color('#41505f')

  for (let index = 0; index < vertexCount; index += 1) {
    const tone = bgValues?.[index] ?? 0.5
    const shaped = smoothstep(0.06, 0.94, tone)
    const midMix = Math.min(shaped * 1.18, 1)
    const hiMix = Math.pow(shaped, 2.0) * 0.56
    const red =
      shadow.r * (1 - midMix) +
      midtone.r * midMix * (1 - hiMix) +
      highlight.r * hiMix
    const green =
      shadow.g * (1 - midMix) +
      midtone.g * midMix * (1 - hiMix) +
      highlight.g * hiMix
    const blue =
      shadow.b * (1 - midMix) +
      midtone.b * midMix * (1 - hiMix) +
      highlight.b * hiMix
    const offset = index * 3
    colors[offset] = red
    colors[offset + 1] = green
    colors[offset + 2] = blue
  }

  return colors
}

function buildConstellationColorArray(anatomyColors) {
  const colors = new Float32Array(anatomyColors.length)
  for (let index = 0; index < anatomyColors.length; index += 3) {
    colors[index] = clamp(anatomyColors[index] * 0.72 + 0.12, 0, 1)
    colors[index + 1] = clamp(anatomyColors[index + 1] * 0.8 + 0.18, 0, 1)
    colors[index + 2] = clamp(anatomyColors[index + 2] * 1.06 + 0.28, 0, 1)
  }
  return colors
}

function buildInteriorConstellationColorArray(anatomyColors) {
  const colors = new Float32Array(anatomyColors.length)
  for (let index = 0; index < anatomyColors.length; index += 3) {
    colors[index] = clamp(anatomyColors[index] * 0.46 + 0.08, 0, 1)
    colors[index + 1] = clamp(anatomyColors[index + 1] * 0.7 + 0.16, 0, 1)
    colors[index + 2] = clamp(anatomyColors[index + 2] * 1.18 + 0.34, 0, 1)
  }
  return colors
}

function buildSignalPayload(activationValues, smoothingCache) {
  const floor = percentile(activationValues, 0.72)
  const ceiling = Math.max(percentile(activationValues, 0.995), floor + 1e-5)
  const smoothedActivationValues = smoothVertexValues(activationValues, smoothingCache, 6, 0.38)
  const smoothedFloor = percentile(smoothedActivationValues, 0.5)
  const smoothedCeiling = Math.max(percentile(smoothedActivationValues, 0.985), smoothedFloor + 1e-5)
  const surfaceColors = new Float32Array(activationValues.length * 3)
  const surfaceStrengths = new Float32Array(activationValues.length)
  const shellColors = new Float32Array(activationValues.length * 3)
  const shellStrengths = new Float32Array(activationValues.length)
  const scatterColors = new Float32Array(activationValues.length * 3)
  const scatterStrengths = new Float32Array(activationValues.length)

  for (let index = 0; index < activationValues.length; index += 1) {
    const activation = activationValues[index]
    const normalized = smoothstep(floor, ceiling, activation)
    const smoothedNormalized = smoothstep(smoothedFloor, smoothedCeiling, smoothedActivationValues[index])
    const surfaceStrength = Math.pow(normalized, 1.24)
    const shellStrength = Math.pow(normalized, 0.88)
    const scatterStrength = Math.pow(smoothedNormalized, 1.08) * 0.78

    surfaceStrengths[index] = surfaceStrength
    shellStrengths[index] = shellStrength
    scatterStrengths[index] = scatterStrength

    const sourceIndex = index * 3
    const surfaceColor = sampleInferno(normalized)
    const shellColor = sampleInferno(Math.min(1, normalized * 0.92 + 0.05))
    const scatterColor = sampleInferno(Math.min(1, smoothedNormalized * 0.86 + 0.08))

    if (shellStrength > 0.0005) {
      const shellScale = 0.2 + shellStrength * 0.64
      surfaceColors[sourceIndex] = clamp(surfaceColor.r * (0.18 + surfaceStrength * 0.82), 0, 1)
      surfaceColors[sourceIndex + 1] = clamp(
        surfaceColor.g * (0.18 + surfaceStrength * 0.82),
        0,
        1,
      )
      surfaceColors[sourceIndex + 2] = clamp(
        surfaceColor.b * (0.18 + surfaceStrength * 0.82),
        0,
        1,
      )

      shellColors[sourceIndex] = clamp(shellColor.r * shellScale, 0, 1)
      shellColors[sourceIndex + 1] = clamp(shellColor.g * shellScale, 0, 1)
      shellColors[sourceIndex + 2] = clamp(shellColor.b * shellScale, 0, 1)
    }

    if (scatterStrength > 0.0005) {
      const scatterScale = 0.1 + scatterStrength * 0.44
      scatterColors[sourceIndex] = clamp(scatterColor.r * scatterScale, 0, 1)
      scatterColors[sourceIndex + 1] = clamp(scatterColor.g * scatterScale, 0, 1)
      scatterColors[sourceIndex + 2] = clamp(scatterColor.b * scatterScale, 0, 1)
    }
  }

  return {
    surfaceColors,
    surfaceStrengths,
    shellColors,
    shellStrengths,
    scatterColors,
    scatterStrengths,
  }
}

function getFrameSignalPayload(frame, cache, smoothingCache) {
  if (!frame?.activation_b64) {
    return null
  }
  let payload = cache.get(frame.activation_b64)
  if (!payload) {
    payload = buildSignalPayload(decodeNormalizedBytes(frame.activation_b64), smoothingCache)
    cache.set(frame.activation_b64, payload)
  }
  return payload
}

function blendSignalPayloads(fromPayload, toPayload, mix, target) {
  const inverse = 1 - mix
  const surfaceColors = target.surfaceColors
  const surfaceStrengths = target.surfaceStrengths
  const shellColors = target.shellColors
  const shellStrengths = target.shellStrengths
  const scatterColors = target.scatterColors
  const scatterStrengths = target.scatterStrengths

  for (let index = 0; index < surfaceStrengths.length; index += 1) {
    surfaceStrengths[index] =
      fromPayload.surfaceStrengths[index] * inverse + toPayload.surfaceStrengths[index] * mix
    shellStrengths[index] =
      fromPayload.shellStrengths[index] * inverse + toPayload.shellStrengths[index] * mix
    scatterStrengths[index] =
      fromPayload.scatterStrengths[index] * inverse + toPayload.scatterStrengths[index] * mix
  }

  for (let index = 0; index < surfaceColors.length; index += 1) {
    surfaceColors[index] =
      fromPayload.surfaceColors[index] * inverse + toPayload.surfaceColors[index] * mix
    shellColors[index] = fromPayload.shellColors[index] * inverse + toPayload.shellColors[index] * mix
    scatterColors[index] =
      fromPayload.scatterColors[index] * inverse + toPayload.scatterColors[index] * mix
  }

  return target
}

function buildMotionSeedArray(positions) {
  const seeds = new Float32Array(positions.length / 3)
  for (let index = 0; index < seeds.length; index += 1) {
    const offset = index * 3
    const x = positions[offset]
    const y = positions[offset + 1]
    const z = positions[offset + 2]
    const seed = Math.abs(x * 0.0731 + y * 0.117 + z * 0.0917)
    seeds[index] = seed - Math.floor(seed)
  }
  return seeds
}

function createAnimatedSignalPayload(template) {
  return {
    surfaceColors: new Float32Array(template.surfaceColors.length),
    surfaceStrengths: new Float32Array(template.surfaceStrengths.length),
    shellColors: new Float32Array(template.shellColors.length),
    shellStrengths: new Float32Array(template.shellStrengths.length),
    scatterColors: new Float32Array(template.scatterColors.length),
    scatterStrengths: new Float32Array(template.scatterStrengths.length),
    initialized: false,
  }
}

function resetAnimatedSignalPayload(animatedPayload) {
  if (!animatedPayload) {
    return
  }
  animatedPayload.surfaceColors.fill(0)
  animatedPayload.surfaceStrengths.fill(0)
  animatedPayload.shellColors.fill(0)
  animatedPayload.shellStrengths.fill(0)
  animatedPayload.scatterColors.fill(0)
  animatedPayload.scatterStrengths.fill(0)
  animatedPayload.initialized = false
}

function updateAnimatedSignalPayload(
  animatedPayload,
  targetPayload,
  motionSeeds,
  timeSeconds,
  deltaSeconds,
) {
  if (!animatedPayload || !targetPayload) {
    return targetPayload
  }

  if (!animatedPayload.initialized) {
    animatedPayload.surfaceColors.set(targetPayload.surfaceColors)
    animatedPayload.surfaceStrengths.set(targetPayload.surfaceStrengths)
    animatedPayload.shellColors.set(targetPayload.shellColors)
    animatedPayload.shellStrengths.set(targetPayload.shellStrengths)
    animatedPayload.scatterColors.set(targetPayload.scatterColors)
    animatedPayload.scatterStrengths.set(targetPayload.scatterStrengths)
    animatedPayload.initialized = true
  }

  const settle = 1 - Math.exp(-deltaSeconds * 5.6)
  const shellCarry = Math.exp(-deltaSeconds * 1.55)
  const scatterCarry = Math.exp(-deltaSeconds * 1.25)
  const twoPi = Math.PI * 2

  for (let index = 0; index < targetPayload.surfaceStrengths.length; index += 1) {
    const targetSurface = targetPayload.surfaceStrengths[index]
    const targetShell = targetPayload.shellStrengths[index]
    const targetScatter = targetPayload.scatterStrengths[index]
    const seed = motionSeeds[index] ?? 0
    const phase = seed * twoPi
    const activity = Math.max(targetSurface, targetShell, targetScatter)
    const speed = 0.45 + Math.pow(activity, 0.7) * 2.8
    const waveA = Math.sin(timeSeconds * (0.95 + speed * 1.3) + phase * 1.8)
    const waveB = Math.sin(timeSeconds * (2.2 + speed * 2.05) - phase * 3.6)
    const waveC = Math.cos(timeSeconds * (0.62 + speed * 0.82) + phase * 6.4)
    const buzz = waveA * 0.46 + waveB * 0.34 + waveC * 0.2
    const surge = Math.max(0, waveA * 0.58 + waveC * 0.42)

    const surfaceBase =
      animatedPayload.surfaceStrengths[index] +
      (targetSurface - animatedPayload.surfaceStrengths[index]) * settle
    const shellBase = Math.max(
      animatedPayload.shellStrengths[index] * shellCarry,
      animatedPayload.shellStrengths[index] +
        (targetShell - animatedPayload.shellStrengths[index]) * settle,
    )
    const scatterBase = Math.max(
      animatedPayload.scatterStrengths[index] * scatterCarry,
      animatedPayload.scatterStrengths[index] +
        (targetScatter - animatedPayload.scatterStrengths[index]) * settle,
    )

    animatedPayload.surfaceStrengths[index] = clamp(
      surfaceBase + buzz * (0.008 + targetSurface * 0.028) + surge * targetSurface * 0.018,
      0,
      1,
    )
    animatedPayload.shellStrengths[index] = clamp(
      shellBase + buzz * (0.014 + targetShell * 0.052) + surge * (0.01 + targetShell * 0.06),
      0,
      1,
    )
    animatedPayload.scatterStrengths[index] = clamp(
      scatterBase +
        buzz * (0.02 + targetScatter * 0.082) +
        surge * (0.012 + targetScatter * 0.095),
      0,
      1,
    )
  }

  for (let index = 0; index < targetPayload.surfaceColors.length; index += 3) {
    const vertexIndex = index / 3
    const activity = Math.max(
      targetPayload.surfaceStrengths[vertexIndex],
      targetPayload.shellStrengths[vertexIndex],
      targetPayload.scatterStrengths[vertexIndex],
    )
    const seed = motionSeeds[vertexIndex] ?? 0
    const phase = seed * twoPi
    const speed = 0.58 + Math.pow(activity, 0.72) * 2.6
    const glow =
      Math.max(0, Math.sin(timeSeconds * (1.2 + speed * 1.6) + phase * 3.2)) * 0.1 +
      Math.max(0, Math.cos(timeSeconds * (0.72 + speed * 0.92) - phase * 4.1)) * 0.06

    const surfaceBoost = 0.96 + glow + animatedPayload.surfaceStrengths[vertexIndex] * 0.05
    const shellBoost = 0.98 + glow * 1.18 + animatedPayload.shellStrengths[vertexIndex] * 0.08
    const scatterBoost = 1 + glow * 1.35 + animatedPayload.scatterStrengths[vertexIndex] * 0.11

    animatedPayload.surfaceColors[index] = clamp(
      targetPayload.surfaceColors[index] * surfaceBoost,
      0,
      1,
    )
    animatedPayload.surfaceColors[index + 1] = clamp(
      targetPayload.surfaceColors[index + 1] * surfaceBoost,
      0,
      1,
    )
    animatedPayload.surfaceColors[index + 2] = clamp(
      targetPayload.surfaceColors[index + 2] * surfaceBoost,
      0,
      1,
    )

    animatedPayload.shellColors[index] = clamp(targetPayload.shellColors[index] * shellBoost, 0, 1)
    animatedPayload.shellColors[index + 1] = clamp(
      targetPayload.shellColors[index + 1] * shellBoost,
      0,
      1,
    )
    animatedPayload.shellColors[index + 2] = clamp(
      targetPayload.shellColors[index + 2] * shellBoost,
      0,
      1,
    )

    animatedPayload.scatterColors[index] = clamp(
      targetPayload.scatterColors[index] * scatterBoost,
      0,
      1,
    )
    animatedPayload.scatterColors[index + 1] = clamp(
      targetPayload.scatterColors[index + 1] * scatterBoost,
      0,
      1,
    )
    animatedPayload.scatterColors[index + 2] = clamp(
      targetPayload.scatterColors[index + 2] * scatterBoost,
      0,
      1,
    )
  }

  return animatedPayload
}

function resolveFrameBlend(frames, selectedTimestep, playbackState) {
  if (!frames.length) {
    return { fromIndex: -1, toIndex: -1, blend: 0 }
  }

  const clampedIndex = clamp(selectedTimestep, 0, frames.length - 1)
  if (!playbackState?.isPlaying || frames.length === 1) {
    return { fromIndex: clampedIndex, toIndex: clampedIndex, blend: 0 }
  }

  let fromIndex = clampedIndex
  const currentFrame = frames[fromIndex] ?? frames[0]
  if (playbackState.time < Number(currentFrame?.start_s ?? 0) && fromIndex > 0) {
    fromIndex -= 1
  }

  const toIndex = Math.min(fromIndex + 1, frames.length - 1)
  if (toIndex === fromIndex) {
    return { fromIndex, toIndex, blend: 0 }
  }

  const fromFrame = frames[fromIndex]
  const toFrame = frames[toIndex]
  const fromStart = Number(fromFrame?.start_s ?? fromIndex)
  const toStart = Number(toFrame?.start_s ?? toIndex)
  const span = Math.max(toStart - fromStart, Number(fromFrame?.duration_s ?? 0), 0.001)
  const rawBlend = clamp((Number(playbackState.time ?? fromStart) - fromStart) / span, 0, 1)
  return {
    fromIndex,
    toIndex,
    blend: smoothstep(0, 1, rawBlend),
  }
}

function setSignalLayerAttributes(layer, colors, strengths) {
  if (!layer) {
    return
  }
  layer.color.array.set(colors)
  layer.strength.array.set(strengths)
  layer.color.needsUpdate = true
  layer.strength.needsUpdate = true
}

function clearSignalLayer(layer) {
  if (!layer) {
    return
  }
  layer.color.array.fill(0)
  layer.strength.array.fill(0)
  layer.color.needsUpdate = true
  layer.strength.needsUpdate = true
}

function resetTrailState(trailState) {
  if (!trailState) {
    return
  }
  trailState.colors.fill(0)
  trailState.strengths.fill(0)
}

function createTrailState(vertexCount) {
  return {
    colors: new Float32Array(vertexCount * 3),
    strengths: new Float32Array(vertexCount),
  }
}

function updateTrailState(trailState, sourceColors, sourceStrengths, deltaSeconds) {
  if (!trailState) {
    return
  }

  const decay = Math.exp(-deltaSeconds / 1.08)

  for (let index = 0; index < sourceStrengths.length; index += 1) {
    const nextStrength = sourceStrengths[index]
    const decayedStrength = trailState.strengths[index] * decay
    const colorOffset = index * 3

    if (nextStrength >= decayedStrength) {
      trailState.strengths[index] = nextStrength
      trailState.colors[colorOffset] = sourceColors[colorOffset]
      trailState.colors[colorOffset + 1] = sourceColors[colorOffset + 1]
      trailState.colors[colorOffset + 2] = sourceColors[colorOffset + 2]
    } else if (decayedStrength <= 0.0015) {
      trailState.strengths[index] = 0
      trailState.colors[colorOffset] = 0
      trailState.colors[colorOffset + 1] = 0
      trailState.colors[colorOffset + 2] = 0
    } else {
      trailState.strengths[index] = decayedStrength
    }
  }
}

function buildMeshEdgePairs(faces) {
  const seenEdges = new Set()
  const pairs = []
  const edges = [
    [0, 1],
    [1, 2],
    [2, 0],
  ]

  for (const face of faces) {
    for (const [leftIndex, rightIndex] of edges) {
      const leftVertex = face[leftIndex]
      const rightVertex = face[rightIndex]
      const edgeKey =
        leftVertex < rightVertex
          ? `${leftVertex}:${rightVertex}`
          : `${rightVertex}:${leftVertex}`
      if (seenEdges.has(edgeKey)) {
        continue
      }
      seenEdges.add(edgeKey)
      pairs.push(leftVertex, rightVertex)
    }
  }

  return new Uint32Array(pairs)
}

function buildFilamentLayer(positions, normals, edgePairs, offset) {
  const geometry = new THREE.BufferGeometry()
  const linePositions = new Float32Array(edgePairs.length * 3)
  const lineColors = new Float32Array(edgePairs.length * 3)
  const lineStrengths = new Float32Array(edgePairs.length)

  for (let index = 0; index < edgePairs.length; index += 1) {
    const vertex = edgePairs[index]
    const sourceOffset = vertex * 3
    const targetOffset = index * 3
    linePositions[targetOffset] = positions[sourceOffset] + normals[sourceOffset] * offset
    linePositions[targetOffset + 1] = positions[sourceOffset + 1] + normals[sourceOffset + 1] * offset
    linePositions[targetOffset + 2] = positions[sourceOffset + 2] + normals[sourceOffset + 2] * offset
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3))
  geometry.setAttribute('signalStrength', new THREE.Float32BufferAttribute(lineStrengths, 1))

  const material = buildFilamentMaterial({ opacity: 0.48 })
  const filaments = new THREE.LineSegments(geometry, material)
  filaments.renderOrder = 7

  return {
    lines: filaments,
    color: geometry.getAttribute('color'),
    strength: geometry.getAttribute('signalStrength'),
    edgePairs,
  }
}

function updateFilamentLayer(layer, sourceColors, sourceStrengths, timeSeconds) {
  if (!layer) {
    return
  }

  const { edgePairs } = layer
  const colorArray = layer.color.array
  const strengthArray = layer.strength.array

  for (let index = 0; index < edgePairs.length; index += 2) {
    const leftVertex = edgePairs[index]
    const rightVertex = edgePairs[index + 1]
    const leftStrength = sourceStrengths[leftVertex] ?? 0
    const rightStrength = sourceStrengths[rightVertex] ?? 0
    const coupled = Math.min(leftStrength, rightStrength)
    const average = (leftStrength + rightStrength) * 0.5
    const baseVisibleStrength =
      smoothstep(0.18, 0.78, coupled) * Math.pow(average, 0.78)
    const speed = 0.6 + Math.pow(baseVisibleStrength, 0.72) * 3.6
    const pulse =
      0.88 +
      0.12 *
        Math.sin(
          timeSeconds * (1.4 + speed * 1.85) +
            leftVertex * 0.013 +
            rightVertex * 0.021,
        )
    const visibleStrength = baseVisibleStrength * pulse

    const leftColorOffset = leftVertex * 3
    const rightColorOffset = rightVertex * 3
    const targetOffset = index * 3

    strengthArray[index] = visibleStrength
    strengthArray[index + 1] = visibleStrength

    const red =
      ((sourceColors[leftColorOffset] ?? 0) + (sourceColors[rightColorOffset] ?? 0)) * 0.5
    const green =
      ((sourceColors[leftColorOffset + 1] ?? 0) + (sourceColors[rightColorOffset + 1] ?? 0)) *
      0.5
    const blue =
      ((sourceColors[leftColorOffset + 2] ?? 0) + (sourceColors[rightColorOffset + 2] ?? 0)) *
      0.5
    const intensity = 0.22 + visibleStrength * 1.18

    for (let segmentIndex = 0; segmentIndex < 2; segmentIndex += 1) {
      const colorOffset = targetOffset + segmentIndex * 3
      colorArray[colorOffset] = clamp(red * intensity, 0, 1)
      colorArray[colorOffset + 1] = clamp(green * intensity, 0, 1)
      colorArray[colorOffset + 2] = clamp(blue * intensity, 0, 1)
    }
  }

  layer.color.needsUpdate = true
  layer.strength.needsUpdate = true
}

function buildBoundaryPayload({
  positions,
  normals,
  faces,
  regionIndices,
  colorResolver,
  offset,
}) {
  if (!regionIndices?.length) {
    return null
  }

  const seenEdges = new Set()
  const linePositions = []
  const lineColors = []
  const edges = [
    [0, 1],
    [1, 2],
    [2, 0],
  ]

  for (const face of faces) {
    for (const [leftIndex, rightIndex] of edges) {
      const leftVertex = face[leftIndex]
      const rightVertex = face[rightIndex]
      if (
        regionIndices[leftVertex] === undefined ||
        regionIndices[rightVertex] === undefined ||
        regionIndices[leftVertex] === regionIndices[rightVertex]
      ) {
        continue
      }

      const edgeKey =
        leftVertex < rightVertex
          ? `${leftVertex}:${rightVertex}`
          : `${rightVertex}:${leftVertex}`
      if (seenEdges.has(edgeKey)) {
        continue
      }
      seenEdges.add(edgeKey)

      for (const vertex of [leftVertex, rightVertex]) {
        const sourceIndex = vertex * 3
        linePositions.push(
          positions[sourceIndex] + normals[sourceIndex] * offset,
          positions[sourceIndex + 1] + normals[sourceIndex + 1] * offset,
          positions[sourceIndex + 2] + normals[sourceIndex + 2] * offset,
        )
        const [red, green, blue] = colorResolver(vertex)
        lineColors.push(red, green, blue)
      }
    }
  }

  return {
    positions: new Float32Array(linePositions),
    colors: new Float32Array(lineColors),
  }
}

function buildZoneColorTable(zoneKeys = []) {
  return zoneKeys.map((zoneKey) => {
    const color = new THREE.Color(getZoneColor(zoneKey))
    return [color.r, color.g, color.b]
  })
}

function createBoundaryLines(payload, materialOptions, renderOrder) {
  if (!payload || payload.positions.length === 0) {
    return null
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(payload.positions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(payload.colors, 3))
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    ...materialOptions,
  })
  const lines = new THREE.LineSegments(geometry, material)
  lines.renderOrder = renderOrder
  return lines
}

export function ThreeBrainViewer({
  mesh,
  frames = [],
  selectedTimestep = 0,
  playbackRef = null,
  overlayMode = 'hcp',
  className = '',
}) {
  const hostRef = useRef(null)
  const controlsRef = useRef(null)
  const cameraRef = useRef(null)
  const viewStateRef = useRef(null)
  const sceneLayerRefs = useRef(null)
  const signalParticlesRef = useRef(null)
  const signalScatterRef = useRef(null)
  const signalSurfaceRef = useRef(null)
  const signalShellRef = useRef(null)
  const signalFilamentsRef = useRef(null)
  const overlayRefs = useRef({ hcp: null, systems: null })
  const signalCacheRef = useRef(new Map())
  const framesRef = useRef(frames)
  const selectedTimestepRef = useRef(selectedTimestep)
  const blendedSignalRef = useRef(null)
  const animatedSignalRef = useRef(null)
  const particleTrailRef = useRef(null)
  const [renderMode, setRenderMode] = useState('hybrid')

  useEffect(() => {
    framesRef.current = frames
    selectedTimestepRef.current = selectedTimestep
  }, [frames, selectedTimestep])

  useEffect(() => {
    if (!hostRef.current || !mesh) {
      return undefined
    }

    const host = hostRef.current
    const scene = new THREE.Scene()
    const signalCache = signalCacheRef.current

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 0.8
    renderer.setClearColor(0x000000, 0)
    host.appendChild(renderer.domElement)

    const pmremGenerator = new THREE.PMREMGenerator(renderer)
    const roomEnvironment = new RoomEnvironment()
    const environmentTarget = pmremGenerator.fromScene(roomEnvironment, 0.022)
    scene.environment = environmentTarget.texture

    const ambientLight = new THREE.AmbientLight(0xe4eef6, 0.18)
    scene.add(ambientLight)

    const hemisphereLight = new THREE.HemisphereLight(0xb9d9ef, 0x071018, 0.52)
    scene.add(hemisphereLight)

    const keyLight = new THREE.DirectionalLight(0xf7fbff, 1.5)
    keyLight.position.set(-7.5, 6.8, 9.5)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0x7cb8ff, 0.26)
    fillLight.position.set(8.5, 1.6, 4.5)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0x82e8ff, 0.62)
    rimLight.position.set(2.2, 5.8, -10.5)
    scene.add(rimLight)

    const bounceLight = new THREE.DirectionalLight(0xffd5b3, 0.1)
    bounceLight.position.set(-3.5, -5.8, 3.2)
    scene.add(bounceLight)

    const positions = toDisplayCoordinates(mesh.coords)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setIndex(mesh.faces.flat())
    geometry.center()
    geometry.computeBoundingBox()
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()

    const centeredPositions = new Float32Array(geometry.getAttribute('position').array)
    const centeredNormals = new Float32Array(geometry.getAttribute('normal').array)
    const vertexCount = geometry.getAttribute('position').count
    const motionSeeds = buildMotionSeedArray(centeredPositions)
    const smoothingCache = buildSmoothingCache(mesh.faces, vertexCount)
    const bgValues = mesh.bg_b64 ? decodeNormalizedBytes(mesh.bg_b64) : null
    const anatomyColors = buildAnatomyColorArray(bgValues, vertexCount)
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(anatomyColors, 3))

    const radius = geometry.boundingSphere?.radius || 1
    const distance = radius * 4.7
    camera.near = Math.max(0.01, radius / 120)
    camera.far = radius * 80
    camera.position.set(-distance, radius * 0.56, distance * 0.2)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()

    const anatomyMaterial = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      roughness: 0.82,
      metalness: 0.02,
      envMapIntensity: 0.22,
      transparent: true,
      opacity: 0.24,
      transmission: 0.72,
      thickness: 0.16,
      ior: 1.08,
      attenuationDistance: 2.4,
      attenuationColor: new THREE.Color('#96b7d6'),
      clearcoat: 0.1,
      clearcoatRoughness: 0.66,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const anatomySurface = new THREE.Mesh(geometry, anatomyMaterial)
    anatomySurface.renderOrder = 1
    scene.add(anatomySurface)

    const rimGeometry = geometry.clone()
    const rimMaterial = buildRimMaterial()
    const rimShell = new THREE.Mesh(rimGeometry, rimMaterial)
    rimShell.scale.setScalar(1.012)
    rimShell.renderOrder = 5
    scene.add(rimShell)

    const basePointGeometry = new THREE.BufferGeometry()
    basePointGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(centeredPositions, 3),
    )
    basePointGeometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(buildConstellationColorArray(anatomyColors), 3),
    )
    const basePointMaterial = buildConstellationMaterial({
      opacity: 0.12,
      size: 2.1,
    })
    const basePointCloud = new THREE.Points(basePointGeometry, basePointMaterial)
    basePointCloud.renderOrder = 1
    scene.add(basePointCloud)

    const interiorPointPositions = new Float32Array(centeredPositions.length)
    const interiorInset = radius * 0.034
    for (let index = 0; index < centeredPositions.length; index += 3) {
      interiorPointPositions[index] =
        centeredPositions[index] - centeredNormals[index] * interiorInset
      interiorPointPositions[index + 1] =
        centeredPositions[index + 1] - centeredNormals[index + 1] * interiorInset
      interiorPointPositions[index + 2] =
        centeredPositions[index + 2] - centeredNormals[index + 2] * interiorInset
    }
    const interiorPointGeometry = new THREE.BufferGeometry()
    interiorPointGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(interiorPointPositions, 3),
    )
    interiorPointGeometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(buildInteriorConstellationColorArray(anatomyColors), 3),
    )
    const interiorPointMaterial = buildConstellationMaterial({
      opacity: 0.18,
      size: 2.6,
    })
    const interiorPointCloud = new THREE.Points(interiorPointGeometry, interiorPointMaterial)
    interiorPointCloud.renderOrder = 1
    scene.add(interiorPointCloud)

    const signalScatterGeometry = geometry.clone()
    const signalScatterColor = new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3)
    const signalScatterStrength = new THREE.Float32BufferAttribute(new Float32Array(vertexCount), 1)
    signalScatterGeometry.setAttribute('signalColor', signalScatterColor)
    signalScatterGeometry.setAttribute('signalStrength', signalScatterStrength)
    const signalScatterMaterial = buildSignalMaterial({
      opacity: 0.18,
      intensity: 0.62,
      fresnelStrength: 0.16,
      fresnelPower: 1.2,
      floorStrength: 0.0,
      motionAmount: 0.022,
      displacementScale: radius * 0.008,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    })
    const signalScatter = new THREE.Mesh(signalScatterGeometry, signalScatterMaterial)
    signalScatter.scale.setScalar(0.992)
    signalScatter.renderOrder = 2
    scene.add(signalScatter)
    signalScatterRef.current = {
      color: signalScatterColor,
      strength: signalScatterStrength,
    }

    const signalSurfaceGeometry = geometry.clone()
    const signalSurfaceColor = new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3)
    const signalSurfaceStrength = new THREE.Float32BufferAttribute(new Float32Array(vertexCount), 1)
    signalSurfaceGeometry.setAttribute('signalColor', signalSurfaceColor)
    signalSurfaceGeometry.setAttribute('signalStrength', signalSurfaceStrength)
    const signalSurfaceMaterial = buildSignalMaterial({
      opacity: 0.78,
      intensity: 0.96,
      fresnelStrength: 0.18,
      fresnelPower: 2.8,
      floorStrength: 0.06,
      motionAmount: 0.008,
      displacementScale: radius * 0.014,
      blending: THREE.NormalBlending,
      polygonOffset: true,
    })
    const signalSurface = new THREE.Mesh(signalSurfaceGeometry, signalSurfaceMaterial)
    signalSurface.renderOrder = 3
    scene.add(signalSurface)
    signalSurfaceRef.current = {
      color: signalSurfaceColor,
      strength: signalSurfaceStrength,
    }

    const signalShellGeometry = geometry.clone()
    const signalShellColor = new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3)
    const signalShellStrength = new THREE.Float32BufferAttribute(new Float32Array(vertexCount), 1)
    signalShellGeometry.setAttribute('signalColor', signalShellColor)
    signalShellGeometry.setAttribute('signalStrength', signalShellStrength)
    const signalShellMaterial = buildSignalMaterial({
      opacity: 0.34,
      intensity: 0.82,
      fresnelStrength: 0.82,
      fresnelPower: 1.8,
      floorStrength: 0.02,
      motionAmount: 0.05,
      displacementScale: radius * 0.024,
      blending: THREE.AdditiveBlending,
    })
    const signalShell = new THREE.Mesh(signalShellGeometry, signalShellMaterial)
    signalShell.scale.setScalar(1.008)
    signalShell.renderOrder = 4
    scene.add(signalShell)
    signalShellRef.current = {
      color: signalShellColor,
      strength: signalShellStrength,
    }

    const signalParticleGeometry = new THREE.BufferGeometry()
    signalParticleGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(centeredPositions, 3),
    )
    signalParticleGeometry.setAttribute(
      'normal',
      new THREE.Float32BufferAttribute(centeredNormals, 3),
    )
    const signalParticleColor = new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3)
    const signalParticleStrength = new THREE.Float32BufferAttribute(new Float32Array(vertexCount), 1)
    signalParticleGeometry.setAttribute('signalColor', signalParticleColor)
    signalParticleGeometry.setAttribute('signalStrength', signalParticleStrength)
    const signalParticleMaterial = buildParticleMaterial({
      opacity: 0.68,
      intensity: 1.2,
      size: 20,
      lift: radius * 0.026,
    })
    const signalParticles = new THREE.Points(signalParticleGeometry, signalParticleMaterial)
    signalParticles.renderOrder = 6
    scene.add(signalParticles)
    signalParticlesRef.current = {
      color: signalParticleColor,
      strength: signalParticleStrength,
    }
    particleTrailRef.current = createTrailState(vertexCount)

    const edgePairs = buildMeshEdgePairs(mesh.faces)
    const filamentLayer = buildFilamentLayer(
      centeredPositions,
      centeredNormals,
      edgePairs,
      radius * 0.017,
    )
    scene.add(filamentLayer.lines)
    signalFilamentsRef.current = filamentLayer

    const hcpBoundaryPayload = buildBoundaryPayload({
      positions: centeredPositions,
      normals: centeredNormals,
      faces: mesh.faces,
      regionIndices: mesh.parcel_indices,
      colorResolver: () => [0.91, 0.95, 0.99],
      offset: radius * 0.006,
    })
    const hcpLines = createBoundaryLines(
      hcpBoundaryPayload,
      {
        opacity: 0.18,
      },
      5,
    )
    if (hcpLines) {
      scene.add(hcpLines)
    }

    const zoneColorTable = buildZoneColorTable(mesh.zone_keys ?? [])
    const systemsBoundaryPayload = buildBoundaryPayload({
      positions: centeredPositions,
      normals: centeredNormals,
      faces: mesh.faces,
      regionIndices: mesh.zone_indices,
      colorResolver: (vertex) =>
        zoneColorTable[mesh.zone_indices?.[vertex] ?? zoneColorTable.length - 1] ?? [0.85, 0.87, 0.9],
      offset: radius * 0.0075,
    })
    const systemsLines = createBoundaryLines(
      systemsBoundaryPayload,
      {
        opacity: 0.36,
      },
      6,
    )
    if (systemsLines) {
      scene.add(systemsLines)
    }

    overlayRefs.current = {
      hcp: hcpLines,
      systems: systemsLines,
    }

    const shadowTexture = buildContactShadowTexture()
    const shadowPlane = shadowTexture
      ? new THREE.Mesh(
          new THREE.PlaneGeometry(radius * 3.5, radius * 3.05),
          new THREE.MeshBasicMaterial({
            map: shadowTexture,
            transparent: true,
            opacity: 0.34,
            depthWrite: false,
          }),
        )
      : null
    if (shadowPlane) {
      shadowPlane.rotation.x = -Math.PI / 2
      shadowPlane.position.y = (geometry.boundingBox?.min.y ?? -radius) - radius * 0.18
      shadowPlane.renderOrder = 0
      scene.add(shadowPlane)
    }

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.075
    controls.rotateSpeed = 0.5
    controls.enablePan = true
    controls.minDistance = radius * 1.8
    controls.maxDistance = radius * 10
    controls.target.set(0, 0, 0)
    controls.update()
    controlsRef.current = controls
    viewStateRef.current = {
      radius,
      distance,
      reset: new THREE.Vector3(-distance, radius * 0.56, distance * 0.2),
    }
    sceneLayerRefs.current = {
      anatomySurface,
      anatomyMaterial,
      rimShell,
      shadowPlane,
      basePointMaterial,
      interiorPointCloud,
      interiorPointMaterial,
      signalScatter,
      signalSurface,
      signalShell,
      signalParticles,
      filaments: filamentLayer.lines,
    }

    const resize = () => {
      const width = Math.max(host.clientWidth, 320)
      const height = Math.max(host.clientHeight, 420)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }

    const observer = new ResizeObserver(() => resize())
    observer.observe(host)
    resize()

    let active = true
    let previousFrameTime = performance.now() / 1000
    const renderLoop = () => {
      if (!active) {
        return
      }
      const nowSeconds = performance.now() / 1000
      const deltaSeconds = clamp(nowSeconds - previousFrameTime, 1 / 240, 0.2)
      previousFrameTime = nowSeconds
      basePointMaterial.uniforms.time.value = nowSeconds
      interiorPointMaterial.uniforms.time.value = nowSeconds
      signalParticleMaterial.uniforms.time.value = nowSeconds
      signalScatterMaterial.uniforms.time.value = nowSeconds
      signalSurfaceMaterial.uniforms.time.value = nowSeconds
      signalShellMaterial.uniforms.time.value = nowSeconds
      signalScatterMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.04 : 0.026
      signalSurfaceMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.03 : 0.018
      signalShellMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.082 : 0.05

      const activeFrames = framesRef.current ?? []
      if (
        activeFrames.length &&
        signalSurfaceRef.current &&
        signalShellRef.current &&
        signalParticlesRef.current &&
        signalFilamentsRef.current
      ) {
        const playbackState = playbackRef?.current ?? {
          time: activeFrames[selectedTimestepRef.current]?.start_s ?? 0,
          isPlaying: false,
        }
        const blendState = resolveFrameBlend(
          activeFrames,
          selectedTimestepRef.current,
          playbackState,
        )

        if (blendState.fromIndex >= 0) {
          const fromPayload = getFrameSignalPayload(
            activeFrames[blendState.fromIndex],
            signalCache,
            smoothingCache,
          )
          const toPayload = getFrameSignalPayload(
            activeFrames[blendState.toIndex],
            signalCache,
            smoothingCache,
          )

          if (fromPayload && toPayload) {
            let displayPayload = fromPayload
            if (
              blendState.fromIndex === blendState.toIndex ||
              blendState.blend <= 0.001
            ) {
            } else {
              if (
                !blendedSignalRef.current ||
                blendedSignalRef.current.surfaceColors.length !==
                  fromPayload.surfaceColors.length
              ) {
                blendedSignalRef.current = {
                  surfaceColors: new Float32Array(fromPayload.surfaceColors.length),
                  surfaceStrengths: new Float32Array(fromPayload.surfaceStrengths.length),
                  shellColors: new Float32Array(fromPayload.shellColors.length),
                  shellStrengths: new Float32Array(fromPayload.shellStrengths.length),
                  scatterColors: new Float32Array(fromPayload.scatterColors.length),
                  scatterStrengths: new Float32Array(fromPayload.scatterStrengths.length),
                }
              }
              const blendedPayload = blendSignalPayloads(
                fromPayload,
                toPayload,
                blendState.blend,
                blendedSignalRef.current,
              )
              displayPayload = blendedPayload
            }

            const particleTrail = particleTrailRef.current
            if (
              !animatedSignalRef.current ||
              animatedSignalRef.current.surfaceColors.length !== displayPayload.surfaceColors.length
            ) {
              animatedSignalRef.current = createAnimatedSignalPayload(displayPayload)
            }
            const animatedPayload = updateAnimatedSignalPayload(
              animatedSignalRef.current,
              displayPayload,
              motionSeeds,
              nowSeconds,
              deltaSeconds,
            )
            setSignalLayerAttributes(
              signalScatterRef.current,
              animatedPayload.scatterColors,
              animatedPayload.scatterStrengths,
            )
            setSignalLayerAttributes(
              signalSurfaceRef.current,
              animatedPayload.surfaceColors,
              animatedPayload.surfaceStrengths,
            )
            setSignalLayerAttributes(
              signalShellRef.current,
              animatedPayload.shellColors,
              animatedPayload.shellStrengths,
            )
            updateTrailState(
              particleTrail,
              animatedPayload.shellColors,
              animatedPayload.scatterStrengths,
              deltaSeconds,
            )
            if (particleTrail) {
              setSignalLayerAttributes(
                signalParticlesRef.current,
                particleTrail.colors,
                particleTrail.strengths,
              )
            }
            updateFilamentLayer(
              signalFilamentsRef.current,
              animatedPayload.shellColors,
              animatedPayload.shellStrengths,
              nowSeconds,
            )
          }
        }
      }
      controls.update()
      renderer.render(scene, camera)
      window.requestAnimationFrame(renderLoop)
    }
    renderLoop()

    return () => {
      active = false
      observer.disconnect()
      controls.dispose()
      geometry.dispose()
      rimGeometry.dispose()
      basePointGeometry.dispose()
      interiorPointGeometry.dispose()
      signalScatterGeometry.dispose()
      signalSurfaceGeometry.dispose()
      signalShellGeometry.dispose()
      signalParticleGeometry.dispose()
      anatomyMaterial.dispose()
      rimMaterial.dispose()
      basePointMaterial.dispose()
      interiorPointMaterial.dispose()
      signalParticleMaterial.dispose()
      signalScatterMaterial.dispose()
      signalSurfaceMaterial.dispose()
      signalShellMaterial.dispose()
      filamentLayer.lines.geometry.dispose()
      filamentLayer.lines.material.dispose()
      hcpLines?.geometry.dispose()
      hcpLines?.material.dispose()
      systemsLines?.geometry.dispose()
      systemsLines?.material.dispose()
      shadowPlane?.geometry.dispose()
      shadowPlane?.material.dispose()
      shadowTexture?.dispose()
      environmentTarget.dispose()
      pmremGenerator.dispose()
      if (typeof roomEnvironment.dispose === 'function') {
        roomEnvironment.dispose()
      }
      renderer.dispose()
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement)
      }
      controlsRef.current = null
      cameraRef.current = null
      viewStateRef.current = null
      signalScatterRef.current = null
      signalSurfaceRef.current = null
      signalShellRef.current = null
      signalParticlesRef.current = null
      signalFilamentsRef.current = null
      animatedSignalRef.current = null
      particleTrailRef.current = null
      sceneLayerRefs.current = null
      overlayRefs.current = { hcp: null, systems: null }
      signalCache.clear()
    }
  }, [mesh, playbackRef])

  useEffect(() => {
    if ((frames?.length ?? 0) > 0) {
      resetAnimatedSignalPayload(animatedSignalRef.current)
      resetTrailState(particleTrailRef.current)
      return
    }
    resetAnimatedSignalPayload(animatedSignalRef.current)
    clearSignalLayer(signalParticlesRef.current)
    clearSignalLayer(signalScatterRef.current)
    clearSignalLayer(signalSurfaceRef.current)
    clearSignalLayer(signalShellRef.current)
    clearSignalLayer(signalFilamentsRef.current)
    resetTrailState(particleTrailRef.current)
  }, [frames])

  useEffect(() => {
    const overlays = overlayRefs.current
    if (!overlays) {
      return
    }
    if (overlays.hcp) {
      overlays.hcp.visible = overlayMode === 'hcp'
    }
    if (overlays.systems) {
      overlays.systems.visible = overlayMode === 'systems'
    }
  }, [overlayMode, mesh])

  useEffect(() => {
    const layers = sceneLayerRefs.current
    if (!layers) {
      return
    }

    const plotted = renderMode === 'plotted'
    layers.anatomySurface.visible = !plotted
    layers.rimShell.visible = !plotted
    layers.signalScatter.visible = !plotted
    layers.signalSurface.visible = !plotted
    layers.signalShell.visible = true
    layers.signalParticles.visible = true
    layers.filaments.visible = true
    layers.interiorPointCloud.visible = !plotted

    if (layers.shadowPlane) {
      layers.shadowPlane.visible = !plotted
    }

    layers.basePointMaterial.uniforms.opacity.value = plotted ? 0.46 : 0.19
    layers.basePointMaterial.uniforms.size.value = plotted ? 3.2 : 2.1
    layers.interiorPointMaterial.uniforms.opacity.value = plotted ? 0 : 0.22
    layers.interiorPointMaterial.uniforms.size.value = plotted ? 2.6 : 3.1

    layers.anatomyMaterial.opacity = plotted ? 0.24 : 0.18
    layers.anatomyMaterial.transmission = plotted ? 0.72 : 0.84
    layers.anatomyMaterial.thickness = plotted ? 0.16 : 0.1
    layers.anatomyMaterial.envMapIntensity = plotted ? 0.22 : 0.28
    layers.anatomyMaterial.attenuationDistance = plotted ? 2.4 : 3.2
  }, [renderMode])

  function applyViewPreset(preset) {
    const camera = cameraRef.current
    const controls = controlsRef.current
    const viewState = viewStateRef.current
    if (!camera || !controls || !viewState) {
      return
    }

    const { distance, radius, reset } = viewState
    const nextPosition = reset.clone()

    if (preset === 'left') {
      nextPosition.set(-distance, radius * 0.42, distance * 0.18)
    } else if (preset === 'right') {
      nextPosition.set(distance, radius * 0.42, -distance * 0.18)
    } else if (preset === 'dorsal') {
      nextPosition.set(distance * 0.08, distance * 1.02, distance * 0.05)
    }

    camera.position.copy(nextPosition)
    controls.target.set(0, 0, 0)
    controls.update()
  }

  return (
    <div className={`brain-viewer ${className}`.trim()}>
      <div className="brain-viewer__toolbar">
        <button
          type="button"
          className="button brain-viewer__button"
          onClick={() => applyViewPreset('left')}
        >
          Left
        </button>
        <button
          type="button"
          className="button brain-viewer__button"
          onClick={() => applyViewPreset('right')}
        >
          Right
        </button>
        <button
          type="button"
          className="button brain-viewer__button"
          onClick={() => applyViewPreset('dorsal')}
        >
          Dorsal
        </button>
        <button
          type="button"
          className="button brain-viewer__button"
          onClick={() => applyViewPreset('reset')}
        >
          Reset
        </button>
        <button
          type="button"
          className={`button brain-viewer__button ${
            renderMode === 'hybrid' ? 'brain-viewer__button--active' : ''
          }`.trim()}
          onClick={() => setRenderMode('hybrid')}
        >
          Hybrid
        </button>
        <button
          type="button"
          className={`button brain-viewer__button ${
            renderMode === 'plotted' ? 'brain-viewer__button--active' : ''
          }`.trim()}
          onClick={() => setRenderMode('plotted')}
        >
          Plotted
        </button>
      </div>
      <div className="brain-viewer__hud">
        <span className="brain-viewer__hud-pill">
          {renderMode === 'plotted' ? 'Vertex field' : 'Relief surface'}
        </span>
        <span className="brain-viewer__hud-pill">Particle halo</span>
        <span className="brain-viewer__hud-pill">Hot-edge filaments</span>
      </div>
      <div ref={hostRef} className="brain-viewer__canvas" />
    </div>
  )
}
