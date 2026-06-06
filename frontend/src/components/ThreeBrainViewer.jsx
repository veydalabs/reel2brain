import { useEffect, useRef } from 'react'
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
        float motionWave =
          sin(position.x * 4.8 + time * 1.6) * 0.28 +
          sin(position.y * 5.6 - time * 1.3) * 0.22 +
          sin(position.z * 4.2 + time * 1.1) * 0.16;
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
        float shimmerWave =
          sin(vLocalPosition.x * 5.6 + time * 2.2) * 0.5 +
          sin(vLocalPosition.y * 7.4 - time * 1.8) * 0.32 +
          sin(vLocalPosition.z * 4.1 + time * 1.25) * 0.18;
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
        float pulse =
          0.88 +
          0.12 * sin(
            time * 2.6 +
            position.x * 3.4 +
            position.y * 4.1 +
            position.z * 2.7
          );
        vec3 displacedPosition = position + normal * lift * (0.55 + strength * 1.9);
        vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);

        vSignalColor = signalColor;
        vSignalStrength = strength;

        gl_PointSize =
          max(0.0, (1.8 + pow(strength, 0.82) * size * pulse) * (260.0 / max(1.0, -mvPosition.z)));
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

function updateFilamentLayer(layer, sourceColors, sourceStrengths) {
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
    const visibleStrength = smoothstep(0.18, 0.78, coupled) * Math.pow(average, 0.78)

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
  const particleTrailRef = useRef(null)

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
    const smoothingCache = buildSmoothingCache(mesh.faces, vertexCount)
    const bgValues = mesh.bg_b64 ? decodeNormalizedBytes(mesh.bg_b64) : null
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(buildAnatomyColorArray(bgValues, vertexCount), 3),
    )

    const radius = geometry.boundingSphere?.radius || 1
    const distance = radius * 4.7
    camera.near = Math.max(0.01, radius / 120)
    camera.far = radius * 80
    camera.position.set(-distance, radius * 0.56, distance * 0.2)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()

    const anatomyMaterial = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.02,
      envMapIntensity: 0.12,
      transparent: true,
      opacity: 0.9,
      transmission: 0.18,
      thickness: 0.42,
      ior: 1.12,
      attenuationDistance: 0.92,
      attenuationColor: new THREE.Color('#8ea7c4'),
      clearcoat: 0.03,
      clearcoatRoughness: 0.8,
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
      signalParticleMaterial.uniforms.time.value = nowSeconds
      signalScatterMaterial.uniforms.time.value = nowSeconds
      signalSurfaceMaterial.uniforms.time.value = nowSeconds
      signalShellMaterial.uniforms.time.value = nowSeconds
      signalScatterMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.024 : 0.012
      signalSurfaceMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.014 : 0.008
      signalShellMaterial.uniforms.motionAmount.value =
        playbackRef?.current?.isPlaying ? 0.052 : 0.028

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
              setSignalLayerAttributes(
                signalScatterRef.current,
                fromPayload.scatterColors,
                fromPayload.scatterStrengths,
              )
              setSignalLayerAttributes(
                signalSurfaceRef.current,
                fromPayload.surfaceColors,
                fromPayload.surfaceStrengths,
              )
              setSignalLayerAttributes(
                signalShellRef.current,
                fromPayload.shellColors,
                fromPayload.shellStrengths,
              )
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
              setSignalLayerAttributes(
                signalScatterRef.current,
                blendedPayload.scatterColors,
                blendedPayload.scatterStrengths,
              )
              setSignalLayerAttributes(
                signalSurfaceRef.current,
                blendedPayload.surfaceColors,
                blendedPayload.surfaceStrengths,
              )
              setSignalLayerAttributes(
                signalShellRef.current,
                blendedPayload.shellColors,
                blendedPayload.shellStrengths,
              )
            }

            const particleTrail = particleTrailRef.current
            updateTrailState(
              particleTrail,
              displayPayload.shellColors,
              displayPayload.scatterStrengths,
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
              displayPayload.shellColors,
              displayPayload.shellStrengths,
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
      signalScatterGeometry.dispose()
      signalSurfaceGeometry.dispose()
      signalShellGeometry.dispose()
      signalParticleGeometry.dispose()
      anatomyMaterial.dispose()
      rimMaterial.dispose()
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
      particleTrailRef.current = null
      overlayRefs.current = { hcp: null, systems: null }
      signalCache.clear()
    }
  }, [mesh, playbackRef])

  useEffect(() => {
    if ((frames?.length ?? 0) > 0) {
      resetTrailState(particleTrailRef.current)
      return
    }
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
      </div>
      <div className="brain-viewer__hud">
        <span className="brain-viewer__hud-pill">Relief surface</span>
        <span className="brain-viewer__hud-pill">Particle halo</span>
        <span className="brain-viewer__hud-pill">Hot-edge filaments</span>
      </div>
      <div ref={hostRef} className="brain-viewer__canvas" />
    </div>
  )
}
