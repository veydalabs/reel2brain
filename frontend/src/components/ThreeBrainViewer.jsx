import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { getZoneColor } from '../zonePalette'

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
      rimOpacity: { value: 0.14 },
      rimPower: { value: 3.4 },
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
  blending,
  polygonOffset = false,
}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: opacity },
      intensity: { value: intensity },
      fresnelStrength: { value: fresnelStrength },
      fresnelPower: { value: fresnelPower },
      floorStrength: { value: floorStrength },
    },
    vertexShader: `
      attribute vec3 signalColor;
      attribute float signalStrength;

      varying vec3 vSignalColor;
      varying float vSignalStrength;
      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vSignalColor = signalColor;
        vSignalStrength = signalStrength;
        vNormal = normalize(normalMatrix * normal);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float opacity;
      uniform float intensity;
      uniform float fresnelStrength;
      uniform float fresnelPower;
      uniform float floorStrength;

      varying vec3 vSignalColor;
      varying float vSignalStrength;
      varying vec3 vNormal;
      varying vec3 vViewPosition;

      void main() {
        float strength = clamp(vSignalStrength, 0.0, 1.0);
        if (strength <= 0.001) {
          discard;
        }

        vec3 viewDirection = normalize(vViewPosition);
        float facing = max(dot(normalize(vNormal), viewDirection), 0.0);
        float fresnel = pow(1.0 - facing, fresnelPower);
        float body = smoothstep(floorStrength, 1.0, strength);
        float alpha = (body * 0.78 + fresnel * fresnelStrength * strength) * opacity;
        vec3 color = vSignalColor * (0.48 + strength * intensity + fresnel * fresnelStrength * 0.24);

        if (alpha < 0.0025) {
          discard;
        }

        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
    blending,
    toneMapped: false,
    polygonOffset,
    polygonOffsetFactor: polygonOffset ? -2 : 0,
    polygonOffsetUnits: polygonOffset ? -4 : 0,
  })
}

function buildZoneColorTable(zoneKeys = []) {
  return zoneKeys.map((zoneKey) => {
    const color = new THREE.Color(getZoneColor(zoneKey))
    return [color.r, color.g, color.b]
  })
}

function buildAnatomyColorArray(bgValues, vertexCount) {
  const colors = new Float32Array(vertexCount * 3)
  const shadow = new THREE.Color('#121a22')
  const midtone = new THREE.Color('#26323c')
  const highlight = new THREE.Color('#455463')

  for (let index = 0; index < vertexCount; index += 1) {
    const tone = bgValues?.[index] ?? 0.5
    const shaped = smoothstep(0.06, 0.94, tone)
    const midMix = Math.min(shaped * 1.22, 1)
    const hiMix = Math.pow(shaped, 2.1) * 0.58
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

function buildSignalPayload({ colorBytes, activationValues, zoneIndices, zoneColorTable }) {
  const floor = percentile(activationValues, 0.74)
  const ceiling = Math.max(percentile(activationValues, 0.992), floor + 1e-5)
  const surfaceColors = new Float32Array(activationValues.length * 3)
  const surfaceStrengths = new Float32Array(activationValues.length)
  const shellColors = new Float32Array(activationValues.length * 3)
  const shellStrengths = new Float32Array(activationValues.length)

  for (let index = 0; index < activationValues.length; index += 1) {
    const activation = activationValues[index]
    const shaped = Math.pow(smoothstep(floor, ceiling, activation), 1.12)
    const surfaceStrength = Math.pow(shaped, 1.45)
    const shellStrength = Math.pow(shaped, 0.84)

    surfaceStrengths[index] = surfaceStrength
    shellStrengths[index] = shellStrength

    if (shellStrength <= 0.0005) {
      continue
    }

    const sourceIndex = index * 3
    const heatRed = colorBytes[sourceIndex] / 255
    const heatGreen = colorBytes[sourceIndex + 1] / 255
    const heatBlue = colorBytes[sourceIndex + 2] / 255

    const zoneColor =
      zoneColorTable[zoneIndices?.[index] ?? zoneColorTable.length - 1] ??
      zoneColorTable[zoneColorTable.length - 1] ??
      [0.85, 0.87, 0.9]

    const zoneWeight = 0.88 - shaped * 0.24
    const heatWeight = 1 - zoneWeight

    const mixedRed = zoneColor[0] * zoneWeight + heatRed * heatWeight
    const mixedGreen = zoneColor[1] * zoneWeight + heatGreen * heatWeight
    const mixedBlue = zoneColor[2] * zoneWeight + heatBlue * heatWeight

    const surfaceEnergy = 0.14 + shaped * 0.86
    const shellEnergy = 0.08 + shellStrength * 0.74

    surfaceColors[sourceIndex] = clamp(mixedRed * surfaceEnergy, 0, 0.94)
    surfaceColors[sourceIndex + 1] = clamp(mixedGreen * surfaceEnergy, 0, 0.94)
    surfaceColors[sourceIndex + 2] = clamp(mixedBlue * surfaceEnergy, 0, 0.94)

    shellColors[sourceIndex] = clamp((zoneColor[0] * 0.92 + heatRed * 0.08) * shellEnergy, 0, 0.86)
    shellColors[sourceIndex + 1] = clamp((zoneColor[1] * 0.92 + heatGreen * 0.08) * shellEnergy, 0, 0.86)
    shellColors[sourceIndex + 2] = clamp((zoneColor[2] * 0.92 + heatBlue * 0.08) * shellEnergy, 0, 0.86)
  }

  return {
    surfaceColors,
    surfaceStrengths,
    shellColors,
    shellStrengths,
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

export function ThreeBrainViewer({ mesh, frame, className = '' }) {
  const hostRef = useRef(null)
  const controlsRef = useRef(null)
  const cameraRef = useRef(null)
  const viewStateRef = useRef(null)
  const signalSurfaceRef = useRef(null)
  const signalShellRef = useRef(null)
  const signalCacheRef = useRef(new Map())

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
    renderer.toneMappingExposure = 0.84
    renderer.setClearColor(0x000000, 0)
    host.appendChild(renderer.domElement)

    const pmremGenerator = new THREE.PMREMGenerator(renderer)
    const roomEnvironment = new RoomEnvironment()
    const environmentTarget = pmremGenerator.fromScene(roomEnvironment, 0.022)
    scene.environment = environmentTarget.texture

    const ambientLight = new THREE.AmbientLight(0xe4eef6, 0.22)
    scene.add(ambientLight)

    const hemisphereLight = new THREE.HemisphereLight(0xb9d9ef, 0x071018, 0.58)
    scene.add(hemisphereLight)

    const keyLight = new THREE.DirectionalLight(0xf7fbff, 1.65)
    keyLight.position.set(-7.5, 6.8, 9.5)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0x7cb8ff, 0.32)
    fillLight.position.set(8.5, 1.6, 4.5)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0x82e8ff, 0.72)
    rimLight.position.set(2.2, 5.8, -10.5)
    scene.add(rimLight)

    const bounceLight = new THREE.DirectionalLight(0xffd5b3, 0.12)
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

    const vertexCount = geometry.getAttribute('position').count
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

    const anatomyMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.9,
      metalness: 0.03,
      envMapIntensity: 0.18,
      side: THREE.DoubleSide,
    })
    const anatomySurface = new THREE.Mesh(geometry, anatomyMaterial)
    anatomySurface.renderOrder = 1
    scene.add(anatomySurface)

    const rimGeometry = geometry.clone()
    const rimMaterial = buildRimMaterial()
    const rimShell = new THREE.Mesh(rimGeometry, rimMaterial)
    rimShell.scale.setScalar(1.014)
    rimShell.renderOrder = 2
    scene.add(rimShell)

    const signalSurfaceGeometry = geometry.clone()
    const signalSurfaceColor = new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3)
    const signalSurfaceStrength = new THREE.Float32BufferAttribute(new Float32Array(vertexCount), 1)
    signalSurfaceGeometry.setAttribute('signalColor', signalSurfaceColor)
    signalSurfaceGeometry.setAttribute('signalStrength', signalSurfaceStrength)
    const signalSurfaceMaterial = buildSignalMaterial({
      opacity: 0.68,
      intensity: 0.9,
      fresnelStrength: 0.22,
      fresnelPower: 2.8,
      floorStrength: 0.06,
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
      opacity: 0.44,
      intensity: 0.82,
      fresnelStrength: 0.92,
      fresnelPower: 1.95,
      floorStrength: 0.02,
      blending: THREE.AdditiveBlending,
    })
    const signalShell = new THREE.Mesh(signalShellGeometry, signalShellMaterial)
    signalShell.scale.setScalar(1.01)
    signalShell.renderOrder = 4
    scene.add(signalShell)
    signalShellRef.current = {
      color: signalShellColor,
      strength: signalShellStrength,
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
    const renderLoop = () => {
      if (!active) {
        return
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
      signalSurfaceGeometry.dispose()
      signalShellGeometry.dispose()
      anatomyMaterial.dispose()
      rimMaterial.dispose()
      signalSurfaceMaterial.dispose()
      signalShellMaterial.dispose()
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
      signalSurfaceRef.current = null
      signalShellRef.current = null
      signalCache.clear()
    }
  }, [mesh])

  useEffect(() => {
    if (!signalSurfaceRef.current || !signalShellRef.current || !mesh) {
      return
    }

    if (!frame?.colors_b64 || !frame?.activation_b64) {
      clearSignalLayer(signalSurfaceRef.current)
      clearSignalLayer(signalShellRef.current)
      return
    }

    const cacheKey = `${frame.colors_b64}:${frame.activation_b64}`
    let payload = signalCacheRef.current.get(cacheKey)
    if (!payload) {
      payload = buildSignalPayload({
        colorBytes: decodeBase64ToUint8(frame.colors_b64),
        activationValues: decodeNormalizedBytes(frame.activation_b64),
        zoneIndices: mesh.zone_indices ?? [],
        zoneColorTable: buildZoneColorTable(mesh.zone_keys ?? []),
      })
      signalCacheRef.current.set(cacheKey, payload)
    }

    setSignalLayerAttributes(
      signalSurfaceRef.current,
      payload.surfaceColors,
      payload.surfaceStrengths,
    )
    setSignalLayerAttributes(
      signalShellRef.current,
      payload.shellColors,
      payload.shellStrengths,
    )
  }, [frame, mesh])

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
      <div ref={hostRef} className="brain-viewer__canvas" />
    </div>
  )
}
