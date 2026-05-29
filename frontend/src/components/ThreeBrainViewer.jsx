import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const MAX_HOTSPOTS = 1800

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function decodeBase64ToUint8(base64Value) {
  const binary = window.atob(base64Value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

function decodeActivationBytes(base64Value) {
  const bytes = decodeBase64ToUint8(base64Value)
  const activations = new Float32Array(bytes.length)
  for (let index = 0; index < bytes.length; index += 1) {
    activations[index] = bytes[index] / 255
  }
  return activations
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

function buildHotspotPayload(basePositions, baseNormals, colorBytes, activationValues, radius) {
  const threshold = percentile(activationValues, 0.965)
  const ceiling = Math.max(percentile(activationValues, 0.9985), threshold + 1e-5)
  const candidates = []

  for (let index = 0; index < activationValues.length; index += 1) {
    const activation = activationValues[index]
    const visible = clamp((activation - threshold) / (ceiling - threshold), 0, 1)
    if (visible <= 0) {
      continue
    }
    candidates.push({
      index,
      shaped: Math.pow(visible, 0.46),
    })
  }

  candidates.sort((left, right) => right.shaped - left.shaped)
  const count = Math.min(candidates.length, MAX_HOTSPOTS)
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)

  const cool = new THREE.Color('#00d7ff')
  const mid = new THREE.Color('#58f5ca')
  const warm = new THREE.Color('#ffd65a')
  const hot = new THREE.Color('#ff7f4d')
  const peak = new THREE.Color('#fff8ef')
  const mixed = new THREE.Color()

  for (let pointIndex = 0; pointIndex < count; pointIndex += 1) {
    const { index, shaped } = candidates[pointIndex]
    const sourceIndex = index * 3
    const targetIndex = pointIndex * 3
    const offset = radius * (0.005 + shaped * 0.014)

    positions[targetIndex] = basePositions[sourceIndex] + baseNormals[sourceIndex] * offset
    positions[targetIndex + 1] =
      basePositions[sourceIndex + 1] + baseNormals[sourceIndex + 1] * offset
    positions[targetIndex + 2] =
      basePositions[sourceIndex + 2] + baseNormals[sourceIndex + 2] * offset

    const red = colorBytes[sourceIndex] / 255
    const green = colorBytes[sourceIndex + 1] / 255
    const blue = colorBytes[sourceIndex + 2] / 255

    if (shaped < 0.42) {
      mixed.copy(cool).lerp(mid, shaped / 0.42)
    } else if (shaped < 0.76) {
      mixed.copy(mid).lerp(warm, (shaped - 0.42) / 0.34)
    } else if (shaped < 0.93) {
      mixed.copy(warm).lerp(hot, (shaped - 0.76) / 0.17)
    } else {
      mixed.copy(hot).lerp(peak, (shaped - 0.93) / 0.07)
    }

    const energy = 0.72 + shaped * 1.35
    colors[targetIndex] = clamp((mixed.r * 0.8 + red * 0.35) * energy, 0, 1)
    colors[targetIndex + 1] = clamp((mixed.g * 0.82 + green * 0.28) * energy, 0, 1)
    colors[targetIndex + 2] = clamp((mixed.b * 0.78 + blue * 0.18) * energy, 0, 1)
  }

  return { positions, colors, count }
}

export function ThreeBrainViewer({ mesh, frame, className = '' }) {
  const hostRef = useRef(null)
  const controlsRef = useRef(null)
  const cameraRef = useRef(null)
  const viewStateRef = useRef(null)
  const hotspotGeometryRef = useRef(null)
  const hotspotCacheRef = useRef(new Map())
  const basePositionsRef = useRef(null)
  const baseNormalsRef = useRef(null)
  const radiusRef = useRef(1)

  useEffect(() => {
    if (!hostRef.current || !mesh) {
      return undefined
    }

    const host = hostRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x05090e)
    const hotspotCache = hotspotCacheRef.current

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.02
    host.appendChild(renderer.domElement)

    const ambientLight = new THREE.AmbientLight(0xd8f2ff, 0.7)
    scene.add(ambientLight)

    const hemisphereLight = new THREE.HemisphereLight(0x8aeaff, 0x020508, 1.15)
    scene.add(hemisphereLight)

    const keyLight = new THREE.DirectionalLight(0xb5f5ff, 2.45)
    keyLight.position.set(-7, 4, 8)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0x2b7fff, 1.15)
    fillLight.position.set(6, -3, 4)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0x92ffe5, 1.45)
    rimLight.position.set(0, 5, -8)
    scene.add(rimLight)

    const positions = toDisplayCoordinates(mesh.coords)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setIndex(mesh.faces.flat())
    geometry.center()
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()

    const centeredPositions = new Float32Array(geometry.getAttribute('position').array)
    const centeredNormals = new Float32Array(geometry.getAttribute('normal').array)
    basePositionsRef.current = centeredPositions
    baseNormalsRef.current = centeredNormals

    const radius = geometry.boundingSphere?.radius || 1
    radiusRef.current = radius
    const distance = radius * 4.6
    camera.near = Math.max(0.01, radius / 120)
    camera.far = radius * 80
    camera.position.set(-distance, radius * 0.62, distance * 0.24)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()

    const anatomyMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x1b2731,
      roughness: 0.72,
      metalness: 0.04,
      clearcoat: 0.12,
      clearcoatRoughness: 0.72,
      side: THREE.DoubleSide,
    })
    const anatomySurface = new THREE.Mesh(geometry, anatomyMaterial)
    anatomySurface.renderOrder = 1
    scene.add(anatomySurface)

    const shellGeometry = geometry.clone()
    const glowShell = new THREE.Mesh(
      shellGeometry,
      new THREE.MeshBasicMaterial({
        color: 0x5dd8ff,
        transparent: true,
        opacity: 0.06,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    )
    glowShell.scale.setScalar(1.02)
    glowShell.renderOrder = 0
    scene.add(glowShell)

    const hotspotGeometry = new THREE.BufferGeometry()
    hotspotGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(new Float32Array(MAX_HOTSPOTS * 3), 3),
    )
    hotspotGeometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(new Float32Array(MAX_HOTSPOTS * 3), 3),
    )
    hotspotGeometry.setDrawRange(0, 0)
    hotspotGeometryRef.current = hotspotGeometry

    const hotspotGlowMaterial = new THREE.PointsMaterial({
      size: 16,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity: 0.26,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    hotspotGlowMaterial.toneMapped = false
    const hotspotGlow = new THREE.Points(hotspotGeometry, hotspotGlowMaterial)
    hotspotGlow.renderOrder = 4
    scene.add(hotspotGlow)

    const hotspotCoreMaterial = new THREE.PointsMaterial({
      size: 6,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity: 0.98,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    hotspotCoreMaterial.toneMapped = false
    const hotspotCore = new THREE.Points(hotspotGeometry, hotspotCoreMaterial)
    hotspotCore.renderOrder = 5
    scene.add(hotspotCore)

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
      reset: new THREE.Vector3(-distance, radius * 0.62, distance * 0.24),
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
      shellGeometry.dispose()
      hotspotGeometry.dispose()
      anatomyMaterial.dispose()
      glowShell.material.dispose()
      hotspotGlowMaterial.dispose()
      hotspotCoreMaterial.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement)
      }
      controlsRef.current = null
      cameraRef.current = null
      viewStateRef.current = null
      hotspotGeometryRef.current = null
      basePositionsRef.current = null
      baseNormalsRef.current = null
      hotspotCache.clear()
    }
  }, [mesh])

  useEffect(() => {
    if (
      !frame?.colors_b64 ||
      !frame?.activation_b64 ||
      !hotspotGeometryRef.current ||
      !basePositionsRef.current ||
      !baseNormalsRef.current
    ) {
      return
    }

    const cacheKey = `${frame.colors_b64}:${frame.activation_b64}`
    let payload = hotspotCacheRef.current.get(cacheKey)
    if (!payload) {
      payload = buildHotspotPayload(
        basePositionsRef.current,
        baseNormalsRef.current,
        decodeBase64ToUint8(frame.colors_b64),
        decodeActivationBytes(frame.activation_b64),
        radiusRef.current,
      )
      hotspotCacheRef.current.set(cacheKey, payload)
    }

    const geometry = hotspotGeometryRef.current
    const positionAttribute = geometry.getAttribute('position')
    const colorAttribute = geometry.getAttribute('color')
    positionAttribute.array.fill(0)
    colorAttribute.array.fill(0)
    positionAttribute.array.set(payload.positions)
    colorAttribute.array.set(payload.colors)
    positionAttribute.needsUpdate = true
    colorAttribute.needsUpdate = true
    geometry.setDrawRange(0, payload.count)
  }, [frame])

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
      nextPosition.set(-distance, radius * 0.44, distance * 0.18)
    } else if (preset === 'right') {
      nextPosition.set(distance, radius * 0.44, -distance * 0.18)
    } else if (preset === 'dorsal') {
      nextPosition.set(distance * 0.08, distance * 1.02, distance * 0.06)
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
