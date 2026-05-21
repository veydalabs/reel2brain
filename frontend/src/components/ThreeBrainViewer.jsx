import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

function decodeBase64ToUint8(base64Value) {
  const binary = window.atob(base64Value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

function colorsToFloatArray(bytes) {
  const colors = new Float32Array(bytes.length)
  for (let index = 0; index < bytes.length; index += 1) {
    colors[index] = bytes[index] / 255
  }
  return colors
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

export function ThreeBrainViewer({ mesh, frame, className = '' }) {
  const hostRef = useRef(null)
  const rendererRef = useRef(null)
  const geometryRef = useRef(null)
  const controlsRef = useRef(null)
  const frameCacheRef = useRef(new Map())

  useEffect(() => {
    if (!hostRef.current || !mesh) {
      return undefined
    }

    const host = hostRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x04080d)
    const frameCache = frameCacheRef.current

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    rendererRef.current = renderer
    host.appendChild(renderer.domElement)

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.25)
    scene.add(ambientLight)

    const keyLight = new THREE.DirectionalLight(0xb4f7ff, 2.1)
    keyLight.position.set(-8, 2, 7)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0x3ba2ff, 0.7)
    fillLight.position.set(5, -4, 3)
    scene.add(fillLight)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(toDisplayCoordinates(mesh.coords), 3),
    )
    geometry.setIndex(mesh.faces.flat())
    geometry.center()
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()
    geometryRef.current = geometry

    const radius = geometry.boundingSphere?.radius || 1
    const distance = radius * 4.25
    camera.near = Math.max(0.01, radius / 120)
    camera.far = radius * 80
    camera.position.set(-distance, radius * 1.15, distance * 0.42)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.94,
      metalness: 0.02,
      side: THREE.DoubleSide,
    })
    const surface = new THREE.Mesh(geometry, material)
    scene.add(surface)

    const halo = new THREE.Mesh(
      geometry.clone(),
      new THREE.MeshBasicMaterial({
        color: 0x4de7ff,
        wireframe: true,
        transparent: true,
        opacity: 0.025,
      }),
    )
    halo.scale.setScalar(1.005)
    scene.add(halo)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.rotateSpeed = 0.55
    controls.enablePan = true
    controls.minDistance = radius * 1.35
    controls.maxDistance = radius * 10
    controls.target.set(0, 0, 0)
    controls.update()
    controlsRef.current = controls

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
      material.dispose()
      halo.geometry.dispose()
      halo.material.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement)
      }
      rendererRef.current = null
      geometryRef.current = null
      controlsRef.current = null
      frameCache.clear()
    }
  }, [mesh])

  useEffect(() => {
    if (!geometryRef.current || !frame?.colors_b64) {
      return
    }

    let colors = frameCacheRef.current.get(frame.colors_b64)
    if (!colors) {
      colors = colorsToFloatArray(decodeBase64ToUint8(frame.colors_b64))
      frameCacheRef.current.set(frame.colors_b64, colors)
    }

    const geometry = geometryRef.current
    const attribute = new THREE.Float32BufferAttribute(colors, 3)
    geometry.setAttribute('color', attribute)
    geometry.attributes.color.needsUpdate = true
  }, [frame])

  return <div ref={hostRef} className={className} />
}
