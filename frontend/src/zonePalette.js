export const ZONE_STYLE_BY_KEY = {
  visual_occipital: {
    label: 'Occipital visual',
    color: '#59b8ff',
  },
  auditory_temporal: {
    label: 'Auditory / temporal',
    color: '#31d3c6',
  },
  tpj_social: {
    label: 'TPJ / social association',
    color: '#f6bf52',
  },
  dorsal_attention_parietal: {
    label: 'Dorsal parietal / attention',
    color: '#89d956',
  },
  frontoparietal_control: {
    label: 'Frontal / control',
    color: '#a78bfa',
  },
  medial_value_cingulate: {
    label: 'Medial / cingulate / value',
    color: '#fb7185',
  },
  sensorimotor_opercular: {
    label: 'Sensorimotor / opercular',
    color: '#fb923c',
  },
  association_other: {
    label: 'Diffuse association',
    color: '#d7dee8',
  },
}

const DEFAULT_ZONE_KEY = 'association_other'

export function getZoneStyle(zoneKey) {
  return ZONE_STYLE_BY_KEY[zoneKey] ?? ZONE_STYLE_BY_KEY[DEFAULT_ZONE_KEY]
}

export function getZoneColor(zoneKey) {
  return getZoneStyle(zoneKey).color
}

export function getZoneLabel(zoneKey, fallbackLabel = '') {
  return fallbackLabel || getZoneStyle(zoneKey).label
}
