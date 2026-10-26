export type ColorMapName = "TURBO" | "VIRIDIS" | "JET";

/**
 * Interpolates RGB control points into a 256-entry [R, G, B] lookup table
 */
function buildLut(stops: [number, number, number, number][]): [number, number, number][] {
  const lut: [number, number, number][] = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s0 = stops[0];
    let s1 = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k][0] && t <= stops[k + 1][0]) {
        s0 = stops[k];
        s1 = stops[k + 1];
        break;
      }
    }
    const span = Math.max(1e-6, s1[0] - s0[0]);
    const localT = (t - s0[0]) / span;
    const r = Math.round(s0[1] + (s1[1] - s0[1]) * localT);
    const g = Math.round(s0[2] + (s1[2] - s0[2]) * localT);
    const b = Math.round(s0[3] + (s1[3] - s0[3]) * localT);
    lut[i] = [
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b)),
    ];
  }
  return lut;
}

const VIRIDIS_LUT = buildLut([
  [0.0, 15, 12, 42],
  [0.2, 59, 82, 139],
  [0.45, 33, 145, 140],
  [0.7, 94, 201, 98],
  [1.0, 253, 231, 37],
]);

const TURBO_LUT = buildLut([
  [0.0, 18, 18, 48],
  [0.18, 48, 118, 235],
  [0.38, 28, 212, 185],
  [0.58, 162, 245, 62],
  [0.78, 251, 162, 28],
  [1.0, 225, 34, 24],
]);

const JET_LUT = buildLut([
  [0.0, 8, 12, 75],
  [0.22, 0, 75, 255],
  [0.45, 0, 235, 255],
  [0.68, 255, 235, 0],
  [0.88, 255, 60, 0],
  [1.0, 165, 0, 20],
]);

const LUT_MAP: Record<ColorMapName, [number, number, number][]> = {
  TURBO: TURBO_LUT,
  VIRIDIS: VIRIDIS_LUT,
  JET: JET_LUT,
};

export function mapNormalizedToRgb(
  val01: number,
  colorMap: ColorMapName
): [number, number, number] {
  const idx = Math.max(0, Math.min(255, Math.floor(val01 * 255)));
  return LUT_MAP[colorMap][idx];
}
