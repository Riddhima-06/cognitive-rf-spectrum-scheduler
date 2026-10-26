import { SUB_BANDS } from "./rfSynthesizer";

export type SchedulerMode = "LINEAR" | "UCB1" | "THOMPSON";

export interface RewardWeights {
  w1Detection: number; // Default 0.65
  w2Uncertainty: number; // Default 0.35
  w3DwellPenalty: number; // Default 0.15
  dwellNormalized: number; // Default 0.25 (normalized tau_dwell)
  explorationC: number; // Default 0.85 for UCB1
  discountGamma: number; // Default 0.96 for non-stationary tracking
}

export interface ArmStatistics {
  armIndex: number;
  label: string;
  freqRangeMHz: string;
  pulls: number;
  discountedPulls: number;
  hits: number;
  empiricalMeanReward: number;
  ucbBonus: number;
  ucbScore: number;
  alpha: number; // Thompson Beta(alpha, beta)
  beta: number;
  thompsonSample: number;
  lastReward: number;
  lastUncertainty: number;
  empiricalHitRate: number;
}

export interface SchedulerComparisonMetrics {
  totalSteps: number;
  linear: {
    probabilityOfIntercept: number; // %
    meanInterceptTimeMs: number; // ms
    bandwidthEfficiency: number; // %
    cumulativeReward: number;
  };
  cognitive: {
    probabilityOfIntercept: number; // %
    meanInterceptTimeMs: number; // ms
    bandwidthEfficiency: number; // %
    cumulativeReward: number;
  };
  lastRewardBreakdown: {
    detectionTerm: number;
    uncertaintyTerm: number;
    dwellPenaltyTerm: number;
    totalReward: number;
  };
}

/**
 * Samples a Gamma(shape, 1) random variable using Marsaglia and Tsang's method
 * Used to draw exact Beta(alpha, beta) samples for Thompson Sampling.
 */
function sampleGamma(shape: number): number {
  if (shape < 1) {
    const u = Math.max(1e-8, Math.random());
    return sampleGamma(1 + shape) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = 0;
    let v = 0;
    do {
      const u1 = Math.max(1e-8, Math.random());
      const u2 = Math.max(1e-8, Math.random());
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.max(1e-8, Math.random());
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export function sampleBeta(alpha: number, beta: number): number {
  const x = sampleGamma(Math.max(0.1, alpha));
  const y = sampleGamma(Math.max(0.1, beta));
  return x / Math.max(1e-9, x + y);
}

export function createInitialArmStats(): ArmStatistics[] {
  return SUB_BANDS.map((b) => ({
    armIndex: b.index,
    label: b.label,
    freqRangeMHz: `${b.startMHz}-${b.endMHz}`,
    pulls: 1,
    discountedPulls: 1.0,
    hits: 0,
    empiricalMeanReward: 0.35,
    ucbBonus: 0.45,
    ucbScore: 0.8,
    alpha: 1.5,
    beta: 1.5,
    thompsonSample: 0.5,
    lastReward: 0,
    lastUncertainty: 0.5,
    empiricalHitRate: 0,
  }));
}

export function createInitialComparisonMetrics(): SchedulerComparisonMetrics {
  return {
    totalSteps: 0,
    linear: {
      probabilityOfIntercept: 38.5,
      meanInterceptTimeMs: 142.0,
      bandwidthEfficiency: 36.2,
      cumulativeReward: 0,
    },
    cognitive: {
      probabilityOfIntercept: 78.4,
      meanInterceptTimeMs: 48.5,
      bandwidthEfficiency: 81.6,
      cumulativeReward: 0,
    },
    lastRewardBreakdown: {
      detectionTerm: 0,
      uncertaintyTerm: 0,
      dwellPenaltyTerm: 0,
      totalReward: 0,
    },
  };
}

/**
 * Evaluates the explicit Cognitive Bandit Reward Function:
 * R = w1 * Detection + w2 * ModulationUncertainty - w3 * tau_dwell
 */
export function computeBanditReward(
  detected: boolean,
  modulationUncertainty: number,
  weights: RewardWeights
): {
  detectionTerm: number;
  uncertaintyTerm: number;
  dwellPenaltyTerm: number;
  totalReward: number;
  normalizedReward01: number;
} {
  const detectionVal = detected ? 1.0 : 0.0;
  const effectiveUncertainty = detected ? modulationUncertainty : 0.08;

  const detectionTerm = Number((weights.w1Detection * detectionVal).toFixed(3));
  const uncertaintyTerm = Number((weights.w2Uncertainty * effectiveUncertainty).toFixed(3));
  const dwellPenaltyTerm = Number((weights.w3DwellPenalty * weights.dwellNormalized).toFixed(3));

  const rawReward = detectionTerm + uncertaintyTerm - dwellPenaltyTerm;
  const totalReward = Number(rawReward.toFixed(3));

  // Map reward into [0, 1] for Beta posterior updates
  const maxPossible = Math.max(0.2, weights.w1Detection + weights.w2Uncertainty);
  const normalizedReward01 = Math.max(
    0.02,
    Math.min(0.98, (rawReward + weights.w3DwellPenalty) / (maxPossible + weights.w3DwellPenalty))
  );

  return {
    detectionTerm,
    uncertaintyTerm,
    dwellPenaltyTerm,
    totalReward,
    normalizedReward01,
  };
}

/**
 * Selects the next sub-band arm (0..7) based on the active Scheduler Mode
 */
export function selectNextSubBandArm(
  mode: SchedulerMode,
  currentArm: number,
  arms: ArmStatistics[],
  weights: RewardWeights
): { nextArm: number; updatedArmsWithScores: ArmStatistics[] } {
  const totalDiscountedPulls = arms.reduce((acc, a) => acc + a.discountedPulls, 0);

  const updated = arms.map((arm) => {
    const ucbBonus =
      weights.explorationC *
      Math.sqrt(Math.log(Math.max(2, totalDiscountedPulls)) / Math.max(0.5, arm.discountedPulls));
    const ucbScore = arm.empiricalMeanReward + ucbBonus;
    const thompsonSample = sampleBeta(arm.alpha, arm.beta);

    return {
      ...arm,
      ucbBonus: Number(ucbBonus.toFixed(3)),
      ucbScore: Number(ucbScore.toFixed(3)),
      thompsonSample: Number(thompsonSample.toFixed(3)),
    };
  });

  if (mode === "LINEAR") {
    return {
      nextArm: (currentArm + 1) % SUB_BANDS.length,
      updatedArmsWithScores: updated,
    };
  }

  if (mode === "UCB1") {
    let bestArm = 0;
    let bestScore = -Infinity;
    for (const arm of updated) {
      // Tie-break with small random perturbation
      const score = arm.ucbScore + Math.random() * 0.015;
      if (score > bestScore) {
        bestScore = score;
        bestArm = arm.armIndex;
      }
    }
    return { nextArm: bestArm, updatedArmsWithScores: updated };
  }

  // THOMPSON SAMPLING
  let bestArm = 0;
  let bestSample = -Infinity;
  for (const arm of updated) {
    // Combine Thompson posterior sample with uncertainty bonus
    const score = arm.thompsonSample + 0.12 * arm.lastUncertainty;
    if (score > bestSample) {
      bestSample = score;
      bestArm = arm.armIndex;
    }
  }
  return { nextArm: bestArm, updatedArmsWithScores: updated };
}

/**
 * Updates Non-Stationary MAB statistics after dwelling on `chosenArm`
 */
export function updateBanditAfterDwell(
  arms: ArmStatistics[],
  chosenArm: number,
  detected: boolean,
  modulationUncertainty: number,
  weights: RewardWeights
): {
  updatedArms: ArmStatistics[];
  rewardBreakdown: {
    detectionTerm: number;
    uncertaintyTerm: number;
    dwellPenaltyTerm: number;
    totalReward: number;
  };
} {
  const gamma = weights.discountGamma;
  const rewardInfo = computeBanditReward(detected, modulationUncertainty, weights);

  const updatedArms = arms.map((arm) => {
    // Apply non-stationary discounting across all arms so agile/hopping emitters are tracked
    const discPulls = arm.discountedPulls * gamma + (arm.armIndex === chosenArm ? 1 : 0);
    const discAlpha = Math.max(1.0, (arm.alpha - 1) * gamma + 1);
    const discBeta = Math.max(1.0, (arm.beta - 1) * gamma + 1);

    if (arm.armIndex !== chosenArm) {
      return {
        ...arm,
        discountedPulls: Number(discPulls.toFixed(2)),
        alpha: Number(discAlpha.toFixed(2)),
        beta: Number(discBeta.toFixed(2)),
      };
    }

    const newPulls = arm.pulls + 1;
    const newHits = arm.hits + (detected ? 1 : 0);
    const learningRate = 1 / Math.min(18, Math.max(2, discPulls));
    const newMeanReward =
      (1 - learningRate) * arm.empiricalMeanReward + learningRate * rewardInfo.totalReward;

    return {
      ...arm,
      pulls: newPulls,
      discountedPulls: Number(discPulls.toFixed(2)),
      hits: newHits,
      empiricalMeanReward: Number(newMeanReward.toFixed(3)),
      alpha: Number((discAlpha + rewardInfo.normalizedReward01 * 1.4).toFixed(2)),
      beta: Number((discBeta + (1 - rewardInfo.normalizedReward01) * 1.4).toFixed(2)),
      lastReward: rewardInfo.totalReward,
      lastUncertainty: modulationUncertainty,
      empiricalHitRate: Number(((newHits / newPulls) * 100).toFixed(1)),
    };
  });

  return {
    updatedArms,
    rewardBreakdown: {
      detectionTerm: rewardInfo.detectionTerm,
      uncertaintyTerm: rewardInfo.uncertaintyTerm,
      dwellPenaltyTerm: rewardInfo.dwellPenaltyTerm,
      totalReward: rewardInfo.totalReward,
    },
  };
}
