/**
 * Kernel functions for Gaussian Process
 */

/**
 * RBF (Radial Basis Function) / Squared Exponential kernel
 * k(x, x') = σ² * exp(-||x - x'||² / (2ℓ²))
 *
 * @param x1 - First input
 * @param x2 - Second input
 * @param lengthScale - Length scale parameter (ℓ)
 * @param signalVariance - Signal variance (σ²)
 * @returns Kernel value
 */
export function rbf(x1: number, x2: number, lengthScale: number, signalVariance: number): number {
    const d = x1 - x2;
    const l2 = lengthScale ** 2;
    return signalVariance * Math.exp(-0.5 * d * d / l2);
}

/**
 * Matérn 1/2 kernel (equivalent to Ornstein-Uhlenbeck / Exponential kernel)
 * k(x, x') = σ² * exp(-|x - x'| / ℓ)
 *
 * This is the roughest of the Matérn family, producing sample paths that are
 * continuous but not differentiable.
 *
 * @param x1 - First input
 * @param x2 - Second input
 * @param lengthScale - Length scale parameter (ℓ)
 * @param signalVariance - Signal variance (σ²)
 * @returns Kernel value
 */
export function matern12(x1: number, x2: number, lengthScale: number, signalVariance: number): number {
    const r = Math.abs(x1 - x2) / lengthScale;
    return signalVariance * Math.exp(-r);
}

/**
 * Matérn 3/2 kernel
 * k(x, x') = σ² * (1 + √3 * r) * exp(-√3 * r)
 * where r = |x - x'| / ℓ
 *
 * Produces sample paths that are once differentiable.
 *
 * @param x1 - First input
 * @param x2 - Second input
 * @param lengthScale - Length scale parameter (ℓ)
 * @param signalVariance - Signal variance (σ²)
 * @returns Kernel value
 */
export function matern32(x1: number, x2: number, lengthScale: number, signalVariance: number): number {
    const r = Math.abs(x1 - x2) / lengthScale;
    const sqrt3r = Math.sqrt(3) * r;
    return signalVariance * (1 + sqrt3r) * Math.exp(-sqrt3r);
}

/**
 * Matérn 5/2 kernel
 * k(x, x') = σ² * (1 + √5 * r + 5r²/3) * exp(-√5 * r)
 * where r = |x - x'| / ℓ
 *
 * Produces sample paths that are twice differentiable.
 *
 * @param x1 - First input
 * @param x2 - Second input
 * @param lengthScale - Length scale parameter (ℓ)
 * @param signalVariance - Signal variance (σ²)
 * @returns Kernel value
 */
export function matern52(x1: number, x2: number, lengthScale: number, signalVariance: number): number {
    const r = Math.abs(x1 - x2) / lengthScale;
    const sqrt5r = Math.sqrt(5) * r;
    return signalVariance * (1 + sqrt5r + (5 * r * r) / 3) * Math.exp(-sqrt5r);
}

/**
 * Kernel function type
 */
export type KernelFn = (x1: number, x2: number, lengthScale: number, signalVariance: number) => number;

/**
 * Kernel function lookup
 */
export const KERNEL_FNS: Record<string, KernelFn> = {
    rbf,
    matern12,
    matern32,
    matern52,
};

/**
 * Available kernels registry with metadata
 */
export const KERNELS = {
    rbf: {
        name: 'RBF (Squared Exponential)',
        fn: rbf,
        description: 'Infinitely differentiable, very smooth'
    },
    matern52: {
        name: 'Matérn 5/2',
        fn: matern52,
        description: 'Twice differentiable, fairly smooth'
    },
    matern32: {
        name: 'Matérn 3/2',
        fn: matern32,
        description: 'Once differentiable, moderately smooth'
    },
    matern12: {
        name: 'Matérn 1/2 (Exponential)',
        fn: matern12,
        description: 'Continuous but rough, not differentiable'
    }
};
