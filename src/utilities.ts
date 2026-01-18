/**
 * Matrix and math utilities for Gaussian Process computations
 */

/**
 * Create n×m matrix filled with value
 */
export function full(n: number, m: number, val: number): number[][] {
    return Array(n).fill(null).map(() => Array(m).fill(val));
}

/**
 * Create n×n identity matrix
 */
export function eye(n: number): number[][] {
    return full(n, n, 0).map((row, i) => {
        row[i] = 1;
        return row;
    });
}

/**
 * Create evenly spaced array from start to end with n points
 */
export function linspace(start: number, end: number, n: number): number[] {
    const step = (end - start) / (n - 1);
    return Array(n).fill(null).map((_, i) => start + i * step);
}

/**
 * Transpose a matrix
 */
export function transpose(A: number[][]): number[][] {
    return A[0].map((_, j) => A.map(row => row[j]));
}

/**
 * Matrix-vector multiplication
 */
export function matVec(A: number[][], x: number[]): number[] {
    return A.map(row => row.reduce((s, a, i) => s + a * x[i], 0));
}

/**
 * Add scalar to diagonal of matrix
 */
export function addDiag(A: number[][], c: number): number[][] {
    return A.map((row, i) => row.map((v, j) => (i === j ? v + c : v)));
}

/**
 * Cholesky decomposition - returns lower triangular matrix L such that A = L * L^T
 */
export function cholesky(A: number[][]): number[][] {
    const n = A.length;
    const L = full(n, n, 0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }
            L[i][j] = i === j
                ? Math.sqrt(Math.max(A[i][i] - sum, 1e-10))
                : (A[i][j] - sum) / L[j][j];
        }
    }
    return L;
}

/**
 * Forward substitution: solve L * x = b where L is lower triangular
 */
export function solveL(L: number[][], b: number[]): number[] {
    const n = L.length;
    const x = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        let sum = b[i];
        for (let j = 0; j < i; j++) {
            sum -= L[i][j] * x[j];
        }
        x[i] = sum / L[i][i];
    }
    return x;
}

/**
 * Backward substitution: solve L^T * x = b where L is lower triangular
 */
export function solveLT(L: number[][], b: number[]): number[] {
    const n = L.length;
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = b[i];
        for (let j = i + 1; j < n; j++) {
            sum -= L[j][i] * x[j];
        }
        x[i] = sum / L[i][i];
    }
    return x;
}

/**
 * Solve A * x = b using Cholesky decomposition (A must be positive definite)
 */
export function choleskySolve(L: number[][], b: number[]): number[] {
    return solveLT(L, solveL(L, b));
}

/**
 * Box-Muller transform to generate standard normal random variable
 */
export function randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
