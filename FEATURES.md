# GPzoo.js Features

GPzoo.js strives for API compatibility with the GPzoo Python library. This document tracks the implementation status of Gaussian Process variants and features.

In the tables below, we use a color legend to refer to features in GPzoo:

- 🟢 = supported
- 🟡 = supported, with API limitations
- 🟠 = not supported, easy to add (<1 day)
- 🔴 = not supported
- ⚪️ = not applicable, will not be supported (see notes)

## Gaussian Process Models

| Model  | Support | Notes  |
| ------ | ------- | ------ |
| SVGP   | 🔴      | Sparse Variational Gaussian Process |
| WSVGP  | 🔴      | Warped Sparse Variational Gaussian Process |
| LCGP   | 🔴      | Linear Coregionalization Gaussian Process |
| MGGPs  | 🔴      | Multi-Output Gaussian Processes |

## Kernels

| Kernel | Support | Notes  |
| ------ | ------- | ------ |
| RBF    | 🔴      | Radial Basis Function kernel |
| Matern | 🔴      | Matern class of kernels |
| Linear | 🔴      | Linear kernel |
| White  | 🔴      | White noise kernel |

## Utilities

| Function | Support | Notes  |
| -------- | ------- | ------ |
| Optimization | 🔴      | Hyperparameter optimization |
| Predictions  | 🔴      | Mean and variance predictions |
| Log-likelihood | 🔴      | Model likelihood computation |

## Implementation Notes

- Current implementation is a basic JavaScript GP placeholder for integration with other applications
- Not yet ported from GPzoo Python library - this is interim working code
- The library will use TypeScript for type safety and Bun for building
- Multiple output formats (ESM, CJS, UMD) will be provided for different environments
