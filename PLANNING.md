# GPzoo.js - Planning

## Project Structure

```
GPzoo.js/
├── src/
│   ├── index.ts          # Main entry point, exports all modules
│   ├── gp.ts             # Gaussian Process implementation
│   ├── kernels.ts        # Kernel functions
│   └── utilities.ts      # Utility functions
├── dist/                 # Build output (generated)
│   ├── index.mjs         # ESM format
│   ├── index.cjs         # CommonJS format
│   └── index.js          # UMD format for browsers
├── test/                 # Test files
│   ├── gp.test.ts
│   ├── kernels.test.ts
│   └── utilities.test.ts
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
└── PLANNING.md           # This file
```

## Tools & Technologies

- **Runtime/Build**: Bun
- **Language**: TypeScript
- **Testing**: Bun test
- **Linting**: Biome (Bun's preferred linter)

## Build Scripts

```json
{
  "scripts": {
    "build": "bun build ./src/index.ts --outfile ./dist/index.mjs && bun build ./src/index.ts --outfile ./dist/index.cjs --format cjs && bun build ./src/index.ts --outfile ./dist/index.js --format iife --global gpzoo",
    "test": "bun test",
    "lint": "bunx @biomejs/biome check src/"
  }
}
```

This generates three distributable formats:
- `dist/index.mjs` - ESM module for modern bundlers
- `dist/index.cjs` - CommonJS module for Node.js
- `dist/index.js` - IIFE for direct browser use

## Porting Checklist

- [ ] Initialize package.json
- [ ] Create TypeScript config
- [ ] Port gp.py → gp.ts
- [ ] Port kernels.py → kernels.ts
- [ ] Port utilities.py → utilities.ts
- [ ] Set up build scripts
- [ ] Add tests
- [ ] Configure linting
