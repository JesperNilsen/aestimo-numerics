# aestimo-numerics

Zero-dependency numerical toolkit for finance in JavaScript — six single-file
UMD modules that run identically under bare Node (`require`) and in the
browser (`<script>` tag → `globalThis.Aestimo<Module>`). No build step, no
dependencies.

The headline module is **`aestimo-boot.js`: a BCa bootstrap in JavaScript**
(bias-corrected and accelerated confidence intervals, with iid, moving-block,
and stationary resampling) — something the JS ecosystem has essentially
lacked. The rest of the toolkit is what it takes to trust one: an audited
seeded PRNG, safeguarded root-finding, SPD linear algebra, and the finance
applications (YTM, IRR/XIRR, market beta, Kelly sizing).

Extracted from [Aestimo](https://github.com/JesperNilsen/Aestimo), a suite of
offline investment-research instruments, where these files are the
computational layer.

## Modules

| File | Browser global | Contents |
|---|---|---|
| `aestimo-random.js` | `AestimoRandom` | Seeded PRNG — splitmix32 seed diffusion → xoshiro128\*\* streams (`makeRNG`: `next`, `nextU32`, `clone`); Box–Muller Gaussian sampling; `erf`/`erfc`/`invNormCDF`; correlated normal draws; sample moments. Core streams validated bit-for-bit against an independent C reference. |
| `aestimo-boot.js` | `AestimoBoot` | Bootstrap inference: `bootstrap`/`bootstrapCI` (normal, basic, percentile, **BCa** intervals), `blockBootstrap` + `stationaryBootstrap` for dependent data, jackknife acceleration, quantiles pinned to R type 7. Ships statistic helpers (`STATS`: mean, std, sharpe, skewness, maxDrawdown). Depends on `aestimo-random.js`. |
| `aestimo-solve.js` | `AestimoSolve` | Safeguarded scalar root-finding (`bisect`/`newton`/`secant`/`solve`); bond pricing and `ytm`; `npv`/`irr` with sign-change counting and `irrAll` for multiple-IRR detection; dated-flow `xnpv`/`xirr`. Solvers return structured results with `converged` and `residual` — failure is legible, never a plausible wrong number. |
| `aestimo-chol.js` | `AestimoChol` | SPD Cholesky, forward/back substitution, `solveSPD`/`invSPD` (no explicit inverses), covariance↔correlation conversion, LDLᵀ as the semi-definite degradation path. Non-PD input returns `{ ok: false, reason, failedAt }` — Cholesky succeeding *is* the SPD test. |
| `aestimo-beta.js` | `AestimoBeta` | Market-model beta: OLS `regress`/`beta`, `rollingBeta`, Blume and Vasicek shrinkage toward 1 (including peer-group Vasicek). Standalone. |
| `aestimo-kelly.js` | `AestimoKelly` | Kelly position sizing: `kellyBinary`/`kellyDiscrete`/`kellySample` (exact empirical)/`kellyGaussian` (with the fat-tail overbet caveat), `fractionalKelly`, `drawdownProb`, growth diagnostics, and the multi-asset `kellyVector` Σ⁻¹(μ−rf·𝟙) solved as an SPD system. Depends on `aestimo-solve.js` + `aestimo-chol.js`. |

Dependency edges — everything else is standalone:

```
random ──▶ boot        solve ──┬──▶ kelly
                       chol  ──┘
```

## Use

Node — plain `require`; consume by copying the files (not yet published to
npm):

```js
const Boot = require('./aestimo-boot.js');

const returns = [0.021, -0.013, 0.034, /* … */];
const ci = Boot.bootstrapCI(returns, Boot.STATS.sharpe, { B: 5000, seed: 42 });
ci.estimate;   // 0.3918
ci.bca;        // { lo: -0.0983, hi: 0.9403, ... }  — the 95% BCa interval
```

```js
const Solve = require('./aestimo-solve.js');

Solve.ytm({ price: 95, face: 100, coupon: 5, periods: 10 });
// { periodicYield: 0.0566871..., annualYield: 0.0566871...,
//   annualization: 'bey', converged: true, residual: 5.7e-14 }

const Kelly = require('./aestimo-kelly.js');
Kelly.kellyBinary({ p: 0.55, b: 1 });   // 0.1 — the classic edge/odds result
```

Browser — load a module's dependencies first; each file registers its global:

```html
<script src="aestimo-random.js"></script>
<script src="aestimo-boot.js"></script>
<script>
  const { bootstrapCI, STATS } = globalThis.AestimoBoot;
</script>
```

## Testing

```bash
npm test
```

(equivalently `node run-all.js`). Every module ships its canonical invariant
battery with golden regression locks; the `*.test.js` runners are thin
printers over it, and `run-all.js` aggregates them plus two cross-module
consistency locks: the inline Cholesky that `aestimo-random.js` uses for
correlated draws must agree with `aestimo-chol.js` on both the success path
and the rejection path — duplication without the possibility of drift.

The golden policy, inherited from Aestimo: goldens are locked computed values
used as regression tripwires, never modified to make a test pass; the
strongest are external cross-checks (the PRNG core is validated against an
independent C reference, stream for stream). Tolerances are scale-relative —
`|a−b| ≤ tol·(1 + max(|a|,|b|))` — never bare absolute floors, except where
exact binary64 representability justifies exact equality.

## Provenance and status

The six modules and their test batteries are extracted **byte-identical**
from [Aestimo](https://github.com/JesperNilsen/Aestimo)'s `engine/` layer;
the only adapted file is `run-all.js`, trimmed to the modules shipped here.
For the time being Aestimo remains upstream — changes land there first and
are mirrored here. The engine modules kept behind (`aestimo-dcf.js`,
`aestimo-portfolio.js`) belong to the Aestimo product rather than to the
toolkit.

## License

MIT © 2026 Jesper Nilsen.
