/* ===========================================================================
 * aestimo-boot.test.js  —  Invariant battery for the bootstrap engine
 * ===========================================================================
 *  Run:  node aestimo-boot.test.js   (requires aestimo-random.js present)
 *  Prints PASS/FAIL per invariant; exits non-zero on any failure.
 *
 *  Every check is DETERMINISTIC: all randomness flows through the seeded RNG,
 *  so the statistical invariants are exact given a seed and never flaky. Each
 *  tolerance is stated with its rationale (systematic finite-n term + Monte-
 *  Carlo margin ~1/√B). Test data are generated in-process from the seeded RNG
 *  — no Math.random, no external fixtures.
 * ------------------------------------------------------------------------- */
'use strict';
const A = require('./aestimo-boot.js');
// MIGRATION (2026-07-06): the real aestimo-random.js returns a generator OBJECT
// with .next()->[0,1); the test was written for the sfc32 stand-in's bare
// function. Bridge to the bare `()=>[0,1)` contract — the test-side twin of the
// seam adapted in aestimo-boot.js. (No-op if a bare function is supplied.)
const _Random = require('./aestimo-random.js');
const makeRNG = (seed) => { const g = _Random.makeRNG(seed); return typeof g === 'function' ? g : () => g.next(); };
const {
  resample, quantile, mean, std, sharpe, skewness, maxDrawdown,
  bootstrap, ciNormal, ciBasic, ciPercentile, ciBCa,
  biasCorrection, acceleration, bcaAlphas,
  blockBootstrap, stationaryBootstrap, movingBlockResample, normInv, normCDF
} = A;

// --- assert harness ---------------------------------------------------------
let pass = 0, fail = 0; const failed = [];
function ok(name, cond) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; failed.push(name); console.log('  FAIL  ' + name); }
}
const near = (x, y, tol) => Math.abs(x - y) <= tol;
const nearRel = (x, y, rtol) => Math.abs(x - y) <= rtol * Math.abs(y);
function section(t) { console.log('\n── ' + t + ' ' + '─'.repeat(Math.max(0, 58 - t.length))); }

// --- deterministic data generators (seeded) ---------------------------------
function genNormal(n, seed, mu = 0, sd = 1) {
  const r = makeRNG(seed), out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = mu + sd * normInv(r());
  return out;
}
function genAR1(n, seed, phi, sd = 1) {
  const r = makeRNG(seed), out = new Array(n);
  let x = 0;
  for (let i = 0; i < n; i++) { x = phi * x + sd * normInv(r()); out[i] = x; }
  return out;
}
// plug-in (÷n) variance — named here because it is the ONE place the biased
// divisor is used, to demonstrate that bootstrap bias estimation works.
const varBiased = (s) => { const m = mean(s); let ss = 0; for (const v of s) ss += (v - m) * (v - m); return ss / s.length; };

/* =====================================================================
 * Resampling / statistics
 * ===================================================================*/
section('Resampling / statistics');
{
  const data = [3, 1, 4, 1, 5, 9, 2, 6];
  const r = makeRNG(11);
  const rs = resample(data, r);
  const srcSet = new Set(data);
  ok('resample has length n', rs.length === data.length);
  ok('resample contains only source values', rs.every(v => srcSet.has(v)));

  const a = resample(data, makeRNG(77));
  const b = resample(data, makeRNG(77));
  ok('fixed seed → identical resample', a.every((v, i) => v === b[i]));

  // quantile hand-values on [1..10] under the pinned type-7 convention
  const s10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  ok('quantile p=0   = 1',    quantile(s10, 0) === 1);
  ok('quantile p=1   = 10',   quantile(s10, 1) === 10);
  ok('quantile p=0.5 = 5.5',  near(quantile(s10, 0.5), 5.5, 1e-12));   // h=4.5 → 5 + .5(6−5)
  ok('quantile p=0.25= 3.25', near(quantile(s10, 0.25), 3.25, 1e-12)); // h=2.25 → 3 + .25(4−3)
  ok('quantile p=0.1 = 1.9',  near(quantile(s10, 0.1), 1.9, 1e-12));   // h=0.9 → 1 + .9(2−1)

  // maxDrawdown on a known compounded path: 1 →1.1 →0.88 →0.924; peak 1.1; worst DD .22/1.1 = .20
  ok('maxDrawdown known path = 0.20', near(maxDrawdown([0.10, -0.20, 0.05]), 0.20, 1e-12));
}

/* =====================================================================
 * iid bootstrap
 * ===================================================================*/
section('iid bootstrap');
{
  // Headline: bootstrap SE of the mean ≈ s/√n. The ideal bootstrap SE equals
  // √((n−1)/n)·s/√n (it uses the ÷n variance internally), so it sits a
  // systematic ~0.6% below s/√n at n=80; 5% tolerance covers that plus MC noise.
  const data = genNormal(80, 2024, 0.01, 0.05);
  const B = 50000, seed = 909;
  const boot = bootstrap(data, mean, { B, seed });
  const analytic = std(data) / Math.sqrt(data.length);
  ok('SE(mean) ≈ s/√n (5% rel)', nearRel(boot.se, analytic, 0.05));

  // Bias of the mean ≈ 0 (the mean is a linear, unbiased statistic).
  ok('bias(mean) ≈ 0', Math.abs(boot.bias) < 0.1 * boot.se);

  // Bias of the ÷n variance ≈ −σ̂²/n exactly (E*[var*] = (n−1)/n · var).
  const bv = bootstrap(data, varBiased, { B, seed: seed + 1 });
  const vb = varBiased(data), target = -vb / data.length;
  ok('bias(varBiased) < 0', bv.bias < 0);
  ok('bias(varBiased) ≈ −σ̂²/n (15% rel)', nearRel(bv.bias, target, 0.15));

  // Reproducibility of replicates.
  const r1 = bootstrap(data, mean, { B: 5000, seed: 5 }).replicates;
  const r2 = bootstrap(data, mean, { B: 5000, seed: 5 }).replicates;
  ok('fixed seed → identical replicates', r1.every((v, i) => v === r2[i]));
}

/* =====================================================================
 * CI ladder
 * ===================================================================*/
section('CI ladder');
{
  const data = genNormal(60, 314, 0.008, 0.045);
  const B = 40000, seed = 271;
  for (const [nm, st] of [['mean', mean], ['sharpe', (s) => sharpe(s)], ['maxDrawdown', maxDrawdown]]) {
    const boot = bootstrap(data, st, { B, seed });
    const est = boot.estimate;
    const N = ciNormal(est, boot.se, 0.05);
    const Ba = ciBasic(est, boot.replicates, 0.05);
    const P = ciPercentile(boot.replicates, 0.05);
    const C = ciBCa(data, st, boot.replicates, est, { alpha: 0.05 });
    for (const [cn, ci] of [['normal', N], ['basic', Ba], ['percentile', P], ['bca', C]])
      ok(`${nm}/${cn}: lo ≤ est ≤ hi`, ci.lo <= est + 1e-12 && est <= ci.hi + 1e-12);
  }

  // Widens with confidence level (percentile widths, same replicates).
  const boot = bootstrap(data, mean, { B, seed });
  const w90 = (() => { const c = ciPercentile(boot.replicates, 0.10); return c.hi - c.lo; })();
  const w98 = (() => { const c = ciPercentile(boot.replicates, 0.02); return c.hi - c.lo; })();
  ok('interval widens with confidence (98% > 90%)', w98 > w90);

  // Narrows as n grows: same DGP, SE(n=200) < SE(n=50).
  const big = genNormal(200, 4242, 0, 0.05);
  const se50 = bootstrap(big.slice(0, 50), mean, { B: 20000, seed: 1 }).se;
  const se200 = bootstrap(big, mean, { B: 20000, seed: 1 }).se;
  ok('SE narrows as n grows (n=200 < n=50)', se200 < se50);

  // On symmetric, unbiased input the three simple intervals agree at large B.
  const sym = genNormal(120, 8, 0, 1); // mean of near-symmetric data
  const bs = bootstrap(sym, mean, { B: 60000, seed: 2 });
  const est = bs.estimate;
  const N = ciNormal(est, bs.se, 0.05), Ba = ciBasic(est, bs.replicates, 0.05), P = ciPercentile(bs.replicates, 0.05);
  const width = P.hi - P.lo;
  ok('symmetric: percentile ≈ normal (lo)', near(P.lo, N.lo, 0.05 * width));
  ok('symmetric: percentile ≈ normal (hi)', near(P.hi, N.hi, 0.05 * width));
  ok('symmetric: percentile ≈ basic  (lo)', near(P.lo, Ba.lo, 0.05 * width));
}

/* =====================================================================
 * BCa
 * ===================================================================*/
section('BCa');
{
  // Reduction identity (exact, no Monte-Carlo): z₀=0, a=0 ⇒ adjusted probs are
  // exactly [α/2, 1−α/2], so BCa collapses to the percentile interval.
  const [a1, a2] = bcaAlphas(0, 0, 0.05);
  ok('reduction identity α₁ = 0.025 exactly', near(a1, 0.025, 1e-12));
  ok('reduction identity α₂ = 0.975 exactly', near(a2, 0.975, 1e-12));
  // and a non-trivial (z₀,a) reduces to a clean hand check when a=0:
  //   α₁ = Φ(2z₀ + z_{α/2}). With z₀ = 0.1: Φ(0.2 − 1.959963985).
  ok('bcaAlphas(z₀=0.1,a=0) matches Φ(2z₀+z_α)',
     near(bcaAlphas(0.1, 0, 0.05)[0], normCDF(0.2 + normInv(0.025)), 1e-12));

  // Sign of the bias correction — mechanical/exact via biasCorrection:
  //   > half of replicates below estimate ⇒ z₀ > 0; < half ⇒ z₀ < 0; half ⇒ 0.
  const reps = []; for (let i = 0; i < 1000; i++) reps.push(i); // 0..999
  ok('z₀ > 0 when 75% below estimate', biasCorrection(reps, 750, reps.length) > 0);
  ok('z₀ < 0 when 25% below estimate', biasCorrection(reps, 250, reps.length) < 0);
  ok('z₀ ≈ 0 when 50% below estimate', near(biasCorrection(reps, 500, reps.length), 0, 1e-9));

  // Realistic median-bias: the sample maximum is median-biased (bootstrap
  // distribution has a point mass at the top, a left tail), so ~37% of
  // replicates fall strictly below the estimate → z₀ solidly negative.
  const d = genNormal(50, 17, 0, 1);
  const bm = bootstrap(d, (s) => Math.max(...s), { B: 10000, seed: 3 });
  const z0max = biasCorrection(bm.replicates, bm.estimate, bm.B);
  ok('z₀ < 0 for the sample maximum (median-biased up)', z0max < 0);

  // Acceleration equals the standardized jackknife skewness. Derived closed form
  // for the MEAN: the (n−1) factors cancel, leaving a = Σd³ / (6·(Σd²)^{3/2}).
  const arr = [0.4, -1.1, 2.3, 0.7, -0.5, 1.9, -2.2, 0.1];
  const m = mean(arr); let s2 = 0, s3 = 0; for (const v of arr) { const dd = v - m; s2 += dd * dd; s3 += dd * dd * dd; }
  const aClosed = s3 / (6 * Math.pow(s2, 1.5));
  ok('acceleration(mean) = closed-form jackknife skew', near(acceleration(arr, mean), aClosed, 1e-12));
}

/* =====================================================================
 * Block bootstrap
 * ===================================================================*/
section('Block bootstrap');
{
  const data = genNormal(120, 55, 0, 1);
  const B = 20000, seed = 606;

  // Reduction: moving-block ℓ=1 shares the iid index map and rng stream, so the
  // replicate stream is BIT-IDENTICAL to the iid bootstrap.
  const iid = bootstrap(data, mean, { B, seed }).replicates;
  const mb1 = blockBootstrap(data, mean, { B, blockLength: 1, seed }).replicates;
  ok('moving-block ℓ=1 == iid bootstrap (bit-identical)', iid.every((v, i) => v === mb1[i]));

  // Reduction: stationary p=1 ⇒ every block length 1 ⇒ iid (distributionally;
  // the continuation coin-flips make the stream differ, so compare SE).
  const seIID = bootstrap(data, mean, { B, seed: 2 }).se;
  const seP1 = stationaryBootstrap(data, mean, { B, p: 1, seed: 2 }).se;
  ok('stationary p=1 ≈ iid bootstrap (SE, 5% rel)', nearRel(seP1, seIID, 0.05));

  // Domain justification: positive autocorrelation inflates Var(mean); iid
  // resampling ignores it and understates SE, block resampling captures it.
  const ar = genAR1(300, 9001, 0.9, 1);
  const seIidAR = bootstrap(ar, mean, { B: 5000, seed: 7 }).se;
  const seBlkAR = blockBootstrap(ar, mean, { B: 5000, blockLength: 25, seed: 7 }).se;
  ok('AR(1) φ=0.9: block SE > iid SE', seBlkAR > seIidAR);

  // Reproducibility.
  const q1 = blockBootstrap(data, mean, { B: 4000, blockLength: 10, seed: 8 }).replicates;
  const q2 = blockBootstrap(data, mean, { B: 4000, blockLength: 10, seed: 8 }).replicates;
  ok('block: fixed seed → identical replicates', q1.every((v, i) => v === q2[i]));
}

/* =====================================================================
 * Golden numbers  —  PROVISIONAL (computed against the PLACEHOLDER PRNG).
 * ---------------------------------------------------------------------
 * Deterministic given the seed, so these are exact backward-compatibility
 * locks (1e-12 abs). ANY future change that moves them is breaking and must be
 * acknowledged. >>> Regenerate against the real aestimo-random.js and re-lock. <<<
 * ===================================================================*/
section('Golden numbers (PROVISIONAL — placeholder PRNG)');
{
  // 24 monthly returns, fixed.
  const GOLDEN_DATA = [
     0.031, -0.018,  0.024,  0.007, -0.042,  0.055,  0.012, -0.009,
     0.038,  0.001, -0.027,  0.019,  0.046, -0.061,  0.033,  0.028,
    -0.014,  0.052,  0.006, -0.033,  0.021,  0.017, -0.008,  0.040
  ];
  const GS = 20250101, GB = 40000;

  const seMean = bootstrap(GOLDEN_DATA, mean, { B: GB, seed: GS }).se;

  const bootSh = bootstrap(GOLDEN_DATA, (s) => sharpe(s), { B: GB, seed: GS });
  const bca = ciBCa(GOLDEN_DATA, (s) => sharpe(s), bootSh.replicates, bootSh.estimate, { alpha: 0.05 });

  const seStat = stationaryBootstrap(GOLDEN_DATA, mean, { B: GB, p: 0.1, seed: GS }).se;

  console.log('    GOLDEN iid SE(mean)      = ' + seMean.toPrecision(17));
  console.log('    GOLDEN BCa Sharpe lo/hi  = ' + bca.lo.toPrecision(17) + ' , ' + bca.hi.toPrecision(17));
  console.log('    GOLDEN stationary SE     = ' + seStat.toPrecision(17));

  // Locked from the first fully-green build (placeholder PRNG, seed 20250101,
  // B=40000). This series carries strong NEGATIVE lag-1 autocorrelation
  // (acf₁ ≈ −0.46), so the stationary SE sits below the iid SE — the mirror of
  // the AR(1) positive case, and useful coverage of the "block < iid" branch.
  // Golden regression locks — LOCKED against the real xoshiro128** random
  // (aestimo-random.js) on 2026-07-06 at the author's confirmation. These
  // replaced the earlier sfc32 stand-in values (shown for provenance); the
  // ~0.3% shift is the expected Monte-Carlo effect of the PRNG substitution
  // (kickoff Phase 1 step 4). The 44 analytic invariants are generator-agnostic
  // and were unaffected. See reports/boot-relock.md.
  //   old sfc32 stand-in: seMean 0.0061033074372447999 · bcaShLo -0.15162873138274377
  //                       bcaShHi 0.76658152274982616  · seStat  0.0022165210720237927
  const GOLDEN = {
    seMean:   0.0060824815377683653,
    bcaShLo: -0.14954530430438817,
    bcaShHi:  0.76571828636653472,
    seStat:   0.0022069183133171036
  };
  if (GOLDEN.seMean != null) {
    ok('GOLDEN iid SE(mean) locked',      near(seMean, GOLDEN.seMean, 1e-12));
    ok('GOLDEN BCa Sharpe lower locked',  near(bca.lo, GOLDEN.bcaShLo, 1e-12));
    ok('GOLDEN BCa Sharpe upper locked',  near(bca.hi, GOLDEN.bcaShHi, 1e-12));
    ok('GOLDEN stationary SE locked',     near(seStat, GOLDEN.seStat, 1e-12));
  }
}

/* =====================================================================
 * selfTest parity
 * ===================================================================*/
section('selfTest parity');
ok('engine selfTest() green', A.selfTest() === true);

// --- summary ----------------------------------------------------------------
console.log('\n' + '═'.repeat(62));
console.log(`  ${pass} passed, ${fail} failed`);
if (fail) { console.log('  FAILURES: ' + failed.join(' | ')); process.exit(1); }
console.log('  ALL GREEN');
process.exit(0);
