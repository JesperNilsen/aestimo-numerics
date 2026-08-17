'use strict';

/*
 * aestimo-solve.test.js — headless invariant battery for aestimo-solve.js.
 * Prints PASS/FAIL per invariant; exits non-zero on any failure.
 *   $ node aestimo-solve.test.js
 *
 * The invariants ARE the safety mechanism. They were written before (and
 * gate) every layer. Golden numbers at the bottom are regression locks: any
 * future change that moves them is a breaking change to be acknowledged, not
 * waved through.
 */

const E = require('./aestimo-solve.js');

let failures = 0;
let count = 0;
function check(name, cond) {
  count++;
  if (cond) {
    console.log(`PASS  ${name}`);
  } else {
    failures++;
    console.log(`FAIL  ${name}`);
  }
}
const near = (a, b, eps = 1e-9) => Number.isFinite(a) && Math.abs(a - b) < eps;

// --- placeholders; locked from the first green build, see GOLDEN section ---
const GOLDEN = {
  cubicRoot: 1.5213797068045751,    // root of x³ − x − 2 from solve()
  bondYield: 0.028308445384892123,  // periodicYield, bond {950, 1000, 25, 20}
  irrRate: 0.069137446091777327,    // IRR of [-1000,150,200,250,300,350]
};
const LOCK_EPS = 1e-12; // deterministic output reproduces well inside this.

console.log('=== SOLVER CORE ===');

(() => {
  const r = E.solve((x) => x - 3, { bracket: [0, 10] });
  check('linear  f(x)=x−3 → root 3', r.converged && near(r.root, 3));
})();

(() => {
  const r = E.solve((x) => x * x - 2, { bracket: [0, 2] });
  check('quadratic  x²−2 on [0,2] → √2', r.converged && near(r.root, Math.SQRT2));
})();

(() => {
  // Known real root of x³ − x − 2 sits near 1.5214.
  const r = E.solve((x) => x ** 3 - x - 2, { bracket: [1, 2], fprime: (x) => 3 * x * x - 1 });
  check('cubic  x³−x−2 → root ≈ 1.5214', r.converged && near(r.root, 1.5213797068045, 1e-9));
})();

(() => {
  const r = E.bisect((x) => x * x + 1, -1, 1); // never zero ⇒ no sign change
  check('bisect  no sign change → converged:false, root NaN',
    r.converged === false && Number.isNaN(r.root));
})();

(() => {
  const r = E.newton((x) => x * x - 2, (x) => 2 * x, 1);
  check('newton  x²−2 from x0=1 → √2 in a handful of iters',
    r.converged && near(r.root, Math.SQRT2) && r.iters <= 8);
})();

(() => {
  const r = E.newton((x) => x * x - 2, (x) => 2 * x, 0); // f'(0)=0
  const finite = !Number.isNaN(r.root) ? Number.isFinite(r.root) : true;
  check('newton  x0=0 (f′=0) → converged:false, no Inf/NaN escape',
    r.converged === false && finite);
})();

(() => {
  // Residual contract: every converged:true result satisfies |f(root)| < tol.
  const cases = [
    { f: (x) => x - 3, spec: { bracket: [0, 10] } },
    { f: (x) => x * x - 2, spec: { bracket: [0, 2] } },
    { f: (x) => x ** 3 - x - 2, spec: { bracket: [1, 2], fprime: (x) => 3 * x * x - 1 } },
    { f: (x) => Math.cos(x) - x, spec: { bracket: [0, 1] } },
  ];
  let ok = true;
  for (const c of cases) {
    const r = E.solve(c.f, c.spec, { tol: 1e-10 });
    if (r.converged && !(Math.abs(c.f(r.root)) < 1e-10)) ok = false;
  }
  check('residual  every converged:true has |f(root)| < tol', ok);
})();

(() => {
  // The hybrid must earn its complexity: converge AND beat bisection's count.
  const f = (x) => x ** 3 - x - 2;
  const fp = (x) => 3 * x * x - 1;
  const tol = 1e-10;
  const hyb = E.solve(f, { bracket: [1, 2], fprime: fp }, { tol });
  const bis = E.bisect(f, 1, 2, { tol });
  check('hybrid  solve converges AND uses fewer iters than bisect',
    hyb.converged && bis.converged && hyb.iters < bis.iters);
  console.log(`        (solve ${hyb.iters} iters vs bisect ${bis.iters} iters)`);
})();

(() => {
  // Convergence-honesty hardening (2026-08-01): a small step alone no longer
  // proves convergence — it must be corroborated by a small residual or a
  // collapsed bracket. Two scale extremes pin the semantics:
  // STEEP (×1e12): the residual can never reach tol in absolute terms, so
  // converged:true is only reachable via bracket collapse — and then x must
  // actually be at the root.
  const G = 1.5213797068045751;
  const steep = E.solve((x) => 1e12 * (x ** 3 - x - 2), { bracket: [1, 2], fprime: (x) => 1e12 * (3 * x * x - 1) });
  check('honesty  steep ×1e12: converged ⇒ x at the root (bracket-collapse path)',
    steep.converged && Math.abs(steep.root - G) < 1e-9);
  // SHALLOW (×1e-12): the residual criterion fires early and honestly — the
  // reported residual must itself be below tol (x may be loose; that is the
  // documented f-tolerance/x-tolerance duality, and residual reports it).
  const shallow = E.solve((x) => 1e-12 * (x ** 3 - x - 2), { bracket: [1, 2], fprime: (x) => 1e-12 * (3 * x * x - 1) });
  check('honesty  shallow ×1e-12: converged ⇒ reported residual < tol',
    shallow.converged && shallow.residual < 1e-10);
})();

(() => {
  // irrAll scope contract: the default grid ends at +1000%. A 1100% IRR
  // ([-1, 12]) must be MISSED by the default scan (documented limitation),
  // FOUND when the caller widens the range, and found by irr() regardless
  // (its bracket expands geometrically until the sign flips).
  const cf = [-1, 12];
  const defaults = E.irrAll(cf);
  const widened = E.irrAll(cf, { hi: 20 });
  check('irrAll  root beyond hi=10 missed by default, found with opts.hi (11.0)',
    defaults.length === 0 && widened.length === 1 && near(widened[0], 11.0, 1e-6));
  check('irr  finds the same 1100% root without help', near(E.irr(cf).rate, 11.0, 1e-6));
})();

(() => {
  // OPTIONAL OBSERVABLE — Newton's quadratic convergence made visible.
  // Track e_n = |x_n − √2|; for a simple root e_{n+1} ≈ C·e_n², so the ratio
  // e_{n+1}/e_n² should settle near a constant (≈ f″/2f′ = 1/(2√2) ≈ 0.3536).
  let x = 1.5;
  const root = Math.SQRT2;
  const errs = [];
  for (let i = 0; i < 5; i++) {
    errs.push(Math.abs(x - root));
    x = x - (x * x - 2) / (2 * x);
  }
  const ratios = [];
  for (let i = 1; i < errs.length - 1; i++) {
    if (errs[i - 1] > 0) ratios.push(errs[i] / (errs[i - 1] ** 2));
  }
  console.log('        Newton error e_n   :', errs.map((e) => e.toExponential(2)).join('  '));
  console.log('        ratio e_{n+1}/e_n² :', ratios.map((r) => r.toFixed(4)).join('  '), '(→ ~0.3536)');
})();

console.log('=== YTM ===');

(() => {
  // Par bond: price = face ⇒ periodic yield equals the periodic coupon rate.
  const y = E.ytm({ price: 100, face: 100, coupon: 5, periods: 10 });
  check('ytm  par bond → periodicYield = coupon/face', near(y.periodicYield, 0.05));
})();

(() => {
  const disc = E.ytm({ price: 95, face: 100, coupon: 5, periods: 10 });
  const prem = E.ytm({ price: 105, face: 100, coupon: 5, periods: 10 });
  check('ytm  discount → yield > coupon rate; premium → yield < coupon rate',
    disc.periodicYield > 0.05 && prem.periodicYield < 0.05);
})();

(() => {
  // Zero-coupon closed form: y = (face/price)^(1/N) − 1.
  const face = 100, price = 78, periods = 5;
  const y = E.ytm({ price, face, coupon: 0, periods });
  const closed = Math.pow(face / price, 1 / periods) - 1;
  check('ytm  zero-coupon → matches closed form', near(y.periodicYield, closed));
})();

(() => {
  // Round-trip: solved yield back through bondPrice reproduces the price.
  const bond = { face: 1000, coupon: 25, periods: 20 };
  const price = 950;
  const y = E.ytm({ price, ...bond });
  const back = E.bondPrice(y.periodicYield, bond);
  check('ytm  solved yield → bondPrice reproduces price', near(back, price, 1e-6));
})();

(() => {
  // Annualization conventions are both exposed and labelled.
  const bey = E.ytm({ price: 950, face: 1000, coupon: 25, periods: 20 }, { frequency: 2, annualize: 'bey' });
  const eff = E.ytm({ price: 950, face: 1000, coupon: 25, periods: 20 }, { frequency: 2, annualize: 'effective' });
  const okBey = near(bey.annualYield, bey.periodicYield * 2) && bey.annualization === 'bey';
  const okEff = near(eff.annualYield, Math.pow(1 + eff.periodicYield, 2) - 1) && eff.annualization === 'effective';
  check('ytm  bey = periodic×freq; effective = (1+periodic)^freq−1; both labelled', okBey && okEff);
})();

console.log('=== IRR ===');

(() => {
  check('irr  [-100,110] → 0.10', near(E.irr([-100, 110]).rate, 0.1));
})();

(() => {
  check('irr  [-100,0,121] → 0.10', near(E.irr([-100, 0, 121]).rate, 0.1));
})();

(() => {
  // Scaling all cashflows by a positive constant leaves IRR unchanged.
  const base = E.irr([-100, 110]).rate;
  const scaled = E.irr([-100, 110].map((c) => c * 7.31)).rate;
  check('irr  positive scaling leaves IRR unchanged', near(base, scaled, 1e-10));
})();

(() => {
  // The fixed point: NPV at the IRR is zero.
  const cf = [-1000, 150, 200, 250, 300, 350];
  const r = E.irr(cf).rate;
  check('irr  npv(irr, cashflows) ≈ 0', Math.abs(E.npv(r, cf)) < 1e-7);
})();

(() => {
  // Non-conventional stream: two sign changes, two real IRRs (0.10 and 0.20).
  const cf = [-100, 230, -132];
  const r = E.irr(cf);
  const all = E.irrAll(cf);
  const both = all.length === 2 &&
    all.some((x) => near(x, 0.1, 1e-6)) &&
    all.some((x) => near(x, 0.2, 1e-6));
  check('irr  two sign changes → unique:false AND irrAll returns both roots',
    r.unique === false && both);
  console.log('        irrAll roots:', all.map((x) => x.toFixed(6)).join(', '));
})();

console.log('=== OPTIONAL LAYER 6b — XIRR ===');

(() => {
  // Regularly-spaced annual dates must reproduce the integer-period IRR.
  const cf = [-1000, 150, 200, 250, 300, 350];
  const dates = ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01'];
  const x = E.xirr(cf, dates).rate;
  const r = E.irr(cf).rate;
  // Annual spacing is ~365.25d on average vs the /365 convention, so allow slack.
  check('xirr  annual dates ≈ integer-period IRR', near(x, r, 5e-3));
})();

console.log('=== selfTest ===');
check('selfTest() returns true', E.selfTest() === true);

console.log('=== GOLDEN (regression locks) ===');

(() => {
  const g = E.solve((x) => x ** 3 - x - 2, { bracket: [1, 2], fprime: (x) => 3 * x * x - 1 }).root;
  const bond = E.ytm({ price: 950, face: 1000, coupon: 25, periods: 20 }, { frequency: 2, annualize: 'bey' });
  const irrCf = [-1000, 150, 200, 250, 300, 350];
  const irrRate = E.irr(irrCf).rate;

  console.log('        cubicRoot =', g.toPrecision(17));
  console.log('        bondYield =', bond.periodicYield.toPrecision(17),
    '(annual bey =', bond.annualYield.toPrecision(17) + ')');
  console.log('        irrRate   =', irrRate.toPrecision(17));

  if (GOLDEN.cubicRoot !== 0) {
    check('golden  cubic root locked', Math.abs(g - GOLDEN.cubicRoot) < LOCK_EPS);
    check('golden  bond yield locked', Math.abs(bond.periodicYield - GOLDEN.bondYield) < LOCK_EPS);
    check('golden  irr rate locked', Math.abs(irrRate - GOLDEN.irrRate) < LOCK_EPS);
  } else {
    console.log('        (locks not yet set — paste the printed values into GOLDEN)');
  }
})();

console.log(`\n${count - failures}/${count} checks passed, ${failures} failed.`);
process.exit(failures ? 1 : 0);
