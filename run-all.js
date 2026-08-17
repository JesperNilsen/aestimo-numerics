/*
 * run-all.js — the suite battery aggregator. `npm test` entry point.
 *
 * The canonical module runners (aestimo-*.test.js) are standalone: each prints
 * its own PASS/FAIL table and calls process.exit. So we run them as SUBPROCESSES
 * and aggregate their exit codes — never require() them (that would exit us).
 * The cross-module check is safe to require (no self-exit), so it runs
 * in-process. Exits non-zero if anything fails.
 *
 * Adapted from Aestimo's engine/run-all.js: trimmed to the six modules shipped
 * in this repository (the dcf/portfolio/console sections live with the Aestimo
 * product).
 */
'use strict';

var cp = require('child_process');
var path = require('path');
var ENGINE = __dirname;

// Dependency order matters for readability (random→boot; solve+chol→kelly),
// but each runner loads its own deps by require, so order here is cosmetic.
var RUNNERS = ['chol', 'random', 'solve', 'boot', 'beta', 'kelly'];

var anyFail = false;
var summary = [];

RUNNERS.forEach(function (m) {
  var file = path.join(ENGINE, 'aestimo-' + m + '.test.js');
  console.log('=== ' + m + ' ===');
  try {
    process.stdout.write(cp.execFileSync('node', [file], { encoding: 'utf8' }));
    summary.push(m + ': green');
  } catch (e) {
    if (e.stdout) process.stdout.write(e.stdout);
    if (e.stderr) process.stderr.write(e.stderr);
    anyFail = true;
    summary.push(m + ': FAIL');
  }
  console.log('');
});

// ---- cross-module consistency (spans two engines; lives in neither battery) ----
console.log('=== cross-module ===');
var cross = (function () {
  var Chol = require('./aestimo-chol.js');
  var Rand = require('./aestimo-random.js');
  var results = [];
  function ok(name, cond, detail) { results.push({ name: name, pass: !!cond, detail: detail || '' }); }

  // random ships its own inline cholesky (for correlatedNormals) — a second
  // copy of chol's algorithm, kept inline ON PURPOSE: physically depending on
  // chol would splice all of chol into the random + boot benches to reuse one
  // ~25-line function (bench bloat for a tiny DRY win). Instead this lock makes
  // the duplication provably benign — the two must agree on BOTH paths, so
  // neither can silently drift from the other.
  //
  // (a) Success path: identical L across sizes 2–5, varied conditioning + scale.
  var spd = [
    [[4, 2], [2, 3]],
    [[6, 2, 1], [2, 5, 2], [1, 2, 4]],
    [[1, 0.5], [0.5, 1]],
    [[400, 200], [200, 300]],                                   // scaled ×100
    [[10, 1, 2, 0], [1, 8, 1, 3], [2, 1, 9, 1], [0, 3, 1, 7]],  // 4×4 diag-dominant
    [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0],
      [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]                       // 5×5 tridiagonal SPD
  ];
  var agree = true, maxRel = 0;
  spd.forEach(function (A) {
    var rL = Rand.cholesky(A);
    var cR = Chol.cholesky(A);
    if (!cR.ok) { agree = false; return; }
    for (var i = 0; i < A.length; i++) {
      for (var j = 0; j < A.length; j++) {
        var a = rL[i][j] || 0, b = cR.L[i][j] || 0;
        var rel = Math.abs(a - b) / (1 + Math.max(Math.abs(a), Math.abs(b)));
        if (rel > maxRel) maxRel = rel;
        if (rel > 1e-12) agree = false;
      }
    }
  });
  ok('random.cholesky ≡ chol.cholesky on the success path (6 SPD, sizes 2–5, scaled)',
    agree, 'max rel |Δ| = ' + maxRel.toExponential(2));

  // (b) Failure path: both must REJECT the same symmetric non-PD inputs —
  // random by throwing (its contract, asserted in its own battery), chol by
  // {ok:false}. (Only symmetric inputs: chol also rejects asymmetry, which
  // random doesn't test for, so the two legitimately differ off the diagonal
  // of symmetry — out of scope for this equivalence.)
  var nonPD = [[[1, 2], [2, 1]], [[-1, 0], [0, 1]], [[2, 3], [3, 2]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]];
  var failAgree = true;
  nonPD.forEach(function (A) {
    var randRejected = false;
    try { Rand.cholesky(A); } catch (e) { randRejected = true; }
    if (randRejected !== !Chol.cholesky(A).ok) failAgree = false;
  });
  ok('random.cholesky and chol.cholesky reject the same non-PD inputs',
    failAgree, nonPD.length + ' symmetric non-PD matrices');
  return results;
})();
cross.forEach(function (x) {
  console.log((x.pass ? 'PASS' : 'FAIL') + '  ' + x.name + (x.detail ? '  · ' + x.detail : ''));
  if (!x.pass) anyFail = true;
});
summary.push('cross-module: ' + cross.filter(function (x) { return x.pass; }).length + '/' + cross.length);
console.log('');

console.log('=== battery summary ===');
summary.forEach(function (s) { console.log(s); });
console.log(anyFail ? 'FAILURES PRESENT' : 'ALL GREEN');
process.exit(anyFail ? 1 : 0);
