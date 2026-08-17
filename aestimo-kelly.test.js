/*
 * aestimo-kelly.test.js — headless runner for aestimo-kelly.js.
 *
 * Prints PASS/FAIL per invariant and exits non-zero on any failure. The invariant
 * set lives in the engine's exported battery() (single source of truth); this file
 * renders it and then verifies the browser (UMD) code path in an isolated vm
 * context, matching the discipline used for aestimo-solve / aestimo-chol / aestimo-random.
 *
 * Golden fixtures (locked at first green, 1e-12):
 *   kellyBinary({p:0.62, b:2.3})                              = 0.454782608695652
 *   kellySample([...,-0.35])  (fat-tail μ/σ² divergence case) = 0.262507206188192
 *   growthRatio(0.5)                                          = 0.75
 *   drawdownProb(0.5, 0.5)                                    = 0.125
 *   kellyVector(MU3, SIG3, RF3)  = [1.354545454545454, 0.450000000000000, 0.263636363636364]
 * Root-finds run at tol 1e-14 so located fractions are stable to <1e-12 across
 * any correct safeguarded solver.
 */
'use strict';
var fs = require('fs');
var path = require('path');
var vm = require('vm');

var K = require('./aestimo-kelly.js');

var results = K.battery();
var failed = 0;
results.forEach(function (r) {
  var line = (r.pass ? 'PASS' : 'FAIL') + '  ' + r.name;
  if (r.detail) line += '   (' + r.detail + ')';
  console.log(line);
  if (!r.pass) failed++;
});

// selfTest() must agree with the rendered battery (boolean fold).
var st = K.selfTest();
console.log((st === (failed === 0) ? 'PASS' : 'FAIL') + '  selfTest() agrees with battery');
if (st !== (failed === 0)) failed++;

// ---- Browser code-path check (isolated vm, no `module` in scope) --------------
// Run each engine's browser UMD branch in one sandbox: solve and chol register
// themselves as globals, then kelly composes them off globalThis exactly as it
// would in the browser. Confirms the UMD wrapper resolves deps correctly there.
(function verifyBrowserPath() {
  var sandbox = {};
  vm.createContext(sandbox);
  ['aestimo-solve.js', 'aestimo-chol.js', 'aestimo-kelly.js'].forEach(function (f) {
    var src = fs.readFileSync(path.join(__dirname, f), 'utf8');
    vm.runInContext(src, sandbox, { filename: f });
  });
  var ok = sandbox.AestimoKelly &&
    typeof sandbox.AestimoKelly.selfTest === 'function' &&
    sandbox.AestimoKelly.selfTest() === true;
  console.log((ok ? 'PASS' : 'FAIL') + '  browser (vm) path: AestimoKelly.selfTest() === true');
  if (!ok) failed++;
})();

console.log('\n' + (failed === 0 ? 'ALL GREEN — ' + results.length + ' invariants + selfTest + browser path'
  : failed + ' FAILURE(S)'));
process.exit(failed === 0 ? 0 : 1);
