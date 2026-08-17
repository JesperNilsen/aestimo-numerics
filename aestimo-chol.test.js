/*
 * aestimo-chol.test.js — thin node runner over the engine's canonical
 * battery. The invariant checks and golden locks live in aestimo-chol.js
 * (exported as `battery()`), so there is exactly one copy of them: this
 * file prints and exits, the in-browser bench renders the same array.
 *
 * The only test logic that lives *here* is the one thing the engine can't
 * self-check: that the UMD browser code path (globalThis branch) loads in
 * an isolated vm with no `module` object present. That needs fs/vm, which
 * have no place in the engine.
 */
'use strict';

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var C = require('./aestimo-chol.js');

var results = C.battery();
var failures = 0;

results.forEach(function (r) {
  if (r.pass) {
    console.log('PASS  ' + r.name);
  } else {
    console.log('FAIL  ' + r.name + (r.detail ? '  (' + r.detail + ')' : ''));
    failures++;
  }
});

// --- node-only: verify the browser UMD path in an isolated vm context ---
(function () {
  var name = 'UMD: browser path exposes globalThis.AestimoChol & battery green';
  var pass = false, detail = '';
  try {
    var src = fs.readFileSync(path.join(__dirname, 'aestimo-chol.js'), 'utf8');
    var sandbox = {};
    sandbox.self = sandbox; // mimic browser global; no `module` present
    vm.createContext(sandbox);
    vm.runInContext(src, sandbox);
    var B = sandbox.AestimoChol;
    pass = !!(B && typeof B.battery === 'function' && B.selfTest() === true);
  } catch (e) {
    detail = e.message;
  }
  if (pass) {
    console.log('PASS  ' + name);
  } else {
    console.log('FAIL  ' + name + (detail ? '  (' + detail + ')' : ''));
    failures++;
  }
})();

var total = results.length + 1;
console.log('');
console.log((total - failures) + ' / ' + total + ' passed');
if (failures > 0) { console.log(failures + ' FAILURE(S)'); process.exit(1); }
console.log('ALL GREEN');
process.exit(0);
