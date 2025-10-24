### PR #1: MetalLayout API & deviceGridShape (6 commits)
* **Impact:** Changes existing functionality
* **Details:** Modifies MetalLayoutAttr API to require deviceGridShape parameter
* **Affected:** All code creating MetalLayoutAttr (TTIRToD2M, GridSelection, CAPI, Python bindings)
* **Dependencies:** None (foundation layer)
* **Risk:** **MEDIUM** - API change touches multiple components, but backward compatible (defaults provided)

---

### PR #2: D2M GenericOp Enhancements (6 commits)
* **Impact:** Changes existing functionality
* **Details:** Makes GenericOp indexing_maps optional, adds explicit block_factors mode
* **Affected:** D2M GenericOp behavior, transform passes (Allocate, ElementwiseFusion, GridSelection)
* **Dependencies:** **PR #1** (uses deviceGridShape in GridSelection)
* **Risk:** **MEDIUM** - Changes core GenericOp semantics, new execution paths

---

### PR #3: Bufferization Infrastructure (2 commits)
* **Impact:** Does not touch existing functionality
* **Details:** Adds new D2MBufferizeFunctionArgs pass to pipeline
* **Affected:** Only bufferization pipeline (adds pass after one-shot-bufferize)
* **Dependencies:** **PR #1** (preserves MetalLayoutAttr through bufferization)
* **Risk:** **LOW** - New isolated pass, doesn't modify existing behavior

---

### PR #4: Pykernel Python DSL + Linalg (5 commits)
* **Impact:** Does not touch existing functionality
* **Details:** New Python DSL for custom kernels, generates linalg.generic from Python
* **Affected:** Only new pykernel/ directory and tests
* **Dependencies:** **PR #1, #2, #3** (uses MetalLayoutAttr API, GenericOp with optional indexing_maps, bufferization pass)
* **Risk:** **LOW** - New feature, isolated to pykernel DSL (doesn't affect TTIR/TTNN paths)

---

### PR #5: Bug Fixes & Optimizations (4 commits)
* **Impact:** Changes existing functionality
* **Details:** Fixes bugs and adds workarounds across multiple components
* **Affected:** Flatbuffer generation, TTIRToD2M layout handling, GenericOp canonicalizer, D2M Allocate pass
* **Dependencies:** **PR #2** (fixes for GenericOp canonicalizer), **Indirect: PR #1** (TTIRToD2M uses deviceGridShape)
* **Risk:** **HIGH** - Multiple workarounds with fragile assumptions, touches critical paths

---

### PR #6: Development Infrastructure (2 commits)
* **Impact:** Does not touch existing functionality
* **Details:** macOS build support (runtime stubs), debugging tools (signal handler)
* **Affected:** Build system and developer experience only
* **Dependencies:** None (independent)
* **Risk:** **LOW** - Dev-only, no runtime impact

---

### Before PR #1
- [ ] Address deviceGridShape hardcoding or document why {1, 1} is correct

### Before PR #2
- [ ] Review FIXMEs in commits #11, #12
- [ ] Consider converting to TODOs if design is settled

### Before PR #4
- [ ] Verify all tests pass
- [ ] Document the linalg.generic approach in commit message

### Before PR #5 (CRITICAL)
- [ ] **FIX OR REMOVE:** Rank-2 tensor workaround (commit #21)
- [ ] Test TileMatmulBlockOp check necessity (commit #22)
- [ ] Consider adding DCE pass (commit #23)
- [ ] Verify flatbuffer fix works (commit #20)

### Before PR #6
- [ ] No issues

## All FIXMEs/TODOs/HACKs Found (Complete List)

### PR #1: MetalLayout API
| Commit | File | Issue | Type |
|--------|------|-------|------|
| #4 | TTIRToD2M.cpp:103 | Hardcoded deviceGridShape {1, 1} | TODO |
| #4 | TTIRToD2M.cpp:152 | Hardcoded deviceGridShape {1, 1} | TODO |
| #4 | TTIRToD2M.cpp:160 | Hardcoded deviceGridShape {1, 1} | TODO |

### PR #2: GenericOp
| Commit | File | Issue | Type |
|--------|------|-------|------|
| #11 | D2MOps.cpp:814 | Empty indexing_maps explanation | FIXME |
| #12 | D2MOps.cpp:1343 | Explicit block_factors mode | FIXME |
| #12 | GridSelection.cpp:513 | Skip pykernel DSL generics | FIXME |

### PR #3: Bufferization
| Commit | File | Issue | Type |
|--------|------|-------|------|
| #13 | BufferizeFunctionArgs.cpp | Issue tracker reference | TODO(#2246) |

### PR #4: Pykernel DSL
**No FIXMEs/TODOs found** ✅

### PR #5: Bug Fixes
| Commit | File | Issue | Type |
|--------|------|-------|------|
| #21 | TTIRToD2M.cpp:133 | **Rank-2 tensor workaround** | **FIXME (HIGH)** |
| #22 | D2MOps.cpp:1251 | TileMatmulBlockOp check | Comment (may be obsolete) |
| #23 | Allocate.cpp:365 | Skip dead allocs workaround | **FIXME (MED)** |

### PR #6: Dev Infrastructure
**No FIXMEs/TODOs found** ✅


  🔴 CRITICAL (1):
  - PR #5, Commit #21: Rank-2 tensor workaround - fragile assumption

  ⚠️ MEDIUM (4):
  - PR #1, Commit #4: Hardcoded deviceGridShape {1, 1}
  - PR #5, Commit #20: Fixed inverted assertion (verify it works)
  - PR #5, Commit #22: TileMatmulBlockOp check may be obsolete
  - PR #5, Commit #23: Dead alloc workaround (should add DCE pass)

  ℹ️ LOW (6):
  - PR #2: 3 FIXMEs documenting pykernel DSL design
  - PR #3: 1 TODO reference

  ✅ CLEAN PRs:
  - PR #4: Pykernel DSL + Linalg (no issues!)
  - PR #6: Dev Infrastructure (no issues!)

  There is another document which includes:
  - Exact file locations and line numbers
  - Why each is an issue
  - Recommended fixes
  - Priority levels
  - Review order (PR #5 first - has critical issue)
