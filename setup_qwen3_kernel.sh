#!/bin/bash
# Self-contained TidalDecode kernel build for Qwen3-14B (GQA group_size=5).
# Run from the repo root inside the `tidal` conda env. Idempotent.
#   1) init the flashinfer + pybind submodules
#   2) apply the Qwen3 group_size=5 decode patch to the flashinfer SUBMODULE
#      (this lives inside the submodule, so it is NOT carried by a parent-repo clone)
#   3) build the fused CUDA kernels (tidal/ops/setup.sh has the fmt/spdlog fix:
#      CPM_DOWNLOAD_ALL=ON + DISABLE_FIND_PACKAGE_fmt to avoid conda's fmt12 vs spdlog1.12)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
git submodule update --init kernels/3rdparty/flashinfer kernels/3rdparty/pybind || true
FI=kernels/3rdparty/flashinfer
if ! grep -q SWITCH_GQA_GROUP_SIZE_DEC "$FI/include/flashinfer/utils.cuh" 2>/dev/null; then
  git -C "$FI" apply "$HERE/kernels/patches/flashinfer_qwen3_group5.patch" \
    && echo "[ok] flashinfer Qwen3 group5 patch applied" \
    || echo "[WARN] flashinfer patch failed — check submodule is at the pinned commit"
else
  echo "[skip] flashinfer group5 patch already present"
fi
( cd tidal/ops && rm -rf build && bash setup.sh ) \
  && python -c "from tidal import Qwen3ForCausalLM; print('[ok] TidalDecode KERNEL OK')" \
  || { echo "[ERR] tidal kernel build failed"; exit 1; }
