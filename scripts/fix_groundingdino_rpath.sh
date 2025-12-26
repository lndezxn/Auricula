#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.util as u, site, pathlib, subprocess, torch

spec = u.find_spec("groundingdino._C")
if not spec or not spec.origin:
    raise SystemExit("Could not locate groundingdino._C .so")

so = pathlib.Path(spec.origin)
sp = site.getsitepackages()[0]
torch_lib = (pathlib.Path(torch.__file__).resolve().parent / "lib")
new_rpath = f"{sp}/nvidia/cuda_runtime/lib:{sp}/nvidia/nvjitlink/lib:{sp}/nvidia/cusparse/lib:{torch_lib}"

print("SO:", so)
print("NEW_RPATH:", new_rpath)

subprocess.check_call(["patchelf", "--set-rpath", new_rpath, str(so)])
subprocess.check_call(["patchelf", "--print-rpath", str(so)])
PY
