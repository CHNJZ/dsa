"""Record runtime provenance and check installed TLA files against wheel RECORD."""
import base64
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path
import sys

import torch
import torch_npu
import triton
import triton._C.libtriton as native
import triton.language.extra.tla as tla


def main():
    dist = importlib.metadata.distribution("triton_ascend")
    checked, mismatched, missing_hash = [], [], []
    for item in dist.files or []:
        name = str(item)
        if not ("/language/extra/tla/" in name or name.endswith("/libtriton.so")
                or name.endswith("/compiler/code_generator.py")):
            continue
        if item.hash is None:
            missing_hash.append(name)
            continue
        path = Path(dist.locate_file(item))
        actual = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
        checked.append(name)
        if item.hash.mode != "sha256" or actual != item.hash.value:
            mismatched.append(name)
    result = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "distribution": dist.version,
        "triton_import": triton.__file__, "native_import": native.__file__,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "api": {name: hasattr(tla, name) for name in
                ("scalar_load", "byte_view", "load_gm_to_ub_strided", "subview")},
        "mmad_signature": str(inspect.signature(tla.cube.mmad)),
        "simd_api": {name: hasattr(tla.simd, name) for name in ("vexpdif", "vmuls")},
        "wheel_record_checked": checked, "wheel_record_mismatched": mismatched,
        "wheel_record_missing_hash": missing_hash,
        "record_scope_note": "Only TLA sources, code_generator.py and libtriton.so are checked; RECORD itself is not an independent signed upstream artifact.",
    }
    text = json.dumps(result, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
