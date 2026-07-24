from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

CANDIDATE_MODULES = (
    "mcflashinfer",
    "McFlashInfer",
    "mc_flashinfer",
    "flashinfer",
    "mcoplib",
    "mcop",
)
LIBRARY_ROOTS = (
    Path("/opt/maca/lib"),
    Path("/opt/maca/lib64"),
    Path("/usr/local/lib"),
)


def _public_api(module: ModuleType, limit: int = 100) -> list[str]:
    return sorted(name for name in dir(module) if not name.startswith("_"))[:limit]


def inventory_modules(
    names: Iterable[str],
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name in names:
        try:
            spec = find_spec(name)
        except BaseException as exc:
            results.append({"name": name, "found": False, "probe_error": repr(exc)})
            continue
        if spec is None:
            results.append({"name": name, "found": False})
            continue
        record: dict[str, object] = {
            "name": name,
            "found": True,
            "origin": getattr(spec, "origin", None),
        }
        try:
            module = import_module(name)
            record.update(
                {
                    "import_ok": True,
                    "version": getattr(module, "__version__", None),
                    "public_api": _public_api(module),
                }
            )
        except BaseException as exc:
            record.update({"import_ok": False, "import_error": repr(exc)})
        results.append(record)
    return results


def inventory_distributions() -> list[dict[str, str]]:
    keywords = ("flashinfer", "mcop", "maca", "metax")
    records = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if any(keyword in name.lower() for keyword in keywords):
            records.append({"name": name, "version": distribution.version})
    return sorted(records, key=lambda record: record["name"].lower())


def inventory_libraries(roots: Iterable[Path]) -> list[str]:
    matches: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*flash*", "*mcop*"):
            matches.update(str(path) for path in root.glob(pattern))
    return sorted(matches)


def build_inventory() -> dict[str, object]:
    import torch

    modules = inventory_modules(CANDIDATE_MODULES)
    usable_modules = [
        record["name"]
        for record in modules
        if record.get("found") and record.get("import_ok")
    ]
    device_count = torch.cuda.device_count()
    return {
        "torch_version": torch.__version__,
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "candidate_modules": modules,
        "matching_distributions": inventory_distributions(),
        "matching_libraries": inventory_libraries(LIBRARY_ROOTS),
        "importable_candidate_modules": usable_modules,
        "integration_readiness": "requires_runtime_probe",
        "correctness_backend": "torch_native",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory MetaX fused-attention packages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False))


if __name__ == "__main__":
    main()
