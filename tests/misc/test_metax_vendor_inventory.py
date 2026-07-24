from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "metax" / "vendor_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("vendor_inventory", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inventory_modules_distinguishes_missing_and_importable() -> None:
    module = ModuleType("installed")
    module.__version__ = "1.2.3"
    module.fused_attention = lambda: None

    def find_spec(name: str):
        return SimpleNamespace(origin="/tmp/installed.py") if name == "installed" else None

    records = MODULE.inventory_modules(
        ["installed", "missing"],
        find_spec=find_spec,
        import_module=lambda _name: module,
    )

    assert records[0]["found"] is True
    assert records[0]["import_ok"] is True
    assert records[0]["version"] == "1.2.3"
    assert "fused_attention" in records[0]["public_api"]
    assert records[1] == {"name": "missing", "found": False}
