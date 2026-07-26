# Contributing

Contributions are welcome when they keep the project correctness-first and
evidence-driven.

## Development setup

Use Python 3.10 or newer. On a normal CPU development host:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pip install pytest msgpack pyzmq fastapi uvicorn prompt_toolkit \
  "transformers>=4.56.0,<=4.57.3"
```

Install a CPU PyTorch wheel for host-independent tests. On MetaX hardware,
keep the vendor PyTorch wheel and never replace it with a public PyPI build.

## Validation expectations

- Run the portable tests listed in `.github/workflows/ci.yml`.
- Add a focused regression test for every behavior change.
- For hardware claims, preserve the exact command, bounded workload, runtime
  versions, result, and limitation in `docs/metax_port/`.
- Do not claim performance leadership from a correctness gate.
- Do not claim TP2+ real-model support without a run on the required number of
  visible accelerators.

## Pull requests

Keep changes scoped. Describe the observed failure, why the framework change
is needed, and the checks used to validate it. Do not include model weights,
runtime packages, private logs, credentials, internal URLs, or user-specific
storage paths.
