"""Create and verify a reusable Kaggle bundle for the patched native runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


SERVER_NAME = "llama-omni-server"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def _python_versions() -> dict[str, str]:
    result = {}
    for package in ("torch", "transformers", "accelerate", "gradio", "soundfile", "faster-whisper", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _cuda_arch() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}{minor}"
    except ImportError:
        pass
    return None


def _runtime_env(bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(bin_dir) + ((":" + existing) if existing else "")
    return env


def _linked_libraries(server: Path) -> str:
    result = subprocess.run(
        ["ldd", str(server)], capture_output=True, text=True, check=False,
        env=_runtime_env(server.parent),
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "not found" in output:
        raise RuntimeError(f"runtime dependency check failed:\n{output}")
    return output


def create_runtime_bundle(
    build_bin: Path,
    output_dir: Path,
    *,
    patch: Path,
    source_pin: str,
    source_ref: str,
    build_options: list[str],
) -> Path:
    """Copy the executable and project libraries and record host dependencies."""
    server = build_bin / SERVER_NAME
    if not server.is_file():
        raise FileNotFoundError(server)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(build_bin, temporary / "bin", symlinks=True)
    shutil.copy2(patch, temporary / "context-injection.patch")
    bundled_server = temporary / "bin" / SERVER_NAME
    bundled_server.chmod(bundled_server.stat().st_mode | 0o111)
    linked = _linked_libraries(bundled_server)
    files = {
        str(path.relative_to(temporary)): sha256_file(path)
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_pin": source_pin,
        "source_ref": source_ref,
        "patch_sha256": sha256_file(patch),
        "build_options": build_options,
        "cuda_arch": _cuda_arch(),
        "platform": platform.platform(),
        "toolchain": {
            "cmake": _command_output(["cmake", "--version"]).splitlines()[0],
            "cxx": _command_output(["c++", "--version"]).splitlines()[0],
            "nvcc": _command_output(["nvcc", "--version"]),
            "glibc": _command_output(["ldd", "--version"]).splitlines()[0],
        },
        "python_packages": _python_versions(),
        "linked_libraries": linked,
        "files": files,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(output_dir, ignore_errors=True)
    temporary.rename(output_dir)
    return verify_runtime_bundle(output_dir, expected_pin=source_pin, expected_patch=patch)


def verify_runtime_bundle(
    bundle_dir: Path,
    *,
    expected_pin: str,
    expected_patch: Path,
) -> Path:
    """Verify provenance, contents, GPU architecture, and dynamic dependencies."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_pin") != expected_pin:
        raise ValueError(f"runtime source pin differs: {manifest.get('source_pin')}")
    expected_patch_hash = sha256_file(expected_patch)
    if manifest.get("patch_sha256") != expected_patch_hash:
        raise ValueError("runtime was built with a different context-injection patch")
    for relative, expected_hash in manifest.get("files", {}).items():
        path = bundle_dir / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"runtime artifact checksum failed: {relative}")
    built_arch = manifest.get("cuda_arch")
    current_arch = _cuda_arch()
    if built_arch and current_arch and built_arch != current_arch:
        raise ValueError(f"runtime requires CUDA architecture sm_{built_arch}; current GPU is sm_{current_arch}")
    server = bundle_dir / "bin" / SERVER_NAME
    server.chmod(server.stat().st_mode | 0o111)
    _linked_libraries(server)
    return server


def runtime_environment(bundle_dir: Path) -> dict[str, str]:
    """Environment required when launching a restored dynamic runtime."""
    return _runtime_env(bundle_dir / "bin")
