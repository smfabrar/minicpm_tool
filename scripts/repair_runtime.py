"""Repair an existing Kaggle build incrementally; preserve CUDA object files."""

import argparse
from pathlib import Path
import subprocess

PIN = "64d092c60db4b4ee45768476bd752f03fdcc98ea"


def repair_sse(upstream: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise RuntimeError(f"Expected upstream {PIN}, got {revision}")
    source = upstream / "tools/server/server-omni.cpp"
    text = source.read_text()
    replacements = (
        ('            [&](size_t, httplib::DataSink & sink) -> bool {',
         '            [&, debug_dir, round_idx](size_t, httplib::DataSink & sink) -> bool {'),
        ('                sink.write(ev_done.data(), ev_done.size());',
         '                if (!sink.write(ev_done.data(), ev_done.size())) return false;\n'
         '                sink.done();'),
    )
    for before, after in replacements:
        if after in text:
            continue
        if text.count(before) != 1:
            raise RuntimeError("Unexpected SSE source; refusing an ambiguous edit")
        text = text.replace(before, after, 1)
    source.write_text(text)
    print("SSE callback captures and HTTP response completion repaired.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/kaggle/working/llama.cpp-omni"))
    parser.add_argument("--patch-only", action="store_true")
    args = parser.parse_args()
    repair_sse(args.upstream)
    if not args.patch_only:
        build = args.upstream / "build"
        if not (build / "CMakeCache.txt").is_file():
            raise RuntimeError("Existing build cache is missing; use the notebook build cell")
        subprocess.run(["cmake", "-S", str(args.upstream), "-B", str(build),
                        "-DLLAMA_OPENSSL=OFF", "-DGGML_CUDA_NO_VMM=ON"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "llama-omni-server",
                        "-j", "2"], check=True)
        print("Repaired executable:", build / "bin/llama-omni-server")
