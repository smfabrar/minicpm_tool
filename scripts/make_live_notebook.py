"""Generate the Kaggle trial for OpenBMB's native audio Realtime API.

The notebook uses the official gateway, worker, and C++ backend at fixed commits.
Only the documented tool-context extension changes the C++ backend.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
cells: list[dict] = []


def md(source: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})


def code(source: str) -> None:
    compile(source, "<notebook-cell>", "exec")
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": source.splitlines(True)})


md("""# Native MiniCPM live conversation with tools

This is the **human, real-time** test. MiniCPM itself listens and speaks through OpenBMB's audio Realtime API. Whisper Tiny and Granite 350M receive a *copy* of microphone audio to choose external tools; they do not provide the model's voice. A small pinned C++ extension inserts a completed, versioned tool result into the next native audio unit and reports when evaluation finishes. The extension is isolated in `fixtures/realtime-tool-context.patch`.

Use **Internet on**. Run the CPU cells through the echo check before enabling a GPU. The official C++ runtime is expensive to compile; attach a saved runtime dataset next time or save the runtime folder as notebook output. When you enable GPU, Kaggle restarts the kernel: rerun the package and setup cells, then continue at the GPU heading. A live run keeps one native session open while you speak and while it answers. Stop the run before exporting the ZIP.

Official protocol: [audio Realtime API](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/audio.md). The browser interface here is Gradio because Kaggle cannot expose the official gateway page directly; the model-facing WebSocket protocol and backend are official.
""")

code("""# 1 — Select a tested adapter release. Run again after a Kaggle kernel restart.
SOURCE_REF = 'v0.1.24'
DEMO_PIN = '47709a9210dfd71afa76c058e017fc8c4db5c8d2'
OMNI_PIN = '873056743b74e1a4ce5dcf7290e2298428e214db'
print('Adapter:', SOURCE_REF, 'Demo:', DEMO_PIN[:12], 'C++:', OMNI_PIN[:12])
""")

code("""# 2 — Fetch the adapter and make this kernel import exactly the selected tag.
from pathlib import Path
import importlib, subprocess, sys
ROOT = Path('/kaggle/working/minicpm_tool')
if not ROOT.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', SOURCE_REF,
                    'https://github.com/smfabrar/minicpm_tool.git', str(ROOT)], check=True)
else:
    subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', 'tag', SOURCE_REF], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', SOURCE_REF], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT)], check=True)
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
for name in list(sys.modules):
    if name == 'duplex_tools' or name.startswith('duplex_tools.'):
        del sys.modules[name]
importlib.invalidate_caches()
import duplex_tools
assert Path(duplex_tools.__file__).resolve().is_relative_to(ROOT.resolve())
print('Adapter commit:', subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
""")

code("""# 3 — CPU-only behavior checks; no models loaded.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], check=True)
""")

md("""## CPU-only browser transport check

This echo page tests Kaggle's Gradio share link, microphone stream, and speaker playback without paying for GPU time. Use headphones, speak for several seconds, and verify that sound returns while the microphone remains active. A 504 here is a transport problem; resolve it before loading models. Close the echo page before continuing.
""")

code("""# 4 — CPU-only echo check. Close it in the next cell when finished.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio>=5,<7',
                'numpy', 'soundfile', 'websockets>=16'], check=True)
import secrets
from duplex_tools.live_duplex import make_transport_smoke_ui
SMOKE_DIR = Path('/kaggle/working/live_transport_smoke')
smoke_app = make_transport_smoke_ui(SMOKE_DIR)
smoke_app.queue(default_concurrency_limit=4)
smoke_password = secrets.token_urlsafe(12)
smoke_app.launch(share=True, auth=('tester', smoke_password),
                 inline=False, prevent_thread_lock=True)
print('Username: tester')
print('Password:', smoke_password)
print('Echo recordings:', SMOKE_DIR)
""")

code("""# 5 — Close the echo page. CPU preparation ends here.
smoke_app.close()
print('Enable T4 GPU in Kaggle Settings now. Rerun cells 1–2 after the restart.')
""")

md("""## GPU phase: model and official services

Attach a GGUF model dataset and a previously saved `minicpm_official_runtime_sm75` dataset when available. The first build uses CPU compilation inside the GPU session because Kaggle's CUDA toolkit can depend on the accelerator image; subsequent sessions restore verified binaries and skip the build. The runtime bundle has a manifest, source pin, patch hash, and file hashes. `GGML_CUDA_NO_VMM=ON` avoids Kaggle's missing unversioned CUDA driver link while keeping GPU kernels enabled.

GPU 0 runs the native MiniCPM backend. Whisper Tiny and Granite 350M use GPU 1 when available. The official gateway and worker are lightweight Python services. The vision GGUF is present because the official native session checks the complete model layout, although this trial sends audio only.
""")

code("""# 6 — GPU packages and accelerator placement.
import torch
assert torch.cuda.is_available(), 'Enable GPU in Kaggle Settings.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers>=4.54,<5', 'accelerate', 'huggingface_hub',
                'gradio>=5,<7', 'soundfile', 'faster-whisper', 'websockets>=16',
                'fastapi>=0.128', 'uvicorn>=0.40', 'httpx>=0.28',
                'pydantic>=2.11', 'python-multipart', 'markdown>=3.6',
                'Pygments>=2.18', 'PyYAML>=6', 'librosa>=0.10.2'], check=True)
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

code("""# 7 — Reuse an attached GGUF dataset, or download the exact official layout.
from huggingface_hub import snapshot_download
ATTACHED_MODEL_DIR = None  # e.g. '/kaggle/input/my-gguf/MiniCPM-o-4_5-gguf'
MODEL_DIR = Path(ATTACHED_MODEL_DIR) if ATTACHED_MODEL_DIR else Path('/kaggle/working/models/MiniCPM-o-4_5-gguf')
required = [
    'MiniCPM-o-4_5-Q4_K_M.gguf',
    'vision/MiniCPM-o-4_5-vision-F16.gguf',
    'audio/MiniCPM-o-4_5-audio-F16.gguf',
    'tts/MiniCPM-o-4_5-tts-F16.gguf',
    'tts/MiniCPM-o-4_5-projector-F16.gguf',
    'token2wav-gguf/encoder.gguf',
    'token2wav-gguf/flow_matching.gguf',
    'token2wav-gguf/flow_extra.gguf',
    'token2wav-gguf/hifigan2.gguf',
    'token2wav-gguf/prompt_cache.gguf',
]
if not all((MODEL_DIR / name).is_file() for name in required):
    assert not ATTACHED_MODEL_DIR, 'Attached model dataset is incomplete.'
    snapshot_download(repo_id='openbmb/MiniCPM-o-4_5-gguf',
                      allow_patterns=required, local_dir=str(MODEL_DIR))
missing = [name for name in required if not (MODEL_DIR / name).is_file()]
assert not missing, f'Missing model modules: {missing}'
print('GGUF layout:', MODEL_DIR)
""")

code("""# 8 — Clone the exact official Demo gateway/worker source; these use no GPU.
DEMO = Path('/kaggle/working/MiniCPM-o-Demo')
if not DEMO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/OpenBMB/MiniCPM-o-Demo.git', str(DEMO)], check=True)
subprocess.run(['git', '-C', str(DEMO), 'fetch', '--depth', '1', 'origin', DEMO_PIN], check=True)
subprocess.run(['git', '-C', str(DEMO), 'checkout', '--detach', DEMO_PIN], check=True)
assert subprocess.check_output(['git', '-C', str(DEMO), 'rev-parse', 'HEAD'], text=True).strip() == DEMO_PIN
REF_AUDIO = DEMO / 'assets/ref_audio/ref_en_dlc_1.wav'
assert REF_AUDIO.is_file()
print('Official Demo:', DEMO_PIN, 'reference voice:', REF_AUDIO)
""")

code("""# 9 — Restore the pinned T4 build, or compile it once. A second run reuses build objects.
from duplex_tools.runtime_bundle import create_runtime_bundle, verify_runtime_bundle, runtime_environment
import hashlib, shutil
ATTACHED_RUNTIME_DIR = None  # e.g. '/kaggle/input/my-runtime/minicpm_official_runtime_sm75'
UPSTREAM = Path('/kaggle/working/llama.cpp-omni-official')
RUNTIME_BUNDLE = Path('/kaggle/working/minicpm_official_runtime_sm75')
patch = ROOT / 'fixtures/realtime-tool-context.patch'
if ATTACHED_RUNTIME_DIR:
    if not RUNTIME_BUNDLE.exists():
        shutil.copytree(ATTACHED_RUNTIME_DIR, RUNTIME_BUNDLE, symlinks=True)
    SERVER_BIN = verify_runtime_bundle(RUNTIME_BUNDLE, expected_pin=OMNI_PIN, expected_patch=patch)
    print('Verified saved runtime. Compilation skipped:', SERVER_BIN)
else:
    if not UPSTREAM.exists():
        subprocess.run(['git', 'clone', 'https://github.com/tc-mb/llama.cpp-omni.git', str(UPSTREAM)], check=True)
    subprocess.run(['git', '-C', str(UPSTREAM), 'fetch', '--depth', '1', 'origin', OMNI_PIN], check=True)
    subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', '--detach', OMNI_PIN], check=True)
    already = subprocess.run(['git', '-C', str(UPSTREAM), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
    if already.returncode != 0:
        subprocess.run(['git', '-C', str(UPSTREAM), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(UPSTREAM), 'apply', str(patch)], check=True)
    subprocess.run(['cmake', '-S', str(UPSTREAM), '-B', str(UPSTREAM / 'build'),
                    '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', '-DGGML_CUDA_NO_VMM=ON',
                    '-DGGML_NATIVE=OFF', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DLLAMA_CURL=OFF', '-DLLAMA_OPENSSL=OFF',
                    '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF'], check=True)
    subprocess.run(['cmake', '--build', str(UPSTREAM / 'build'),
                    '--target', 'llama-omni-server', '-j', '2'], check=True)
    SERVER_BIN = create_runtime_bundle(
        UPSTREAM / 'build/bin', RUNTIME_BUNDLE, patch=patch, source_pin=OMNI_PIN,
        source_ref=SOURCE_REF,
        build_options=['GGML_CUDA=ON', 'GGML_CUDA_NO_VMM=ON', 'GGML_NATIVE=OFF',
                       'CMAKE_CUDA_ARCHITECTURES=75', 'LLAMA_OPENSSL=OFF'])
    print('Save this folder as Kaggle notebook output and attach it next session:', RUNTIME_BUNDLE)
SERVER_ENV = runtime_environment(RUNTIME_BUNDLE)
print('Patch SHA256:', hashlib.sha256(patch.read_bytes()).hexdigest())
""")

code("""# 10 — Start the official backend, worker, and gateway on three local ports.
import json, os, secrets, socket, time, urllib.request
from datetime import datetime, timezone

def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]

def wait_health(process, url, log_path, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{url} exited early; last log:\\n{log_path.read_text()[-4000:]}')
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'{url} not ready; last log:\\n{log_path.read_text()[-4000:]}')

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(3)
OUTPUT = Path('/kaggle/working/duplex_voice_runs') / RUN_ID
OUTPUT.mkdir(parents=True, exist_ok=False)
ports = []
while len(ports) < 4:
    candidate = free_port()
    if candidate not in ports:
        ports.append(candidate)
BACKEND_PORT, WORKER_PORT, GATEWAY_PORT, INTERNAL_PORT = ports
BACKEND_URL = f'http://127.0.0.1:{BACKEND_PORT}'
WORKER_URL = f'http://127.0.0.1:{WORKER_PORT}'
GATEWAY_URL = f'http://127.0.0.1:{GATEWAY_PORT}'
log_handles = []
def launch(name, argv, cwd, env=None):
    path = OUTPUT / f'{name}.log'
    handle = path.open('w')
    log_handles.append(handle)
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, path

backend, backend_log = launch('minicpm_backend', [str(SERVER_BIN), '-m',
    str(MODEL_DIR / 'MiniCPM-o-4_5-Q4_K_M.gguf'), '-ngl', '99',
    '--host', '127.0.0.1', '--port', str(BACKEND_PORT)], UPSTREAM if UPSTREAM.exists() else DEMO, SERVER_ENV)
wait_health(backend, BACKEND_URL, backend_log)
gateway, gateway_log = launch('official_gateway', [sys.executable, 'gateway.py',
    '--host', '127.0.0.1', '--port', str(GATEWAY_PORT),
    '--internal-port', str(INTERNAL_PORT), '--http'], DEMO)
wait_health(gateway, GATEWAY_URL, gateway_log, 120)
worker, worker_log = launch('official_worker', [sys.executable, 'worker.py',
    '--host', '127.0.0.1', '--port', str(WORKER_PORT), '--gpu-id', '0',
    '--backend-server-url', BACKEND_URL], DEMO)
wait_health(worker, WORKER_URL, worker_log, 120)
request = urllib.request.Request(f'http://127.0.0.1:{INTERNAL_PORT}/internal/workers/kaggle-t4',
    data=json.dumps({'endpoint': f'127.0.0.1:{WORKER_PORT}', 'gpu_group': 'kaggle-t4'}).encode(),
    headers={'content-type': 'application/json'}, method='PUT')
with urllib.request.urlopen(request, timeout=10) as response:
    assert response.status == 200, response.read()
manifest = {'run_id': RUN_ID, 'started_at': datetime.now(timezone.utc).isoformat(),
    'adapter_tag': SOURCE_REF, 'adapter_commit': subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    'official_demo_commit': DEMO_PIN, 'official_cpp_commit': OMNI_PIN,
    'tool_extension_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
    'model_dir': str(MODEL_DIR), 'gpu_names': [torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())],
    'ports': {'backend': BACKEND_PORT, 'worker': WORKER_PORT, 'gateway': GATEWAY_PORT}}
(OUTPUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print('Official realtime gateway ready:', GATEWAY_URL, 'output:', OUTPUT)
""")

code("""# 11 — Load the independent tool sidecar on GPU 1; MiniCPM handles all speech itself.
import asyncio
from faster_whisper import WhisperModel
from duplex_tools.caller import GraniteToolCaller, TransformersGraniteGenerator
from duplex_tools.controller import ContextController, JsonlEventLog
from duplex_tools.conversation import ConversationRouter
from duplex_tools.official_realtime import OfficialRealtimeExperiment
from duplex_tools.live_duplex import RestartableLiveExperiment, make_live_gradio_ui
from duplex_tools.tools import RoomLookup, SafeCalculator, DocumentSearch

# Five seconds makes the correction scenario observable. It is an experimental
# condition, not a measurement of actual room-lookup latency.
ROOM_LOOKUP_DELAY_S = 5.0
base_room = RoomLookup({'robotics seminar': 'The robotics seminar is in room B742.',
                        'vision seminar': 'The vision seminar is in room C314.'},
                       aliases={'robotic seminar': 'robotics seminar'})
async def delayed_room_lookup(arguments):
    await asyncio.sleep(ROOM_LOOKUP_DELAY_S)
    return await base_room(arguments)

tools = {'room_lookup': delayed_room_lookup, 'calculator': SafeCalculator(),
         'document_search': DocumentSearch({
             'Thesis deadlines': 'The draft is due on October 15; the final copy is due on November 20.',
             'Lab access': 'The lab is open Monday through Friday from 9 to 17.'})}
AUX_GPU = 1 if torch.cuda.device_count() > 1 else 0
granite_backend = TransformersGraniteGenerator(device=f'cuda:{AUX_GPU}')
recognizer = WhisperModel('tiny.en', device='cuda', device_index=AUX_GPU,
                          compute_type='float16')
manifest['granite_revision'] = granite_backend.model_revision
manifest['granite_snapshot'] = granite_backend.model_source
manifest['whisper_model'] = 'tiny.en'
manifest['tool_sidecar_gpu'] = AUX_GPU
(OUTPUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
def new_live_session():
    controller = ContextController(tools, tool_timeout_s=12,
                                   log=JsonlEventLog(OUTPUT / 'controller.jsonl'))
    router = ConversationRouter(GraniteToolCaller(granite_backend), controller)
    return OfficialRealtimeExperiment(
        session=None, router=router, recognizer=recognizer, output_root=OUTPUT,
        gateway_url=f'ws://127.0.0.1:{GATEWAY_PORT}/v1/realtime?mode=audio',
        reference_audio=REF_AUDIO,
        system_prompt='You are a helpful voice assistant. Listen to the user and answer naturally. '
                      'If a verified tool result appears, use it in your reply; do not invent a room or deadline.',
        max_backlog_s=12, final_silence_units=8)

live_experiment = RestartableLiveExperiment(new_live_session)
print('Native speech: MiniCPM. Tool sidecar: Whisper Tiny + Granite on', f'cuda:{AUX_GPU}')
""")

code("""# 12 — Human live trial. This URL is password-protected; keep this cell running.
import secrets
password = secrets.token_urlsafe(12)
app = make_live_gradio_ui(live_experiment)
app.queue(default_concurrency_limit=4)
app.launch(share=True, auth=('tester', password), inline=False,
           prevent_thread_lock=True, allowed_paths=[str(OUTPUT)])
print('Username: tester')
print('Password:', password)
print('Run output:', OUTPUT)
""")

md("""## Speak, inspect, and save

1. Press **Start live session**. Initialization runs in the background, so the share page stays responsive. When status says **running**, click the **record button inside the microphone panel**. The browser must start its own microphone. Keep it recording while you speak and while MiniCPM answers.
2. Say “Where is the robotics seminar?” Then, before the delayed lookup returns, say “Actually, the vision seminar.” The current answer should name **C314**, not B742. Check the tool trace and listen to the spoken answer.
3. While MiniCPM is talking, ask “What is 17 times 23?” This checks whether capture continues during output. If there is no audible overlap, record that honestly.
4. You may stop the microphone recording and click it again to resume within the same native session. To end the run, stop the microphone in its panel, then press **Stop and finish**. Wait for `complete` or `failed`. Record the exact words heard and whether you spoke during the assistant's audio. A later **Start live session** creates a fresh native and tool session without rebuilding the server.
5. Run the export cell below and download the ZIP. A Gradio 504 does not erase the local logs; inspect `live_status.json`, `live_events.jsonl`, and service logs before retrying.

The run directory includes the complete microphone recording, each native input unit, committed utterances, tool events, returned audio chunks, human verdicts, gateway/worker/backend logs, code revisions, and a summary. The session is bounded by the official API's 600-second limit.

If live speech sounds broken up, listen to `speaker_combined.wav` from cell 13. That file joins only MiniCPM's actual output. If it sounds smooth while the live page stutters, the browser delivery is implicated. The timing report below measures gaps between native chunks and adapter dispatch; it cannot tell exactly when browser playback became audible.
""")

code("""# 13 — Inspect a finished run without touching the active model.
from IPython.display import display, FileLink
state = live_experiment.snapshot()
print(json.dumps(state, indent=2))
if state.get('session_id'):
    session_dir = OUTPUT / 'live_sessions' / state['session_id']
    summary_path = session_dir / 'live_summary.json'
    if summary_path.is_file():
        print(summary_path.read_text())
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'analyze_live_export.py'),
                        str(session_dir)], check=True)
        if (session_dir / 'speaker_combined.wav').is_file():
            display(FileLink(str(session_dir / 'speaker_combined.wav')))
""")

code("""# 14 — Save the entire experiment as one downloadable archive.
import zipfile
state = live_experiment.snapshot()
assert state.get('status') in {'complete', 'failed'}, 'Stop and finish the live session first.'
app.close()
live_experiment.close()
for process in (worker, gateway, backend):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
for handle in log_handles:
    handle.close()
ARCHIVE = Path('/kaggle/working') / f'{RUN_ID}_official_live_duplex.zip'
with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT.parent))
print('Download run:', ARCHIVE, 'bytes:', ARCHIVE.stat().st_size)
print('Save runtime for next session:', RUNTIME_BUNDLE)
display(FileLink(str(ARCHIVE)))
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "notebooks/kaggle_live_duplex.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
