"""Regenerate the checked-in Kaggle notebook without a notebook dependency."""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)})


md("""# Duplex Tools on Kaggle: CPU preparation and human voice gate

This notebook pulls the adapter package from `smfabrar/minicpm_tool`. It has two phases.

| Phase | Run these cells | Accelerator | Purpose |
|---|---|---|---|
| A | 1–5, ending after the Granite caller probe | **None** | Install the package, run controller tests, inspect parser behavior, and call the real Granite 350M model on a few transcripts. |
| B | Rerun setup cell 1, then cells 6 onward | **GPU** | Build the patched MiniCPM runtime, attach or download GGUF modules, and speak to the persistent session. |

Enable **Internet** in Kaggle Settings. Adding a GPU restarts the notebook runtime, so Python objects from phase A disappear. Reopen this notebook after the restart, rerun the first setup cell, and continue at the GPU section. Skip the CPU benchmark on the GPU clock. An attached Kaggle Dataset containing the GGUF folder saves download time; the model download cell is a fallback. Do not run the build or download cells during the CPU phase if their output is unlikely to survive your runtime restart.

The voice gate is push-to-talk: record a turn, send it, then listen. It tests a real model-selected tool and a spoken answer in one MiniCPM session. It does not yet demonstrate continuous simultaneous recording and playback. The HTTP API reports context submission; actual KV evaluation remains **unknown** without a native acknowledgement.

Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [IBM Granite model card](https://huggingface.co/ibm-granite/granite-4.0-350m), [Granite tool format](https://github.com/ibm-granite/granite-4.0-language-models/blob/main/Granite%204.0%20Prompt%20engineering%20guide%20v2.md), [MiniCPM GGUF modules](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf).
""")

code("""# 1 — Run once on CPU, and rerun this cell after switching to GPU.
from pathlib import Path
import subprocess, sys

ROOT = Path('/kaggle/working/minicpm_tool')
SOURCE_REF = 'v0.1.5'
if not ROOT.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', SOURCE_REF,
                    'https://github.com/smfabrar/minicpm_tool.git', str(ROOT)], check=True)
else:
    current = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    tagged = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '-q', '--verify', SOURCE_REF],
                            capture_output=True, text=True)
    if tagged.returncode != 0 or tagged.stdout.strip() != current:
        subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', 'tag', SOURCE_REF], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'checkout', SOURCE_REF], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT)], check=True)
# Editable installs add a .pth file for the next interpreter start. Make the
# package visible in this already-running notebook kernel immediately.
source_path = str(ROOT / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)
import importlib
importlib.invalidate_caches()
import duplex_tools
assert Path(duplex_tools.__file__).resolve().is_relative_to(ROOT.resolve())
print('Package:', ROOT)
print('Imported:', duplex_tools.__file__)
print('Revision:', subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
""")

code("""# 2 — CPU only: check controller, correction, cancellation, parser, and adapter behavior.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], check=True)
""")

code("""# 3 — CPU only: inspect the native Granite parser and simulated adapter.
import asyncio
from duplex_tools.caller import parse_granite_output
from duplex_tools.simulated import SimulatedAdapter
from duplex_tools.contracts import AdapterCapabilities

sample = '<tool_call>{"name":"room_lookup","arguments":{"name":"robotics seminar"}}</tool_call>'
print(parse_granite_output(sample))
sim = SimulatedAdapter(AdapterCapabilities(True, False, False, False, True))
print('Simulated capability profile:', sim.capabilities())
""")

code("""# 4 — CPU only: load Granite once and measure staged model routing and arguments.
# Kaggle normally has PyTorch. This installs only the model-side packages.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.54,<5', 'accelerate'], check=True)
import json, time
import importlib
from datetime import datetime, timezone
import duplex_tools.caller as caller_module
importlib.reload(caller_module)
from duplex_tools.caller import GraniteToolCaller, TransformersGraniteGenerator
from duplex_tools.contracts import TranscriptSegment
from duplex_tools.tools import TOOL_SCHEMAS, validate_call

if 'generator' not in globals():
    generator = TransformersGraniteGenerator(device='cpu')
caller = GraniteToolCaller(generator)
cases = json.loads((ROOT / 'fixtures' / 'caller_cases.json').read_text())
rows = []
for case in cases:
    segment = TranscriptSegment(case['id'], 1, datetime.now(timezone.utc), case['text'], True)
    start = time.perf_counter()
    action = await caller.decide(segment, {}, TOOL_SCHEMAS)
    expected_tool = case.get('tool')
    expected_args = case.get('arguments', {})
    try:
        parsed_args = validate_call(action.tool or '', action.arguments) if action.kind == 'call' else {}
    except ValueError:
        parsed_args = {}
    args_match = parsed_args == expected_args or (
        case['id'] == 'calculator' and parsed_args.get('expression', '').replace(' ', '') == expected_args.get('expression', '').replace(' ', '')
    )
    passed = action.kind == case['kind'] and (expected_tool is None or action.tool == expected_tool) and (expected_tool is None or args_match)
    rows.append({'id': case['id'], 'expected': case['kind'], 'actual': action.kind,
                 'tool': action.tool, 'arguments': dict(action.arguments),
                 'passed': passed, 'latency_s': round(time.perf_counter() - start, 2), 'raw': action.raw})
for row in rows:
    print(json.dumps(row, ensure_ascii=False))
print('Fully correct:', sum(r['passed'] for r in rows), '/', len(rows))
print('CPU gate:', 'PASS' if all(r['passed'] for r in rows) else 'FAIL — keep GPU off and review caller errors')
""")

code("""# 5 — CPU only: edit this sentence for a quick human-written transcript probe.
sentence = 'Where is the robotics seminar?'
segment = TranscriptSegment('manual-text', 1, datetime.now(timezone.utc), sentence, True)
proposal = await caller.decide(segment, {}, TOOL_SCHEMAS)
print('Proposal:', proposal)
print('Raw model output:', proposal.raw)
""")

md("""## Stop here and enable the GPU

Proceed only when the CPU caller cell reports **PASS**. In Kaggle Settings select **Accelerator → GPU**. Kaggle restarts the runtime. Rerun **cell 1 only**, then continue below. Phase B needs CUDA for MiniCPM. The 350M caller and tiny speech recognizer stay on CPU so the GPU is reserved for MiniCPM. If the GPU has too little memory, reduce `N_GPU_LAYERS` in the server cell and note the resulting latency.
""")

code("""# 6 — GPU phase: verify the accelerator and install only the voice dependencies.
import torch
assert torch.cuda.is_available(), 'Enable a GPU in Kaggle Settings before continuing.'
print(torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers>=4.54,<5', 'accelerate', 'huggingface_hub',
                'gradio>=5,<7', 'soundfile', 'faster-whisper'], check=True)
""")

code("""# 7 — Prefer an attached Kaggle Dataset with this GGUF folder; otherwise download.
from huggingface_hub import snapshot_download

# Example: Path('/kaggle/input/your-minicpm-gguf/MiniCPM-o-4_5-gguf')
ATTACHED_MODEL_DIR = None
MODEL_DIR = Path(ATTACHED_MODEL_DIR) if ATTACHED_MODEL_DIR else Path('/kaggle/working/models/MiniCPM-o-4_5-gguf')
required = [
    'MiniCPM-o-4_5-Q4_K_M.gguf',
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
    assert ATTACHED_MODEL_DIR is None, 'Attached dataset is missing required files.'
    snapshot_download(repo_id='openbmb/MiniCPM-o-4_5-gguf',
                      allow_patterns=required, local_dir=str(MODEL_DIR))
missing = [name for name in required if not (MODEL_DIR / name).is_file()]
assert not missing, f'Missing GGUF modules: {missing}'
print('Model directory:', MODEL_DIR)
""")

code("""# 8 — Build the pinned upstream runtime with the repository's tested patch.
UPSTREAM = Path('/kaggle/working/llama.cpp-omni')
PIN = '64d092c60db4b4ee45768476bd752f03fdcc98ea'
if not UPSTREAM.exists():
    subprocess.run(['git', 'clone', 'https://github.com/tc-mb/llama.cpp-omni.git', str(UPSTREAM)], check=True)
subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', PIN], check=True)
patch = ROOT / 'fixtures' / 'context-injection.patch'
reverse = subprocess.run(['git', '-C', str(UPSTREAM), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
if reverse.returncode != 0:
    subprocess.run(['git', '-C', str(UPSTREAM), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(UPSTREAM), 'apply', str(patch)], check=True)
subprocess.run(['cmake', '-S', str(UPSTREAM), '-B', str(UPSTREAM / 'build'),
                '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON'], check=True)
subprocess.run(['cmake', '--build', str(UPSTREAM / 'build'), '--target', 'llama-omni-server', '-j', '4'], check=True)
SERVER_BIN = UPSTREAM / 'build' / 'bin' / 'llama-omni-server'
assert SERVER_BIN.is_file()
""")

code("""# 9 — Start one persistent local MiniCPM server and wait for HTTP readiness.
import subprocess, time, urllib.request

N_GPU_LAYERS = 99
BASE_URL = 'http://127.0.0.1:9060'
SERVER_LOG = Path('/kaggle/working/minicpm_server.log')
if 'server_process' not in globals() or server_process.poll() is not None:
    server_log_handle = SERVER_LOG.open('w')
    server_process = subprocess.Popen([
        str(SERVER_BIN), '--host', '127.0.0.1', '--port', '9060',
        '--model', str(MODEL_DIR / 'MiniCPM-o-4_5-Q4_K_M.gguf'),
        '-ngl', str(N_GPU_LAYERS), '--ctx-size', '8192',
    ], stdout=server_log_handle, stderr=subprocess.STDOUT)
for _ in range(120):
    if server_process.poll() is not None:
        raise RuntimeError(SERVER_LOG.read_text()[-4000:])
    try:
        with urllib.request.urlopen(BASE_URL + '/health', timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(2)
else:
    raise TimeoutError('MiniCPM server did not become ready; inspect ' + str(SERVER_LOG))
print('MiniCPM server is ready')
""")

code("""# 10 — Load CPU caller and recognizer once, then initialize MiniCPM once.
from faster_whisper import WhisperModel
from duplex_tools.caller import GraniteToolCaller, TransformersGraniteGenerator
from duplex_tools.controller import ContextController, JsonlEventLog
from duplex_tools.conversation import ConversationRouter
from duplex_tools.minicpm_client import MiniCPMStreamSession, OmniHttpClient
from duplex_tools.tools import RoomLookup, SafeCalculator, DocumentSearch
from duplex_tools.voice_demo import VoiceDemo, make_gradio_ui

OUTPUT = Path('/kaggle/working/duplex_voice_output')
OUTPUT.mkdir(exist_ok=True)
tools = {
    'room_lookup': RoomLookup({'robotics seminar': 'The robotics seminar is in room B742.',
                               'vision seminar': 'The vision seminar is in room C314.'}),
    'calculator': SafeCalculator(),
    'document_search': DocumentSearch({'Thesis deadlines': 'The draft is due on October 15; the final copy is due on November 20.',
                                       'Lab access': 'The lab is open Monday through Friday from 9 to 17.'}),
}
controller = ContextController(tools, log=JsonlEventLog(OUTPUT / 'controller.jsonl'))
router = ConversationRouter(GraniteToolCaller(TransformersGraniteGenerator(device='cpu')), controller)
recognizer = WhisperModel('tiny.en', device='cpu', compute_type='int8')
session = MiniCPMStreamSession(OmniHttpClient(BASE_URL, timeout_s=600), controller)
print(await session.initialize(output_dir=str(OUTPUT), model_dir=str(MODEL_DIR),
                               tts_bin_dir=str(MODEL_DIR / 'tts'), token2wav_device='gpu:0'))
voice_demo = VoiceDemo(session, router, OUTPUT, recognizer)
print('Adapter capabilities:', session.capabilities())
""")

code("""# 11 — Human voice gate. The Gradio share URL is public; this one has a random password.
import secrets
password = secrets.token_urlsafe(12)
app = make_gradio_ui(voice_demo)
app.launch(share=True, auth=('tester', password), inline=False, prevent_thread_lock=True)
print('Gradio username: tester')
print('Gradio password:', password)
""")

md("""## Human test procedure

1. In the Gradio page, record **“Where is the robotics seminar?”** Send it. Check the user transcript, that the proposal names `room_lookup`, the submitted event in the trace, and listen for **B742**.
2. Ask **“What is 17 times 23?”** Listen for **391**. Inspect the exact expression chosen; an incorrect argument is a caller failure even if the speech sounds plausible.
3. Ask for the **thesis deadline**, then try an ordinary greeting. The greeting should make no tool call.
4. Repeat with your own paraphrases. Save a verdict after listening to every answer. Logs are in `/kaggle/working/duplex_voice_output/` as `human_trials.jsonl`, `human_verdicts.jsonl`, and `controller.jsonl`.

The model input counter and session stay live across all turns. `evaluation: unknown` means the HTTP prefill accepted the context but did not confirm token evaluation. Generated text and the audio file are recorded separately; the listening verdict is the spoken-answer ground truth. A missing or incorrect answer is a failed trial, not something to infer away from the tool trace.

The microphone gate records complete turns. Correction or cancellation during an actively running tool, overlapping speech, and live incremental playback remain the next human gates after this first extraction.
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_duplex_tools.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
