"""Compile the actual patched SSE callback with a fake decoder; no model/GPU.

Uses the pinned vendored HTTP header and real TCP requests. Verifies each
response closes and two sequential requests produce exactly two decodes.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
pin = "64d092c60db4b4ee45768476bd752f03fdcc98ea"

with tempfile.TemporaryDirectory(prefix="duplex-sse-") as directory:
    work = Path(directory)
    for relative in ("tools/omni/omni.cpp", "tools/omni/omni.h", "tools/server/server-omni.cpp"):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(
            ["git", "-C", str(args.upstream), "show", f"{pin}:{relative}"]))
    subprocess.run(["git", "apply", "--check", str(root / "fixtures/context-injection.patch")],
                   cwd=work, check=True)
    subprocess.run(["git", "apply", str(root / "fixtures/context-injection.patch")],
                   cwd=work, check=True)
    source = (work / "tools/server/server-omni.cpp").read_text()
    callback = source.split('        // SSE streaming\n', 1)[1].split(
        '    // POST /v1/stream/update_session_config', 1)[0]
    # callback includes the endpoint's closing `});`.
    harness = r'''
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
using json = nlohmann::json;
struct FakeContext {
    std::mutex text_mtx;
    std::condition_variable text_cv;
    std::deque<std::string> text_queue;
    bool text_done_flag = false;
    bool text_streaming = false;
};
std::atomic<int> calls{0};
bool stream_decode(FakeContext *ctx, const std::string &directory, int round) {
    ++calls;
    std::lock_guard<std::mutex> lock(ctx->text_mtx);
    ctx->text_queue.push_back(directory + std::to_string(round));
    ctx->text_done_flag = true;
    ctx->text_cv.notify_all();
    return true;
}
bool server_sent_event(httplib::DataSink &sink, const json &ev) {
    auto message = "data: " + ev.dump() + "\n\n";
    return sink.write(message.data(), message.size());
}
int main() {
    FakeContext context;
    struct State { FakeContext *octx; std::mutex octx_mutex; } state{&context};
    httplib::Server svr;
    svr.Post("/decode", [&](const httplib::Request &, httplib::Response &res) {
        std::string debug_dir = "captured-value-";
        int round_idx = 7;
''' + callback + r'''
    const int port = svr.bind_to_any_port("127.0.0.1");
    if (port < 0) return 2;
    std::thread listener([&] { svr.listen_after_bind(); });
    svr.wait_until_ready();
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(2);
    bool passed = true;
    for (int i = 1; i <= 2; ++i) {
        auto result = client.Post("/decode", "{}", "application/json");
        passed = passed && result && result->status == 200 &&
            result->body.find("captured-value-7") != std::string::npos &&
            result->body.find("data: [DONE]\n\n") != std::string::npos && calls == i;
    }
    svr.stop();
    listener.join();
    std::cout << "decodes=" << calls << " transport=" << (passed ? "PASS" : "FAIL") << "\n";
    return passed ? 0 : 1;
}
'''
    test_source = work / "sse_test.cpp"
    test_source.write_text(harness)
    binary = work / "sse_test"
    subprocess.run(["c++", "-std=c++17", "-pthread", "-O0",
                    "-I", str(args.upstream / "vendor/cpp-httplib"),
                    "-I", str(args.upstream / "vendor"),
                    str(test_source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
