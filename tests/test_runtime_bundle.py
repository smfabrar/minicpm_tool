from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from duplex_tools.runtime_bundle import sha256_file


class RuntimeBundleTests(unittest.TestCase):
    def test_sha256_file_reads_in_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"duplex-runtime")
            self.assertEqual(
                sha256_file(path),
                "808da74a3e3446ac9b6e9b275fad786267f8f317a68a77be1e6788a2186e83a0",
            )


if __name__ == "__main__":
    unittest.main()
