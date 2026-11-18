"""Unit tests for gzip-golf bit-level RFC 1951/1952 encoder and examples."""

from __future__ import annotations

import ast
import gzip
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gzip_golf  # noqa: E402
from examples.minimal_28b_1k import build_1k_zeros_gzip  # noqa: E402

GZIP_BIN = "/usr/bin/gzip" if Path("/usr/bin/gzip").exists() else shutil.which("gzip")


class TestGzipGolf(unittest.TestCase):
    """Verify exact sizes, mathematical milestones, and RFC 1951/1952 compliance."""

    MILESTONES: list[tuple[int, int, str]] = [
        (20, 0, "Fixed Huffman (BTYPE=01)"),
        (21, 1, "Fixed Huffman (BTYPE=01)"),
        (25, 527, "Fixed Huffman (BTYPE=01)"),
        (27, 1033, "Fixed Huffman (BTYPE=01)"),
        (28, 1034, "Fixed Huffman (BTYPE=01)"),
        (33, 1808, "Fixed Huffman (BTYPE=01)"),
        (34, 2581, "Dynamic Huffman (BTYPE=10)"),
        (1024, 1024261, "Dynamic Huffman (BTYPE=10)"),
    ]

    def _verify_with_system_gzip(self, gz_bytes: bytes, expected_len: int) -> None:
        self.assertIsNotNone(GZIP_BIN, "System `gzip` binary not found")
        proc = subprocess.run(
            [str(GZIP_BIN), "-dc"],
            input=gz_bytes,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`{GZIP_BIN} -dc` failed with stderr: {proc.stderr.decode(errors='replace')}",
        )
        self.assertEqual(proc.stderr, b"")
        self.assertEqual(len(proc.stdout), expected_len)
        self.assertEqual(proc.stdout, b"\x00" * expected_len)

    def test_no_compression_imports_in_encoder(self) -> None:
        """Ensure encoder sources never import stdlib compression modules."""
        forbidden = {"zlib", "gzip", "bz2", "lzma"}
        for rel_path in ("gzip_golf.py", "examples/minimal_28b_1k.py"):
            tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(
                imported.isdisjoint(forbidden),
                f"{rel_path} imported forbidden compression library: {imported & forbidden}",
            )

    def test_milestones(self) -> None:
        """Verify every documented Fixed and Dynamic Huffman milestone."""
        for gz_size, expected_uncompressed, expected_mode in self.MILESTONES:
            with self.subTest(gz_size=gz_size):
                est_len, mode = gzip_golf.estimate_uncompressed_size(gz_size)
                self.assertEqual(est_len, expected_uncompressed)
                self.assertEqual(mode, expected_mode)

                gz_bytes, built_len = gzip_golf.build_gzip_sample(gz_size)
                self.assertEqual(built_len, expected_uncompressed)
                self.assertEqual(len(gz_bytes), gz_size)

                decompressed = gzip.decompress(gz_bytes)
                self.assertEqual(len(decompressed), expected_uncompressed)
                self.assertEqual(decompressed, b"\x00" * expected_uncompressed)

                self._verify_with_system_gzip(gz_bytes, expected_uncompressed)

    def test_contiguous_byte_sweep(self) -> None:
        """Verify every target byte size across the Fixed-to-Dynamic boundary (20..64B)."""
        for gz_size in range(20, 65):
            with self.subTest(gz_size=gz_size):
                gz_bytes, expected_len = gzip_golf.build_gzip_sample(gz_size)
                self.assertEqual(len(gz_bytes), gz_size)
                self.assertEqual(gzip.decompress(gz_bytes), b"\x00" * expected_len)

    def test_minimal_28b_1k_example(self) -> None:
        """Verify standalone 28-byte 1,024-zero-byte educational script."""
        gz_bytes = build_1k_zeros_gzip()
        self.assertEqual(len(gz_bytes), 28)
        self.assertEqual(gzip.decompress(gz_bytes), b"\x00" * 1024)
        self._verify_with_system_gzip(gz_bytes, 1024)

    def test_crc32_zeros_matches_zlib(self) -> None:
        """Verify GF(2) CRC-32 implementation against `zlib.crc32`."""
        for length in (0, 1, 2, 7, 31, 255, 527, 1024, 1033, 2581, 65536, 1024261):
            with self.subTest(length=length):
                self.assertEqual(
                    gzip_golf.crc32_zeros(length),
                    zlib.crc32(b"\x00" * length),
                )
        with self.assertRaises(ValueError):
            gzip_golf.crc32_zeros(-1)

    def test_rejects_sizes_below_minimum(self) -> None:
        """Verify sizes under 20 bytes raise ValueError."""
        for invalid_size in (-5, 0, 1, 19):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(ValueError):
                    gzip_golf.estimate_uncompressed_size(invalid_size)
                with self.assertRaises(ValueError):
                    gzip_golf.build_gzip_sample(invalid_size)

    def test_cli_flags_and_interactive_mode(self) -> None:
        """Verify both non-interactive flags and interactive stdin prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Flag mode (-s, -o, -y)
            out_flag = tmp_path / "flag_34b.gz"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gzip_golf.main(["-s", "34", "-o", str(out_flag), "-y"])
            self.assertEqual(rc, 0)
            self.assertEqual(out_flag.stat().st_size, 34)
            self.assertEqual(len(gzip.decompress(out_flag.read_bytes())), 2581)

            # 2. Interactive prompt mode (size + confirmation)
            out_interactive = tmp_path / "interactive_27b.gz"
            buf = io.StringIO()
            with patch("builtins.input", side_effect=["27", "y"]), redirect_stdout(buf):
                rc = gzip_golf.main(["-o", str(out_interactive)])
            self.assertEqual(rc, 0)
            self.assertEqual(out_interactive.stat().st_size, 27)
            self.assertEqual(len(gzip.decompress(out_interactive.read_bytes())), 1033)


if __name__ == "__main__":
    unittest.main()
