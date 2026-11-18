#!/usr/bin/env python3
"""Bit-level RFC 1951/1952 GZIP golfing in pure Python — zero compression libraries.

Constructs valid `.gz` archives of any exact target byte size (>= 20 bytes) that
maximize uncompressed zero-byte output in a single DEFLATE block, automatically
switching from Fixed Huffman (BTYPE=01) to a hand-golfed Dynamic Huffman tree
(BTYPE=10) at the exact 34-byte crossover point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MIN_GZ_SIZE = 20


class DeflateBitWriter:
    """RFC 1951 §3.1.1 bit-stream packer.

    Non-Huffman fields (block headers, extra length/distance bits) are packed
    LSB-first (`write_lsb`), whereas canonical Huffman codes are packed MSB-first
    (`write_huffman`).
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.cur_byte = 0
        self.bit_pos = 0

    def write_lsb(self, value: int, num_bits: int) -> None:
        """Write `num_bits` from `value` starting with the least-significant bit."""
        for i in range(num_bits):
            if (value >> i) & 1:
                self.cur_byte |= 1 << self.bit_pos
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.buf.append(self.cur_byte)
                self.cur_byte = 0
                self.bit_pos = 0

    def write_huffman(self, code: int, num_bits: int) -> None:
        """Write an MSB-first canonical Huffman code into the LSB-first stream."""
        for i in range(num_bits - 1, -1, -1):
            self.write_lsb((code >> i) & 1, 1)

    def write_zero_bits(self, num_bits: int) -> None:
        """Append `num_bits` zero bits in O(bytes) bulk operations."""
        if num_bits <= 0:
            return
        if self.bit_pos > 0:
            fill = min(num_bits, 8 - self.bit_pos)
            self.bit_pos += fill
            num_bits -= fill
            if self.bit_pos == 8:
                self.buf.append(self.cur_byte)
                self.cur_byte = 0
                self.bit_pos = 0
            if num_bits == 0:
                return
        full_bytes, rem_bits = divmod(num_bits, 8)
        if full_bytes > 0:
            self.buf.extend(b"\x00" * full_bytes)
        self.bit_pos = rem_bits

    def finish(self) -> bytes:
        """Flush any partial byte (zero-padded in high bits) and return bytes."""
        if self.bit_pos > 0:
            self.buf.append(self.cur_byte)
            self.cur_byte = 0
            self.bit_pos = 0
        return bytes(self.buf)


def _gf2_matrix_times(mat: list[int], vec: int) -> int:
    """Multiply a 32x32 GF(2) matrix (32 column bitmasks) by a 32-bit vector."""
    total = 0
    idx = 0
    while vec:
        if vec & 1:
            total ^= mat[idx]
        vec >>= 1
        idx += 1
    return total


def crc32_zeros(length: int) -> int:
    """Compute the RFC 1952 (IEEE 802.3) CRC-32 of `length` zero bytes.

    Uses GF(2) matrix exponentiation on the reflected polynomial 0xEDB88320
    to run in O(log length) time with zero memory overhead.
    """
    if length < 0:
        raise ValueError("Uncompressed length must be non-negative.")
    if length == 0:
        return 0

    # Linear operator for advancing the 32-bit CRC state by one 0x00 byte.
    op: list[int] = []
    for bit in range(32):
        c = 1 << bit
        for _ in range(8):
            c = (c >> 1) ^ (0xEDB88320 if (c & 1) else 0)
        op.append(c)

    crc = 0xFFFFFFFF
    n = length
    while n > 0:
        if n & 1:
            crc = _gf2_matrix_times(op, crc)
        n >>= 1
        if n > 0:
            op = [_gf2_matrix_times(op, col) for col in op]
    return crc ^ 0xFFFFFFFF


def estimate_uncompressed_size(gz_size: int) -> tuple[int, str]:
    """Return `(uncompressed_bytes, mode_description)` for a target `.gz` size."""
    if gz_size < MIN_GZ_SIZE:
        raise ValueError(f"Size must be at least {MIN_GZ_SIZE} bytes.")

    if gz_size == 20:
        return 0, "Fixed Huffman (BTYPE=01)"

    if gz_size <= 33:
        # 18 bytes GZIP framing leaves (gz_size - 18) * 8 payload bits.
        # Fixed Huffman header (3b) + EOB (7b) = 10b overhead.
        budget = (gz_size - 18) * 8 - 10
        rem = budget - 8  # Initial 0x00 literal takes 8 bits.
        n_m258, leftover = divmod(rem, 13)  # Length 258 (8b) + Distance 1 (5b) = 13b.
        extra = 10 if leftover == 12 else (1 if leftover >= 8 else 0)
        return 1 + n_m258 * 258 + extra, "Fixed Huffman (BTYPE=01)"

    # Dynamic Huffman header + minimalist trees (103b) + literal 0x00 (2b) + EOB (2b) = 107b.
    # Every subsequent 258-byte RLE match costs exactly 2 zero bits (`0` + `0`).
    avail_bits = (gz_size - 18) * 8
    num_matches = (avail_bits - 107) // 2
    return 1 + 258 * num_matches, "Dynamic Huffman (BTYPE=10)"


def build_gzip_sample(gz_size: int) -> tuple[bytes, int]:
    """Build an exact `gz_size`-byte `.gz` archive maximizing zero-byte output."""
    uncompressed_len, _ = estimate_uncompressed_size(gz_size)
    bw = DeflateBitWriter()

    if gz_size <= 33:
        bw.write_lsb(1, 1)     # BFINAL = 1 (final block)
        bw.write_lsb(0b01, 2)  # BTYPE  = 01 (Fixed Huffman)
        if gz_size >= 21:
            budget = (gz_size - 18) * 8 - 10
            rem = budget - 8
            n_m258, leftover = divmod(rem, 13)
            bw.write_huffman(0x30, 8)          # Literal 0x00
            for _ in range(n_m258):
                bw.write_huffman(0xC5, 8)      # Length 258 (symbol 285)
                bw.write_huffman(0x00, 5)      # Distance 1 (code 0)
            if leftover == 12:
                bw.write_huffman(0x08, 7)      # Length 10 (symbol 264, 7 bits)
                bw.write_huffman(0x00, 5)      # Distance 1 (code 0, 5 bits)
            elif leftover >= 8:
                bw.write_huffman(0x30, 8)      # Trailing literal 0x00 (8 bits)
        bw.write_huffman(0x00, 7)              # EOB (symbol 256, 7 bits)
    else:
        avail_bits = (gz_size - 18) * 8
        num_matches = (avail_bits - 107) // 2

        # 17-bit Dynamic Huffman block header:
        bw.write_lsb(1, 1)     # BFINAL = 1
        bw.write_lsb(0b10, 2)  # BTYPE  = 10 (Dynamic Huffman)
        bw.write_lsb(29, 5)    # HLIT   = 29 -> 286 literal/length codes (0..285)
        bw.write_lsb(0, 5)     # HDIST  = 0  -> 1 distance code (0)
        bw.write_lsb(14, 4)    # HCLEN  = 14 -> 18 code-length codes

        # 54-bit code-length alphabet: {18: 1b ('0'), 1: 2b ('10'), 2: 2b ('11')}.
        cl_order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1]
        cl_lens = {18: 1, 1: 2, 2: 2}
        for sym in cl_order:
            bw.write_lsb(cl_lens.get(sym, 0), 3)

        # 32-bit tree definitions:
        # Lit/Len lengths: sym 0 -> 2 ('10'), sym 256 -> 2 ('11'), sym 285 -> 1 ('0').
        # Distance lengths: sym 0 -> 1 ('0').
        bw.write_huffman(0b11, 2)       # sym 0 (literal 0x00): code length 2
        bw.write_huffman(0b0, 1)        # sym 18: repeat zero 138 times (1..138)
        bw.write_lsb(138 - 11, 7)
        bw.write_huffman(0b0, 1)        # sym 18: repeat zero 117 times (139..255)
        bw.write_lsb(117 - 11, 7)
        bw.write_huffman(0b11, 2)       # sym 256 (EOB): code length 2
        bw.write_huffman(0b0, 1)        # sym 18: repeat zero 28 times (257..284)
        bw.write_lsb(28 - 11, 7)
        bw.write_huffman(0b10, 2)       # sym 285 (length 258): code length 1
        bw.write_huffman(0b10, 2)       # dist 0 (distance 1): code length 1

        # Compressed stream: 1x literal 0x00 + N x (len=258, dist=1) + EOB.
        bw.write_huffman(0b10, 2)       # Emit initial literal 0x00
        bw.write_zero_bits(num_matches * 2)  # Each '00' copies 258 bytes at dist=1
        bw.write_huffman(0b11, 2)       # Emit EOB

    # 10-byte minimal RFC 1952 header (ID1=0x1f, ID2=0x8b, CM=8, FLG=0, MTIME=0, XFL=0, OS=3 Unix).
    header = bytes([0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03])
    payload = bw.finish()
    crc = crc32_zeros(uncompressed_len)
    # 8-byte RFC 1952 trailer: CRC32 (4B LE) + ISIZE mod 2^32 (4B LE).
    trailer = crc.to_bytes(4, "little") + (uncompressed_len & 0xFFFFFFFF).to_bytes(4, "little")
    return header + payload + trailer, uncompressed_len


def format_bytes(num_bytes: int) -> str:
    """Format a byte count with human-readable binary units."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB ({num_bytes:,} B)"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f} MiB ({num_bytes:,} B)"
    return f"{num_bytes / 1024**3:.2f} GiB ({num_bytes:,} B)"


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Bit-level RFC 1951/1952 GZIP golfing in pure Python. "
            "Run with no arguments for interactive mode."
        )
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        help=f"Target compressed .gz size in bytes (>= {MIN_GZ_SIZE}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .gz file path (default: sample_<size>b.gz).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt and write immediately.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point supporting both interactive and flag-driven invocation."""
    args = build_arg_parser().parse_args(argv)

    try:
        if args.size is None:
            raw_input_val = input(
                f"Target compressed .gz size in bytes (>= {MIN_GZ_SIZE}): "
            ).strip()
            gz_size = int(raw_input_val)
        else:
            gz_size = args.size

        uncompressed_len, mode = estimate_uncompressed_size(gz_size)
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCanceled.")
        return 1

    ratio = uncompressed_len / gz_size if gz_size > 0 else 0.0
    out_path = args.output if args.output is not None else Path(f"sample_{gz_size}b.gz")

    print("\n--- Preview ---")
    print(f"Encoding mode     : {mode}")
    print(f"Compressed size   : {format_bytes(gz_size)}")
    print(f"Uncompressed size : {format_bytes(uncompressed_len)}")
    print(f"Compression ratio : {ratio:.2f} : 1")
    print(f"Output file       : {out_path}")

    if not args.yes:
        try:
            confirm = input("\nCreate this file? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCanceled.")
            return 1
        if confirm not in ("y", "yes"):
            print("Canceled.")
            return 0

    gz_data, _ = build_gzip_sample(gz_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gz_data)
    print(f"Saved: {out_path} ({len(gz_data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
