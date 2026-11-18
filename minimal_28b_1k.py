#!/usr/bin/env python3
"""Standalone walkthrough: hand-assembling a 28-byte .gz of 1,024 zero bytes.

Demonstrates how a GZIP archive acts as a tiny byte-generating instruction
stream by manually packing every bit of an RFC 1952 container and an RFC 1951
Fixed Huffman (BTYPE=01) DEFLATE stream — with zero compression libraries.

Bit budget inside the 10-byte (80-bit) DEFLATE payload:
  - Block header : 1b (BFINAL=1) + 2b (BTYPE=01)                  =  3 bits
  - Seed byte    : Literal 0x00 (code 0x30, 8b)          ->    1B =  8 bits
  - 3x max copy  : 3 x [Len 258 (0xC5, 8b) + Dist 1 (0, 5b)] -> 774B = 39 bits
  - Tail copy    : Len 249 (sym 284 = 0xC4, 8b + extra 22, 5b)
                   + Dist 1 (0, 5b)                      ->  249B = 18 bits
  - End-of-block : Symbol 256 (code 0x00, 7b)                     =  7 bits
  - Byte padding : 5 zero padding bits                            =  5 bits
  -------------------------------------------------------------------------
  Total payload  : 80 bits = 10 bytes  (emitting 1 + 774 + 249 = 1,024 bytes)
  Total archive  : 10B GZIP header + 10B DEFLATE + 8B GZIP trailer = 28 bytes
"""

from __future__ import annotations

import sys
from pathlib import Path


class DeflateBitWriter:
    """Minimal RFC 1951 bit packer (LSB-first fields, MSB-first Huffman codes)."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.cur_byte = 0
        self.bit_pos = 0

    def write_lsb(self, value: int, num_bits: int) -> None:
        for i in range(num_bits):
            if (value >> i) & 1:
                self.cur_byte |= 1 << self.bit_pos
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.buf.append(self.cur_byte)
                self.cur_byte = 0
                self.bit_pos = 0

    def write_huffman(self, code: int, num_bits: int) -> None:
        for i in range(num_bits - 1, -1, -1):
            self.write_lsb((code >> i) & 1, 1)

    def finish(self) -> bytes:
        if self.bit_pos > 0:
            self.buf.append(self.cur_byte)
            self.cur_byte = 0
            self.bit_pos = 0
        return bytes(self.buf)


def crc32_of_zeros(length: int) -> int:
    """Bit-by-bit IEEE 802.3 CRC-32 (reflected polynomial 0xEDB88320) over zeros."""
    crc = 0xFFFFFFFF
    for _ in range(length):
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF


def build_1k_zeros_gzip() -> bytes:
    """Assemble a 28-byte GZIP archive that expands to 1,024 (1 KiB) zero bytes."""
    target_size = 1024

    # 1. RFC 1952 10-byte GZIP header:
    #    1f 8b (magic) | 08 (DEFLATE) | 00 (no flags) | 00 00 00 00 (MTIME=0)
    #    | 00 (XFL=0)  | 03 (OS=Unix)
    gzip_header = bytes([0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03])

    # 2. RFC 1951 Fixed Huffman DEFLATE instruction stream (75 active bits -> 10 bytes):
    bw = DeflateBitWriter()
    bw.write_lsb(1, 1)             # BFINAL = 1 (final block)
    bw.write_lsb(0b01, 2)          # BTYPE  = 01 (Fixed Huffman)
    bw.write_huffman(0x30, 8)      # EMIT 0x00               (total = 1 byte)

    for _ in range(3):
        bw.write_huffman(0xC5, 8)  # COPY length 258 (sym 285)
        bw.write_huffman(0x00, 5)  #      distance 1 (code 0) (total = 775 bytes)

    bw.write_huffman(0xC4, 8)      # COPY length base 227 (sym 284)
    bw.write_lsb(249 - 227, 5)     #      + 22 extra -> length 249
    bw.write_huffman(0x00, 5)      #      distance 1 (code 0) (total = 1,024 bytes)
    bw.write_huffman(0x00, 7)      # EOB (symbol 256)

    deflate_payload = bw.finish()

    # 3. RFC 1952 8-byte GZIP trailer: CRC32 (4B LE) + ISIZE (4B LE).
    crc = crc32_of_zeros(target_size)
    gzip_trailer = crc.to_bytes(4, "little") + target_size.to_bytes(4, "little")
    return gzip_header + deflate_payload + gzip_trailer


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("manual_1k_zeros.gz")
    raw_gz = build_1k_zeros_gzip()
    out_path.write_bytes(raw_gz)

    print(f"Created : {out_path} ({len(raw_gz)} bytes -> 1,024 zero bytes, 36.57:1)")
    print(f"Header  : {raw_gz[:10].hex(' ')}")
    print(f"DEFLATE : {raw_gz[10:20].hex(' ')}")
    print(f"Trailer : {raw_gz[20:24].hex(' ')} (CRC32) | {raw_gz[24:28].hex(' ')} (ISIZE)")
    print(f"Full hex: {raw_gz.hex(' ')}")


if __name__ == "__main__":
    main()
