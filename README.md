# gzip-golf

**Bit-level RFC 1951/1952 GZIP golfing in pure Python — zero compression libraries.**

`gzip-golf` hand-crafts valid `.gz` archives of any exact byte size (`>= 20` bytes) without importing `zlib` or `gzip`. Instead of feeding buffers into a general-purpose compressor, it treats `inflate` as a tiny byte-generating virtual machine and packs the shortest possible RFC 1951 bit-stream directly into an RFC 1952 frame.

---

## Compressed Archives as Byte-Generating Programs

A `.gz` archive is often thought of as "shrunk data," but at the bit level an **RFC 1951 DEFLATE** stream is an instruction tape executed over a 32 KiB sliding window using just two primitives:

1. **`EMIT <literal>`** — append one literal byte (`0x00..0xFF`) to the output window.
2. **`COPY <length, distance>`** — copy `3..258` bytes starting `1..32768` bytes back in the output window.

When `distance = 1`, `COPY <258, 1>` becomes a hardware-speed run-length multiplier: seeding a **single** `0x00` byte and issuing `N` back-references of `(length=258, distance=1)` synthesizes `1 + 258 * N` zero bytes without ever storing the uncompressed payload in memory.

Wrapping that bit-stream in a minimal **18-byte RFC 1952 envelope** (`10`-byte header + `8`-byte trailer) — with the IEEE 802.3 CRC-32 computed in $O(\log N)$ time via $\text{GF}(2)^{32}$ matrix exponentiation on `0xEDB88320` — lets pure Python generate valid high-ratio `.gz` files in milliseconds.

---

## Anatomy of a 28-Byte `1 KiB` Archive

[`examples/minimal_28b_1k.py`](examples/minimal_28b_1k.py) hand-assembles a **28-byte** `.gz` file that decompresses to **exact `1,024` (`1.00 KiB`) zero bytes** (`36.57 : 1`):

```text
1f 8b 08 00 00 00 00 00 00 03  63 18 05 a3 60 14 8c 58 00 00  2e af b5 ef  00 04 00 00
└───────────┬───────────────┘  └─────────────┬─────────────┘  └────┬────┘  └────┬────┘
   10B RFC 1952 Header            10B DEFLATE Stream (75b)     4B CRC-32    4B ISIZE
```

| Offset | Hex Bytes | Spec Field | Meaning |
| :--- | :--- | :--- | :--- |
| `00..09` | `1f 8b 08 00 00 00 00 00 00 03` | **GZIP Header** | Magic `1f 8b`, `CM=8` (DEFLATE), `FLG=0`, `MTIME=0`, `XFL=0`, `OS=3` (Unix) |
| `10..19` | `63 18 05 a3 60 14 8c 58 00 00` | **DEFLATE Payload** | `BFINAL=1` (1b) + `BTYPE=01` Fixed Huffman (2b)<br>• `EMIT 0x00` (`0x30`, 8b) → **1 B**<br>• `3 × COPY(258, 1)` (`3 × [0xC5 (8b) + 0x00 (5b)]`, 39b) → **+774 B**<br>• `COPY(249, 1)` (`0xC4` (8b) + extra `22` (5b) + `0x00` (5b), 18b) → **+249 B**<br>• `EOB` sym 256 (`0x00`, 7b) + `5` zero pad bits = **80 bits (10 B)** |
| `20..23` | `2e af b5 ef` | **CRC32** | Little-endian IEEE 802.3 CRC-32 of 1,024 `0x00` bytes (`0xefb5af2e`) |
| `24..27` | `00 04 00 00` | **ISIZE** | Little-endian uncompressed length $\bmod 2^{32}$ (`0x00000400 = 1,024`) |

> **Golfing quirk:** Hitting *exact* `1,024` bytes takes **28 bytes** (`75` active bits) because `Length 249` (symbol `284`) carries `5` extra length bits. In RFC 1951, `Length 258` (symbol `285`) has **zero** extra bits — so `1 + 4 × 258 = 1,033` zero bytes actually fits in `70` active bits (**27 bytes**)!

---

## Mathematical Milestones & The `1,032 : 1` Ceiling

`gzip_golf.py` solves for the maximum zero-byte output achievable within any target byte budget `S >= 20`:

| `.gz` Size (`S`) | Uncompressed Output | Ratio | Mode | Why This Byte Count Matters |
| ---: | ---: | ---: | :--- | :--- |
| **20 B** | `0 B` | `0.00 : 1` | Fixed (`01`) | **Minimum valid `.gz`**: `10B` header + `10-bit` empty block (`BFINAL=1, BTYPE=01, EOB`) + `8B` trailer |
| **21 B** | `1 B` | `0.05 : 1` | Fixed (`01`) | First non-empty `.gz` (`18` active bits: header + single `0x00` literal + `EOB`) |
| **25 B** | `527 B` | `21.08 : 1` | Fixed (`01`) | `1` literal + `2 × 258B` copies + `1 × 10B` copy (`sym 264`, 7b + 5b dist) filling a 12-bit remainder |
| **27 B** | `1,033 B` | `38.26 : 1` | Fixed (`01`) | Exceeds `1 KiB` in `70` bits (`1 + 4 × 258B`) because `sym 285` needs `0` extra bits |
| **28 B** | `1,034 B` | `36.93 : 1` | Fixed (`01`) | `27B` stream + one trailing 8-bit `0x00` literal (`78` bits <= `80` bits) |
| **33 B** | `1,808 B` | `54.79 : 1` | Fixed (`01`) | **Last byte size where Fixed Huffman wins** (`1,808 B` Fixed vs. `1,549 B` Dynamic) |
| **34 B** | `2,581 B` | `75.91 : 1` | Dynamic (`10`) | **Dynamic Huffman crossover**: custom 103-bit tree beats Fixed (`2,581 B` vs. `2,065 B`) |
| **1,024 B** | `1,024,261 B` | **`1000.25 : 1`** | Dynamic (`10`) | `1.00 KiB` archive crosses the **`1,000 : 1`** mark (`999.99+ : 1` is breached at `1,016 B`) |
| **$S \to \infty$** | $1032S - 32507\text{ B}$ | **`1032.00 : 1`** | Dynamic (`10`) | **Asymptotic single-pass RFC 1951 DEFLATE ceiling** |

### Why `1,032 : 1` Is the Single-Pass DEFLATE Limit

In Dynamic Huffman mode (`BTYPE=10`), `gzip-golf` emits a compact **103-bit** block preamble (`17`-bit header + `54`-bit code-length table + `32`-bit run-length-encoded tree description using symbol `18`) that assigns the shortest legal canonical Huffman codes:

* Literal/Length tree: `sym 285 (Length 258) -> 0` (**1 bit**), `sym 0 (0x00) -> 10` (**2 bits**), `sym 256 (EOB) -> 11` (**2 bits**).
* Distance tree: `dist 0 (Distance 1) -> 0` (**1 bit**).

After `103` tree bits + `2` seed-literal bits + `2` EOB bits (**107 bits** fixed overhead), every `258`-byte copy is encoded as two zero bits (`00`):

$$\frac{258 \text{ uncompressed bytes}}{2 \text{ compressed bits}} = 129 \text{ bytes/bit} = \mathbf{1,032 \text{ bytes per compressed byte}}$$

For every target archive size $S \ge 34$ bytes (`8(S - 18)` payload bits), the exact output size follows the closed-form linear law:

$$U(S) = 1 + 258 \left\lfloor \frac{8(S - 18) - 107}{2} \right\rfloor = 1032S - 32507$$

*(Note: every payload byte after the preamble is `0x00` until the final `11` EOB code, which is why multi-pass "zip bombs" compress a `.gz` a second time — whereas `1,032 : 1` is the strict information-theoretic ceiling of a single standard RFC 1951 DEFLATE pass.)*

---

## Usage

Requires **Python 3.10+** (standard library only; zero external dependencies).

### 1. Interactive Mode

Run `gzip_golf.py` with no arguments to preview the mode, byte budget, and compression ratio before writing:

```bash
python3 gzip_golf.py
```

```text
Target compressed .gz size in bytes (>= 20): 1024

--- Preview ---
Encoding mode     : Dynamic Huffman (BTYPE=10)
Compressed size   : 1.00 KiB (1,024 B)
Uncompressed size : 1000.25 KiB (1,024,261 B)
Compression ratio : 1000.25 : 1
Output file       : sample_1024b.gz

Create this file? [y/N]: y
Saved: sample_1024b.gz (1024 bytes)
```

### 2. Command-Line Flags

Use `-s/--size`, `-o/--output`, and `-y/--yes` for scripted generation:

```bash
python3 gzip_golf.py -s 34 -o crossover_34b.gz -y
python3 gzip_golf.py -s 1024 -y
```

### 3. Streaming Verification Tip

Verify any generated archive directly with system `gzip` in a streaming pipe (counts decompressed bytes on the fly without writing uncompressed output to disk):

```bash
gzip -dc sample_1024b.gz | wc -c
# 1024261
```

### 4. Standalone 28-Byte `1 KiB` Walkthrough

```bash
python3 examples/minimal_28b_1k.py
gzip -dc manual_1k_zeros.gz | wc -c
# 1024
```

### 5. Python API

```python
from gzip_golf import build_gzip_sample, estimate_uncompressed_size

expected_len, mode = estimate_uncompressed_size(1024)
gz_bytes, uncompressed_len = build_gzip_sample(1024)
assert len(gz_bytes) == 1024 and uncompressed_len == 1_024_261
```

---

## Running Tests

The test suite verifies all Fixed and Dynamic Huffman milestones, checks `crc32_zeros` against `zlib.crc32`, ensures zero compression-module imports in the encoder, and validates every generated `.gz` against both `gzip.decompress` and `/usr/bin/gzip -dc`:

```bash
python3 -m unittest discover -s tests
```

---

## License

Released under the [MIT License](LICENSE).

## Author & Credits

Designed, mathematically derived, and written from scratch by **Gemini 4 Carbon (High)** via **Antigravity** (Google DeepMind) — covering the bit-level RFC 1951/1952 encoder, exact Huffman optimality proofs, test suite, and documentation.
