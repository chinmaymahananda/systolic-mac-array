# Systolic MAC Array — Matrix-Multiply Accelerator in Verilog

A synthesizable, fully parameterizable **output-stationary systolic array** that
computes `C = A × B` — the matrix-multiply primitive at the heart of every neural
network. This is the same class of architecture used in Google's TPU MXU and the
matrix engines inside modern GPUs/NPUs, implemented from the processing element
up and verified with a self-checking testbench.

```
A (rows) ─►┌──────┬──────┬──────┬──────┐
           │ PE00 │ PE01 │ PE02 │ PE03 │   each PE:  acc += a·b
           ├──────┼──────┼──────┼──────┤              a ─► right
           │ PE10 │ PE11 │ PE12 │ PE13 │              b ─► down
           ├──────┼──────┼──────┼──────┤
           │ PE20 │ PE21 │ PE22 │ PE23 │   PE(i,j) accumulates C[i][j]
           ├──────┼──────┼──────┼──────┤
           │ PE30 │ PE31 │ PE32 │ PE33 │
           └──┬───┴──┬───┴──┬───┴──┬───┘
              ▼      ▼      ▼      ▼
                 B (columns, fed from the top)
```

## Why this design

A naïve matrix multiply re-reads operands from memory for every multiply. A
systolic array instead lets each operand **flow through a grid of processing
elements**, so every value loaded from memory is reused across an entire row or
column. Operands march one PE per clock; each PE does one multiply-accumulate
per cycle. The result is high arithmetic intensity with simple, local,
short-wire connections — which is exactly why this topology dominates ML
accelerator hardware.

## Features

- **Output-stationary dataflow** — each PE owns one output element and
  accumulates its dot product in place.
- **Diagonal operand skew** — the driver staggers the feed so `A[i][k]` and
  `B[k][j]` always meet at `PE(i,j)` on the same cycle; zero-padding outside the
  valid window keeps the accumulate enable high throughout.
- **Fully parameterizable** — array size `N`, operand width `DATA_WIDTH`, and
  accumulator width `ACC_WIDTH` are all parameters; the grid is built with
  `generate` loops and scales to any `N` with no hand-edits.
- **Signed two's-complement** arithmetic with a widened accumulator to prevent
  overflow across the `K` accumulation steps.
- **Self-checking testbench** — random operands, an in-testbench golden model,
  and automatic PASS/FAIL with a non-zero exit on mismatch. Dumps a VCD for
  waveform inspection.

## Repository layout

```
systolic-mac-array/
├── rtl/
│   ├── pe.v               # processing element (MAC + operand registers)
│   └── systolic_array.v   # N×N grid, parameterizable, generate-built
├── tb/
│   └── tb_systolic.v      # self-checking testbench + golden model
├── sim/
│   └── verify_model.py    # cycle-accurate reference model (verification)
├── Makefile
└── README.md
```

## Running the simulation

Requires [Icarus Verilog](http://iverilog.icarus.com/) (`iverilog`/`vvp`).

```bash
make            # compile RTL + testbench, run the self-check
```

Or manually:

```bash
iverilog -g2012 -o sim/tb rtl/pe.v rtl/systolic_array.v tb/tb_systolic.v
vvp sim/tb
```

Expected output:

```
=== systolic_array self-check : 4x4, DATA_WIDTH=8 ===
  test 1: PASS
  test 2: PASS
  ...
  test 20: PASS
================================================
ALL 20 TESTS PASSED
================================================
```

Inspect the dataflow as a waveform:

```bash
make wave       # opens sim/dump.vcd in GTKWave
```

## Verification

Two independent layers:

1. **RTL testbench** (`tb/tb_systolic.v`) — 20 randomized matrix multiplies per
   run, each checked element-by-element against a golden product computed in the
   testbench.
2. **Cycle-accurate model** (`sim/verify_model.py`) — a Python reference that
   replicates the exact register/accumulate semantics and the skewed feed, run
   over **1,900 cases** spanning 3×3 through 16×16, signed negatives, and the
   full int8 range. Used to prove the architecture and timing before synthesis.

## Design notes

- **Latency.** The last partial product for `PE(N-1,N-1)` is accumulated around
  cycle `3(N-1)`; the testbench clocks well past this before reading results.
- **Throughput.** Back-to-back matrices can be streamed with one full multiply
  every `~N` cycles once the pipeline is primed (current testbench runs one
  multiply at a time for clarity).
- **Overflow.** With `DATA_WIDTH`-bit signed operands and contraction depth `K`,
  the maximum magnitude product is bounded, and `ACC_WIDTH = 32` comfortably
  holds the sum for the tested ranges.

## Possible extensions

- Weight-stationary mode with a preload phase (closer to TPU operation).
- Tiling/streaming control to multiply matrices larger than the array.
- AXI-Stream operand/result interfaces for SoC integration.
- Fixed-point/bfloat16 PEs and a ReLU/requantize post-processing stage.

## Author

**Chinmay Mahananda** — MS ECE, Northeastern University
(Hardware & Software for Machine Intelligence)
