# INT8 CNN Inference Accelerator (Systolic MAC Array)

A weight-stationary INT8 CNN inference accelerator built on top of the
parameterizable systolic MAC array in this repo. Three layers — Conv1, Conv2,
and FC — are each independently verified bit-exact in RTL simulation against
a Python golden model, then chained into a single top-level pipeline with a
unified `start`/`done` handshake and interlayer requantization bridges.

CI runs the full end-to-end testbench on every push (see the badge at the top
of the repo root README) — `tb/tb_top_accelerator.v` feeds all 20 calibration
samples through the real image → Conv1 → Conv2 → FC chain and checks every
output logit against golden values.

## Result

**All 20 end-to-end calibration samples pass bit-exact, from raw image to
final FC logits, matching FP32 baseline accuracy (90%).**

## Pipeline

```
image (8x8) ─► Conv1 (4 ch, 3x3) ─► requant ─► Conv2 (8 ch, 3x3) ─► requant ─► FC (128→10) ─► logits
              [conv1_controller.v]           [conv2_controller.v]           [fc_controller.v]
```

- **Model**: a small CNN trained from scratch on sklearn's `digits` dataset
  (8x8 grayscale, 10-class) — 89.7% FP32 test accuracy.
- **Quantization**: post-training INT8, symmetric per-tensor, with scales
  constrained to powers of two so every interlayer rescale is a plain
  right-shift (ReLU → shift → clamp) — no multiplier needed for
  requantization, matching how real INT8 accelerators handle this. Accuracy
  retained: 90% on the 20-sample calibration set (quantization cost ~0).
- **Golden reference**: a from-scratch INT32 model (`quantize_and_golden_v2.py`)
  that reproduces the exact integer arithmetic — MACs, ReLU, shift, clamp —
  the RTL must match cycle-for-cycle. This is the ground truth used to
  generate every `.hex` test vector under `python/golden_hex*/`.

## Repository layout

```
cnn-accelerator/
├── rtl/
│   ├── mac_unit.v              # INT8×INT8→INT32 MAC primitive
│   ├── conv1_controller.v      # Conv1: 4 output channels, 3x3 kernel, 8x8→6x6
│   ├── conv2_controller.v      # Conv2: Cin=4, Cout=8, 3x3 kernel, →4x4
│   ├── fc_controller.v         # FC: 128→10 matrix-vector product
│   ├── requant_bridge.v        # interlayer ReLU + power-of-two rescale
│   ├── chain_mem.v             # shared memory glue between chained stages
│   └── top_accelerator.v       # top-level: Conv1 → Conv2 → FC, single start/done
├── tb/
│   ├── tb_conv1.v / tb_conv2.v / tb_fc.v   # per-layer bit-exact checks
│   └── tb_top_accelerator.v                # full chain, all 20 samples
├── python/
│   ├── train_model.py              # trains the FP32 CNN
│   ├── quantize_and_golden_v2.py   # quantizes + generates golden INT32 vectors
│   ├── export_hex_vectors.py       # golden vectors → hex for $readmemh
│   ├── export_weights_tapmajor.py  # weights → RTL-ready tap-major layout
│   └── golden_hex*/, weights_hex*/ # generated test vectors and weights
└── docs/
    └── EXTENDING.md             # how Conv2/FC reuse the Conv1 pattern
```

## Running the simulation

Requires [Icarus Verilog](http://iverilog.icarus.com/) (`iverilog`/`vvp`).

```bash
cd cnn-accelerator/python
iverilog -g2012 -o sim_top ../rtl/chain_mem.v ../rtl/conv1_controller.v \
  ../rtl/conv1_mem.v ../rtl/conv2_controller.v ../rtl/conv2_mem.v \
  ../rtl/fc_controller.v ../rtl/fc_mem.v ../rtl/mac_unit.v \
  ../rtl/requant_bridge.v ../rtl/systolic_array.v ../rtl/top_accelerator.v \
  ../tb/tb_top_accelerator.v
vvp sim_top
```

Expected output: all 20 samples reported bit-exact against the golden logits,
ending in `PASS: all 20/20 samples bit-exact end-to-end` (this is exactly the
check CI runs on every push).

Per-layer testbenches (`tb_conv1.v`, `tb_conv2.v`, `tb_fc.v`) can be run the
same way against their respective `.hex` vectors for isolated debugging.

## Notable implementation details

- **Two-cycle control latency.** Both the image/weight ROMs and the
  controller's own address outputs are registered, so `mac_en`/`mac_acc_clear`
  must lag the tap index by two cycles, not one — a real timing bug caught and
  fixed during Conv1 bring-up by comparing waveforms against the golden
  per-tap accumulator values.
- **Fixed, calibrated quantization scales.** An earlier version recomputed
  each activation's scale dynamically per-sample; this was corrected to fixed,
  per-tensor scales (matching real weight-stationary hardware, where a scale
  can't depend on runtime data) with no accuracy loss.
- **Verification discipline.** Every controller is checked the same way:
  golden vectors from the Python model, `$readmemh` into the testbench,
  bit-exact comparison — not "looks about right" on a waveform.

## Possible extensions

- Synthesis (Vivado/Yosys) for real utilization, timing, and power numbers.
- A benchmark harness comparing cycles/inference against a NumPy CPU baseline.
- Streaming/pipelined operation across samples instead of one at a time.

## Author

**Chinmay Mahananda** — MS ECE, Northeastern University
(Hardware & Software for Machine Intelligence)
