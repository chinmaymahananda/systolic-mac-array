# FPGA INT8 CNN Accelerator (Systolic MAC Array)

Weight-stationary INT8 CNN inference accelerator, built from a parameterizable
systolic MAC array, verified against a NumPy golden reference and targeted at
Xilinx FPGA (Vivado) synthesis.

## Day 1 Update -- VERIFIED (real simulation results, not projected)

Conv1 is done and bit-exact. Ran on Icarus Verilog on this machine:

- All 20 golden test samples PASS
- 2,880 total accumulator values (20 samples x 4 channels x 36 pixels) bit-exact match against the Python golden model
- Found and fixed a real pipeline timing bug in the process: mac_en/mac_acc_clear needed to lag tap_idx by TWO cycles (not one as originally written) -- one cycle because img_addr/wgt_addr are themselves registered outputs of the controller, a second cycle because the ROMs are registered reads. This is exactly the kind of bug the verification step exists to catch.
- Command used: `iverilog -o sim.out rtl\mac_unit.v rtl\systolic_array.v rtl\conv1_controller.v rtl\conv1_mem.v rtl\top_conv1_accelerator.v tb\tb_conv1.v` then `vvp sim.out +SAMPLE=N` from the python/ directory (ROMs load hex files with relative paths).

Next: Conv2 controller (Cin=4, Cout=8, output 4x4) following the exact same pattern -- see docs/EXTENDING.md.

## Day 2 Update -- Conv2 VERIFIED, quantization scheme corrected

Conv2 done, first-try pass reusing the Conv1-verified 2-cycle delay pattern:
- All 20 golden samples PASS, 2,560 accumulator values (20 x 8 x 16) bit-exact

Also fixed a real design issue in the golden model itself: v1 recomputed each
activation's quantization scale per-sample (dynamic, runtime max-finding) --
not realistic weight-stationary hardware, since it would make every bias
sample-dependent. v2 uses fixed, calibrated per-tensor scales end-to-end
(including the input layer), constrained to powers of two so the inter-layer
requantization is a plain right-shift (ReLU -> shift -> clamp), no multiplier
needed for the rescale step. This is the standard approach real INT8
accelerators use. Accuracy retained: 90% on the calibration set (unchanged
from v1, confirming the fix didn't cost accuracy).

## Status (honest, as of Day 2 of build)

**Done and verified in software:**
- CNN trained from scratch on sklearn `digits` (8x8, 10-class) — **89.7% FP32 test accuracy**
- Post-training INT8 quantization (symmetric, per-tensor) — **90.0% accuracy retained** on a 20-sample check set (i.e., quantization cost ~0)
- Golden bit-accurate INT32 reference model for Conv1, Conv2, and FC (`quantize_and_golden.py`) — this is the ground truth the RTL must match cycle-for-cycle
- Golden test vectors exported to hex for Verilog `$readmemh` (20 samples)

**Written, NOT yet simulated (no Verilog simulator available in the dev sandbox — do this first on your machine):**
- `mac_unit.v` — INT8×INT8→INT32 MAC primitive
- `systolic_array.v` — parameterizable NxN array (reserved for FC layer / Conv2 extension)
- `conv1_controller.v` + `conv1_mem.v` + `top_conv1_accelerator.v` — full Conv1 pipeline (4 output channels, 3x3 kernel, 8x8→6x6)
- `tb/tb_conv1.v` — testbench skeleton comparing against golden vectors

**Known open risk — this is your Day 5 job, not a bug I'm hiding:**
The image/weight ROMs are registered (1-cycle read latency). The controller's
`acc_clear` timing needs to be verified against this latency in simulation —
see the NOTE at the top of `tb_conv1.v`. This is completely normal for a
first RTL pass; catching and fixing exactly this kind of off-by-one is what
the verification day is for. Don't skip it, don't panic if the first sim run
doesn't match.

**Not yet built:** Conv2 controller, FC controller, top-level chaining all
three layers, synthesis/timing/utilization reports, benchmark harness.
Same pattern as Conv1, different loop bounds — see `docs/EXTENDING.md`.

---

## Setup on your machine (do this today)

You need a Verilog simulator and (for real synthesis numbers) Vivado.

**Simulator — Icarus Verilog (free, fast, good enough for verification):**
- Windows: install via `choco install icarus-verilog` or download from the
  Icarus Verilog website; add to PATH.
- Then: `iverilog -o sim.out rtl/mac_unit.v rtl/systolic_array.v rtl/conv1_controller.v rtl/conv1_mem.v rtl/top_conv1_accelerator.v tb/tb_conv1.v`
- Run: `vvp sim.out +SAMPLE=0`
- Waveforms: add `$dumpfile("wave.vcd"); $dumpvars(0, tb_conv1);` to the
  testbench initial block, then view with **GTKWave** (free).

**Vivado (for real FPGA numbers — utilization, timing, power):**
- If you don't already have Vivado installed from your VLSI coursework,
  download **Vivado ML Edition (free WebPACK license)** from AMD/Xilinx.
- Target any board you already have access to (Basys3/Arty A7 are common
  and cheap if you need to buy one — Nexys A7 also works). If you have zero
  board access this week, that's fine: simulation + synthesis (not
  place-and-route) still gives you resource utilization and timing estimates
  to put in the README/resume, and that's sufficient for the portfolio goal.

**Python (already works, no install needed):**
```
cd python
python3 train_model.py              # trains CNN, saves model_fp32.npz
python3 quantize_and_golden.py       # quantizes, saves golden_vectors.npz
python3 export_hex_vectors.py        # exports golden vectors to hex for RTL
python3 export_weights_tapmajor.py   # exports weights in RTL-ready layout
```
All four already ran successfully — the `.npz` and hex files are included
in this package, so you don't have to rerun them unless you change the model.

---

## Day-by-day (7 days from today)

**Day 1 (today, remaining hours):** Get Icarus Verilog running on your
machine. Run the testbench. It will very likely NOT match the golden values
on the first try — that's expected. Use GTKWave to look at `img_addr` vs
`img_data` vs when `mac_acc_clear` fires, and fix the 1-cycle alignment
noted above. Goal for tonight: Conv1 bit-exact against `golden_hex/conv1_acc_*.hex`.

**Day 2:** Once Conv1 is verified, build `conv2_controller.v` — same
structure as `conv1_controller.v`, but Cin=4 (loop over input channels too),
Cout=8, output 4x4. Verify against `golden_hex/conv2_acc_*.hex`.

**Day 3:** Build `fc_controller.v` (128→10 matrix-vector product — this is
where `systolic_array.v` actually gets used well). Verify against
`golden_hex/logits_*.hex`. Chain conv1→conv2→fc into one top-level
`top_accelerator.v` with a single `start`/`done` handshake.

**Day 4:** End-to-end verification across all 20 golden samples in one
testbench run (loop `+SAMPLE=` 0..19, auto-compare, print pass/fail count).
Fix any remaining mismatches.

**Day 5:** Synthesize in Vivado (even without a physical board, run through
synthesis + implementation for timing/utilization reports). Record: LUTs,
FFs, BRAM usage, max clock frequency, total cycles per inference.

**Day 6:** Build the benchmark harness — measure cycles/inference, compute
throughput (inferences/sec at your achieved clock), compare against a naive
NumPy CPU baseline (`quantize_and_golden.py` timing) for a speedup number.
Add a resource-utilization table and a roofline-style plot to the README.

**Day 7:** Polish README (architecture diagram, results table, GIF/screen
recording of simulation waveforms or board demo), push to GitHub, write the
LinkedIn post, update resume bullet.

---

## Resume bullet (draft, fill in real numbers from Day 5-6)

> Designed and verified an INT8 quantized CNN accelerator in Verilog using a
> parameterizable systolic MAC array; achieved <FILL: cycles/inference> at
> <FILL: MHz> on <FILL: FPGA part>, <FILL: Nx> faster than a NumPy CPU
> baseline, with bit-exact verification against a Python golden model across
> 20 test vectors (<FILL: accuracy>% retained after INT8 quantization).

Do not publish this bullet with placeholder numbers still in it — fill them
in from your actual Day 5/6 results before it goes on your resume.
