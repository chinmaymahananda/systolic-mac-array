# Extending Conv1 -> Conv2 -> FC

Conv1 (`conv1_controller.v`) establishes the pattern. Conv2 and FC reuse it
with different loop bounds — don't redesign from scratch, parameterize.

## Conv2 (Cin=4, Cout=8, 3x3 kernel, input 4x6x6 -> output 8x4x4)

Differences from Conv1:
- **Input channels > 1**: the golden model (`conv2d_int` in
  `quantize_and_golden.py`) sums over `ci` in addition to `ky, kx`. Your tap
  loop must become `Cin * K * K = 4*9 = 36` taps per output pixel instead of 9.
- **Input source**: Conv2's input is Conv1's *quantized, ReLU'd* output
  (`c1_q` in the golden script), not the raw image. You need a requantization
  stage between layers: apply ReLU to the accumulator, rescale by
  `combined_scale`, re-quantize to int8. Do this in a small combinational/
  pipelined block between `conv1` writeback and `conv2` read, or store Conv1's
  int8 output directly (compute requant in Python-verified integer math —
  see `conv2d_int`'s use of `x_scale`/`w_scale` for the exact formula, then
  replicate as fixed-point multiply + shift in RTL, NOT floating point).
- Weight/bias memory layout: same tap-major packing idea, but now indexed by
  `(ci, ky, kx)` tuples — regenerate with a variant of
  `export_weights_tapmajor.py` for `w2`.

## FC (128 -> 10)

This is a matrix-vector product: `out[j] = sum_i(x[i] * w[i][j]) + b[j]`.
This is where `systolic_array.v` is actually a good architectural fit: tile
the 128x10 weight matrix through the NxN array in blocks, streaming the
128-element input vector through. Simpler alternative if throughput doesn't matter for your use case: a single
serial MAC unit looping 128 cycles per output (10 outputs = 1280 cycles
total) -- much easier to get correct, at the cost of throughput. Ship that
first and upgrade to the systolic version once it's working end-to-end.
## General rule

Every new controller should be verified the same way Conv1 is: golden
vectors from the Python model, `$readmemh` into the testbench, bit-exact
comparison, not "looks about right." Recruiters will ask "how do you know
it's correct" — your answer should be "cycle-accurate match against a golden
model across N test vectors," not "I eyeballed the waveform."
