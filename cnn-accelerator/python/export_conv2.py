"""
Export Conv2 golden vectors + weights for RTL, using the v2 (fixed-shift,
calibrated) quantization scheme. Conv2 is verified as a standalone block:
its input (c1_q, Conv1's requantized output) is loaded from memory exactly
like Conv1 loaded the raw image -- chaining Conv1->Conv2 directly happens
at the top-level integration step (Day 3/4), not here.
"""
import numpy as np
import os

def relu_requant(acc32, shift):
    relu = np.maximum(0, acc32.astype(np.int64))
    if shift > 0:
        rounded = (relu + (1 << (shift - 1))) >> shift
    else:
        rounded = relu
    return np.clip(rounded, 0, 127).astype(np.int8)

def to_hex_lines(arr, bits):
    hexchars = bits // 4
    mask = (1 << bits) - 1
    return [f'{int(v) & mask:0{hexchars}x}' for v in arr.flatten()]

def main():
    gv = np.load('golden_vectors_v2.npz')
    q = np.load('model_int8_v2.npz')
    with open('requant_shifts.txt') as f:
        shifts = dict(line.strip().split('=') for line in f if line.strip())
    shift1 = int(shifts['shift1'])

    N = gv['conv1_acc'].shape[0]
    os.makedirs('golden_hex_v2', exist_ok=True)

    c1_q_all = []
    for i in range(N):
        c1_q = relu_requant(gv['conv1_acc'][i], shift1)  # (4,6,6) int8
        c1_q_all.append(c1_q)
        with open(f'golden_hex_v2/conv2_input_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(c1_q, 8)) + '\n')
        with open(f'golden_hex_v2/conv2_acc_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['conv2_acc'][i], 32)) + '\n')

    # Weight export: w2 shape (8,4,3,3) -> tap-major over (ci,ky,kx), Cout=8 channels packed per tap
    w2 = q['w2']
    Cout, Cin, kh, kw = w2.shape
    lines = []
    for ci in range(Cin):
        for ky in range(kh):
            for kx in range(kw):
                word = 0
                for co in range(Cout):
                    byte = int(w2[co, ci, ky, kx]) & 0xFF
                    word |= (byte << (8 * co))
                lines.append(f'{word:016x}')  # 8 channels x 8 bits = 64 bits = 16 hex chars
    os.makedirs('weights_hex_v2', exist_ok=True)
    with open('weights_hex_v2/w2_tapmajor.hex', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Fixed bias b2 (identical across samples now that input_scale/c1_out_scale are fixed) --
    # recompute canonically here instead of trusting per-sample equality by assumption.
    d = np.load('model_fp32.npz')
    x_scale = float(np.max(np.abs(np.load('test_data.npz')['X_test'][:20]))) / 127.0
    w1_scale = float(q['w1_scale'])
    c1_out_scale = x_scale * w1_scale * (2 ** shift1)
    w2_scale = float(q['w2_scale'])
    b2_q = np.round(d['b2'] / (c1_out_scale * w2_scale)).astype(np.int32)
    word = 0
    for co in range(Cout):
        word |= (int(b2_q[co]) & 0xFFFFFFFF) << (32 * co)
    with open('weights_hex_v2/b2.hex', 'w') as f:
        f.write(f'{word:064x}\n')  # 8 x 32-bit = 256 bits = 64 hex chars

    print(f"Exported {N} Conv2 golden samples, w2_tapmajor.hex ({len(lines)} taps), b2.hex")
    print(f"b2_q = {b2_q.tolist()}")

if __name__ == '__main__':
    main()
