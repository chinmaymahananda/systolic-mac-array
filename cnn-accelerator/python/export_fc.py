"""
Export FC layer golden vectors + weights. FC input is Conv2's requantized
output (c2_q, 128 values = 8 channels x 4x4 flattened), reusing shift2 from
requant_shifts.txt exactly as Conv2 reused shift1 from Conv1.
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
    shift2 = int(shifts['shift2'])

    N = gv['conv2_acc'].shape[0]
    os.makedirs('golden_hex_fc', exist_ok=True)

    correct = 0
    for i in range(N):
        c2_q = relu_requant(gv['conv2_acc'][i], shift2)  # (8,4,4) int8
        flat = c2_q.flatten()  # 128 values
        with open(f'golden_hex_fc/fc_input_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(flat, 8)) + '\n')
        with open(f'golden_hex_fc/fc_logits_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['logits'][i], 32)) + '\n')
        pred = int(np.argmax(gv['logits'][i]))
        correct += int(pred == gv['labels'][i])

    print(f"FC-layer end-to-end accuracy check: {correct}/{N} = {correct/N:.4f}")

    # Weight export: w3 shape (128,10) -> tap-major (tap=input index 0..127), 10 channels packed per tap
    w3 = q['w3']
    TAPS, NUM_CH = w3.shape
    lines = []
    for t in range(TAPS):
        word = 0
        for co in range(NUM_CH):
            byte = int(w3[t, co]) & 0xFF
            word |= (byte << (8 * co))
        lines.append(f'{word:020x}')  # 10 channels x 8 bits = 80 bits = 20 hex chars
    os.makedirs('weights_hex_fc', exist_ok=True)
    with open('weights_hex_fc/w3_tapmajor.hex', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Fixed bias b3 (10 x int32) -- recompute canonically (fixed scales end-to-end)
    d = np.load('model_fp32.npz')
    x_scale = float(np.max(np.abs(np.load('test_data.npz')['X_test'][:20]))) / 127.0
    w1_scale = float(q['w1_scale']); w2_scale = float(q['w2_scale']); w3_scale = float(q['w3_scale'])
    with open('requant_shifts.txt') as f:
        shifts2 = dict(line.strip().split('=') for line in f if line.strip())
    shift1 = int(shifts2['shift1'])
    c1_out_scale = x_scale * w1_scale * (2 ** shift1)
    c2_out_scale = c1_out_scale * w2_scale * (2 ** shift2)
    b3_q = np.round(d['b3'] / (c2_out_scale * w3_scale)).astype(np.int32)
    word = 0
    for co in range(NUM_CH):
        word |= (int(b3_q[co]) & 0xFFFFFFFF) << (32 * co)
    with open('weights_hex_fc/b3.hex', 'w') as f:
        f.write(f'{word:080x}\n')  # 10 x 32-bit = 320 bits = 80 hex chars

    print(f"Exported {N} FC golden samples, w3_tapmajor.hex ({TAPS} taps), b3.hex")
    print(f"b3_q = {b3_q.tolist()}")

if __name__ == '__main__':
    main()
