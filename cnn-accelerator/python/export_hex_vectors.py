"""
Export golden_vectors.npz into per-sample hex files the Verilog testbench can
read with $readmemh, so the RTL simulation can compare cycle-accurate
accumulator outputs against this exact golden reference.

Run: python3 export_hex_vectors.py
Outputs: golden_hex/input_N.hex, golden_hex/conv1_acc_N.hex,
         golden_hex/conv2_acc_N.hex, golden_hex/logits_N.hex
"""
import numpy as np
import os

def to_hex_lines(arr, bits):
    """Two's complement hex, one value per line, width = bits/4 hex chars."""
    hexchars = bits // 4
    mask = (1 << bits) - 1
    lines = []
    for v in arr.flatten():
        lines.append(f'{int(v) & mask:0{hexchars}x}')
    return lines

def main():
    gv = np.load('golden_vectors.npz')
    os.makedirs('golden_hex', exist_ok=True)
    N = gv['inputs'].shape[0]
    for i in range(N):
        with open(f'golden_hex/input_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['inputs'][i], 8)) + '\n')
        with open(f'golden_hex/conv1_acc_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['conv1_acc'][i], 32)) + '\n')
        with open(f'golden_hex/conv2_acc_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['conv2_acc'][i], 32)) + '\n')
        with open(f'golden_hex/logits_{i}.hex', 'w') as f:
            f.write('\n'.join(to_hex_lines(gv['logits'][i], 32)) + '\n')
    with open('golden_hex/labels_preds.txt', 'w') as f:
        for i in range(N):
            f.write(f"{i} label={int(gv['labels'][i])} pred={int(gv['preds'][i])}\n")
    print(f"Exported {N} golden samples to golden_hex/")

if __name__ == '__main__':
    main()
