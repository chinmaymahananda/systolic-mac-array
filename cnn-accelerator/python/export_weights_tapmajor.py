"""
The RTL conv1_controller reads weights tap-major (for tap t=0..8, it wants
all 4 channels' weight for that tap packed into one 32-bit word), whereas
model_int8.npz stores weights channel-major (shape (4,1,3,3)).

This script reorders w1 into tap-major layout and emits:
  weights_hex/w1_tapmajor.hex  - 9 lines, each a 32-bit hex word =
                                  {ch3_w[t], ch2_w[t], ch1_w[t], ch0_w[t]}
  weights_hex/b1.hex           - 1 line, 128-bit hex word = 4 packed int32 biases

Run: python3 export_weights_tapmajor.py (after quantize_and_golden.py)
"""
import numpy as np

def main():
    q = np.load('model_int8.npz')
    w1 = q['w1']  # shape (4,1,3,3) int8
    Cout, Cin, kh, kw = w1.shape
    assert Cin == 1

    lines = []
    for ky in range(kh):
        for kx in range(kw):
            word = 0
            for co in range(Cout):
                byte = int(w1[co, 0, ky, kx]) & 0xFF
                word |= (byte << (8 * co))
            lines.append(f'{word:08x}')
    with open('weights_hex/w1_tapmajor.hex', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # bias: need int32 bias-in-accumulator-domain, same as used in golden script
    d = np.load('model_fp32.npz')
    input_scale = 1.0 / 127.0
    w1_scale = float(q['w1_scale'])
    b1 = np.round(d['b1'] / (input_scale * w1_scale)).astype(np.int64)
    word = 0
    for co in range(Cout):
        val = int(b1[co]) & 0xFFFFFFFF
        word |= (val << (32 * co))
    with open('weights_hex/b1.hex', 'w') as f:
        f.write(f'{word:032x}\n')

    print("Wrote weights_hex/w1_tapmajor.hex and b1.hex")

if __name__ == '__main__':
    main()
