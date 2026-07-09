import numpy as np

acc = [int(l.strip(), 16) for l in open("golden_hex/conv1_acc_13.hex")]
acc = [v - 0x100000000 if v > 0x7FFFFFFF else v for v in acc]

golden = [int(l.strip(), 16) for l in open("golden_hex_v2/conv2_input_13.hex")]

mismatches = 0
for i in range(144):
    relu = max(acc[i], 0)
    req = (relu + 64) >> 7
    req = min(req, 127)
    if req != golden[i]:
        mismatches += 1
        print(f"idx={i} (ch={i//36} pix={i%36}): recomputed={req} golden_file={golden[i]}")

print(f"\ntotal mismatches vs golden_hex_v2/conv2_input_13.hex: {mismatches}/144")
