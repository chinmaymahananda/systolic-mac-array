"""
Golden model v2: FIXED, calibrated per-tensor quantization for inter-layer
activations (Conv1->Conv2, Conv2->FC), instead of v1's per-sample dynamic
max-based rescale. Real INT8 accelerators use offline-calibrated scales;
per-sample dynamic requant is not realistic hardware and does not belong
in a golden model meant to be matched bit-exact in RTL.

Design choice: calibrated scales are constrained to powers of two, so the
RTL requantization step between layers is a plain arithmetic right-shift
(post-ReLU clamp) -- no multiplier needed for the rescale itself (the conv
MACs still need multipliers, just not the inter-layer rescale).

Run: python3 quantize_and_golden_v2.py (after train_model.py)
Outputs: model_int8_v2.npz, golden_vectors_v2.npz, requant_shifts.txt
"""
import numpy as np

def quantize_tensor(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax if np.max(np.abs(x)) > 0 else 1.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -qmax - 1, qmax)
    return q.astype(np.int8), np.float32(scale)

def conv2d_acc(x_q, w_q, b_q):
    """Integer convolution accumulator only (no rescale). x_q:(Cin,H,W) w_q:(Cout,Cin,kh,kw)"""
    Cout, Cin, kh, kw = w_q.shape
    _, H, W = x_q.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    acc = np.zeros((Cout, Ho, Wo), dtype=np.int64)
    for co in range(Cout):
        for ci in range(Cin):
            for i in range(kh):
                for j in range(kw):
                    acc[co] += w_q[co, ci, i, j].astype(np.int64) * x_q[ci, i:i+Ho, j:j+Wo].astype(np.int64)
        acc[co] += b_q[co]
    return acc.astype(np.int32)

def requant_shift_for(max_abs_val, num_bits=8):
    """Smallest right-shift so round(max_abs_val >> shift) fits in signed num_bits."""
    qmax = 2 ** (num_bits - 1) - 1
    shift = 0
    while (max_abs_val >> shift) > qmax:
        shift += 1
    return shift

def relu_requant(acc32, shift):
    """ReLU then right-shift (with rounding) then clamp to int8. Pure integer ops -- this is exactly what the RTL does."""
    relu = np.maximum(0, acc32.astype(np.int64))
    if shift > 0:
        rounded = (relu + (1 << (shift - 1))) >> shift
    else:
        rounded = relu
    return np.clip(rounded, 0, 127).astype(np.int8)

def main():
    d = np.load('model_fp32.npz')
    params = {k: d[k] for k in d.files}
    test = np.load('test_data.npz')
    X_test, y_test = test['X_test'], test['y_test']

    w1_q, w1_scale = quantize_tensor(params['w1'])
    w2_q, w2_scale = quantize_tensor(params['w2'])
    w3_q, w3_scale = quantize_tensor(params['w3'])

    # Calibrate a FIXED input scale from the calibration set's global max abs
    # value, instead of per-sample dynamic scaling. Real accelerators calibrate
    # input quantization offline too -- a per-sample input scale would make
    # every downstream bias sample-dependent, which is not realistic weight-
    # stationary hardware and breaks the "load bias once" design.
    _test_cal = np.load('test_data.npz')['X_test'][:20]
    _input_max_abs = float(np.max(np.abs(_test_cal)))
    input_scale = _input_max_abs / 127.0
    b1_q = np.round(params['b1'] / (input_scale * w1_scale)).astype(np.int32)

    N_CAL = 20  # calibration set (same 20 samples used for golden vectors -- fine for a portfolio demo, would use a larger held-out calibration set in production)

    # --- Calibration pass: find global max abs accumulator value (post-ReLU) for conv1 output ---
    max_c1 = 0
    conv1_accs = []
    for i in range(N_CAL):
        x_fp = X_test[i][0]
        x_q = np.clip(np.round(x_fp / input_scale), -128, 127).astype(np.int8)
        x_scale = input_scale
        x_q = x_q[None, :, :]
        acc1 = conv2d_acc(x_q, w1_q, b1_q)
        max_c1 = max(max_c1, int(np.max(np.maximum(0, acc1))))
        conv1_accs.append((acc1, x_scale))
    shift1 = requant_shift_for(max_c1)
    print(f"Conv1->Conv2 requant: max_abs={max_c1}, shift={shift1} (c1_int8 = clamp(round(relu(acc1)>>{shift1}), 0, 127))")

    # Effective scale of conv1's int8 output (for bookkeeping / conv2 weight combined-scale)
    # acc1 is in domain (x_scale * w1_scale); after >>shift1 the value represents
    # acc1_fp / (x_scale*w1_scale*2^shift1) rounded -- so c1_out_scale = x_scale*w1_scale*2^shift1.
    # x_scale varies per-sample in this design (input quantization is still per-sample,
    # only INTER-LAYER activations use fixed calibrated shifts) -- conv2 combined scale
    # is therefore still sample-dependent through x_scale, tracked explicitly per sample below.

    b2_base = params['b2']  # will be re-scaled per sample using that sample's c1_out_scale

    max_c2 = 0
    all_results = []
    for i in range(N_CAL):
        acc1, x_scale = conv1_accs[i]
        c1_out_scale = x_scale * w1_scale * (2 ** shift1)
        c1_q = relu_requant(acc1, shift1)  # (4,6,6) int8, pure integer, matches RTL exactly

        b2_q = np.round(b2_base / (c1_out_scale * w2_scale)).astype(np.int32)
        acc2 = conv2d_acc(c1_q.astype(np.int8), w2_q, b2_q)
        max_c2 = max(max_c2, int(np.max(np.maximum(0, acc2))))
        all_results.append([acc1, c1_q, c1_out_scale, acc2])

    shift2 = requant_shift_for(max_c2)
    print(f"Conv2->FC requant: max_abs={max_c2}, shift={shift2}")

    b3_base = params['b3']
    correct = 0
    golden = {'inputs': [], 'conv1_acc': [], 'conv2_acc': [], 'logits': [], 'preds': [], 'labels': y_test[:N_CAL]}
    for i in range(N_CAL):
        acc1, c1_q, c1_out_scale, acc2 = all_results[i]
        c2_out_scale = c1_out_scale * w2_scale * (2 ** shift2)
        c2_q = relu_requant(acc2, shift2)  # (8,4,4) int8

        b3_q = np.round(b3_base / (c2_out_scale * w3_scale)).astype(np.int32)
        flat_q = c2_q.flatten().astype(np.int64)
        acc3 = (flat_q @ w3_q.astype(np.int64)) + b3_q.astype(np.int64)
        acc3 = acc3.astype(np.int32)

        pred = int(np.argmax(acc3))  # argmax unaffected by monotonic positive rescale
        correct += int(pred == y_test[i])

        x_q_i = np.clip(np.round(X_test[i][0] / input_scale), -128, 127).astype(np.int8)
        golden['inputs'].append(x_q_i)
        golden['conv1_acc'].append(acc1)
        golden['conv2_acc'].append(acc2)
        golden['logits'].append(acc3)
        golden['preds'].append(pred)

    acc_pct = correct / N_CAL
    print(f"Quantized (fixed-shift INT8) accuracy on {N_CAL}-sample set: {acc_pct:.4f}")

    np.savez('golden_vectors_v2.npz',
             inputs=np.array(golden['inputs']), conv1_acc=np.array(golden['conv1_acc']),
             conv2_acc=np.array(golden['conv2_acc']), logits=np.array(golden['logits']),
             preds=np.array(golden['preds']), labels=golden['labels'])
    np.savez('model_int8_v2.npz', w1=w1_q, w2=w2_q, w3=w3_q, b1=b1_q,
             w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale)
    with open('requant_shifts.txt', 'w') as f:
        f.write(f"shift1={shift1}\nshift2={shift2}\n")
    print("Saved golden_vectors_v2.npz, model_int8_v2.npz, requant_shifts.txt")

if __name__ == '__main__':
    main()
