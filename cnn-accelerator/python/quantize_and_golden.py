"""
Quantize the trained FP32 model to INT8 (symmetric per-tensor quantization)
and generate a golden reference (bit-accurate INT32 accumulator simulation)
that the RTL must match exactly.

Run: python3 quantize_and_golden.py
Outputs:
  model_int8.npz        - quantized weights + scales
  golden_vectors.npz    - input/output test vectors for RTL testbench
  weights_hex/*.hex     - weight files formatted for $readmemh in Verilog
"""
import numpy as np
import os

def quantize_tensor(x, num_bits=8):
    """Symmetric per-tensor quantization to signed int8."""
    qmax = 2 ** (num_bits - 1) - 1  # 127
    scale = np.max(np.abs(x)) / qmax if np.max(np.abs(x)) > 0 else 1.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -qmax - 1, qmax)
    return q.astype(np.int8), np.float32(scale)

def relu_i32(x):
    return np.maximum(0, x)

def conv2d_int(x_q, w_q, b_q, x_scale, w_scale, out_bits=8):
    """
    Integer convolution matching what the RTL systolic array computes.
    x_q, w_q: int8 (already quantized)
    Accumulation done in int32 (matches MAC array accumulator width).
    Returns: int32 pre-activation accumulator (the RTL must match this exactly),
             plus re-quantized int8 output + its scale for the next layer.
    """
    Cout, Cin, kh, kw = w_q.shape
    _, H, W = x_q.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    acc = np.zeros((Cout, Ho, Wo), dtype=np.int64)  # int64 headroom for sim only
    for co in range(Cout):
        for ci in range(Cin):
            for i in range(kh):
                for j in range(kw):
                    acc[co] += w_q[co, ci, i, j].astype(np.int64) * \
                               x_q[ci, i:i+Ho, j:j+Wo].astype(np.int64)
        acc[co] += b_q[co]
    acc32 = acc.astype(np.int32)  # this exact value is what RTL accumulator must produce

    combined_scale = x_scale * w_scale
    fp_out = acc32.astype(np.float32) * combined_scale
    relu_out = relu_i32(fp_out)
    out_q, out_scale = quantize_tensor(relu_out, out_bits)
    return acc32, out_q, out_scale

def fc_int(x_q, w_q, b_q, x_scale, w_scale):
    acc = (x_q.astype(np.int64) @ w_q.astype(np.int64)) + b_q.astype(np.int64)
    acc32 = acc.astype(np.int32)
    combined_scale = x_scale * w_scale
    fp_out = acc32.astype(np.float32) * combined_scale
    return acc32, fp_out

def main():
    d = np.load('model_fp32.npz')
    params = {k: d[k] for k in d.files}

    q_params = {}
    for name in ['w1', 'w2', 'w3']:
        q, s = quantize_tensor(params[name])
        q_params[name] = q
        q_params[name + '_scale'] = s
    # biases quantized with input_scale * weight_scale (standard practice);
    # for this small demo we keep biases in int32 pre-scaled to accumulator domain.
    input_scale = 1.0 / 127.0  # inputs already normalized to [0,1]->[0,127] below
    q_params['b1'] = np.round(params['b1'] / (input_scale * q_params['w1_scale'])).astype(np.int32)
    q_params['b2_scale_placeholder'] = 0  # filled after conv1 scale known dynamically per-sample in sim
    np.savez('model_int8.npz', **q_params)

    test = np.load('test_data.npz')
    X_test, y_test = test['X_test'], test['y_test']

    os.makedirs('weights_hex', exist_ok=True)
    def dump_hex(name, arr_int8):
        flat = arr_int8.flatten().astype(np.int32)  # widen before masking to avoid int8 overflow
        with open(f'weights_hex/{name}.hex', 'w') as f:
            for v in flat:
                f.write(f'{int(v) & 0xFF:02x}\n')  # two's complement hex byte

    dump_hex('w1_int8', q_params['w1'])
    dump_hex('w2_int8', q_params['w2'])
    dump_hex('w3_int8', q_params['w3'])

    # Build golden vectors for N test samples end-to-end (quantize input per-sample)
    N = 20
    golden_inputs, golden_conv1_acc, golden_conv2_acc, golden_logits, golden_preds = [], [], [], [], []
    correct = 0
    for i in range(N):
        x_fp = X_test[i][0]  # (8,8)
        x_q, x_scale = quantize_tensor(x_fp)
        x_q = x_q[None, :, :]  # (1,8,8)

        acc1, c1_q, c1_scale = conv2d_int(x_q, q_params['w1'], q_params['b1'], x_scale, q_params['w1_scale'])
        b2 = np.round(params['b2'] / (c1_scale * q_params['w2_scale'])).astype(np.int32)
        acc2, c2_q, c2_scale = conv2d_int(c1_q, q_params['w2'], b2, c1_scale, q_params['w2_scale'])

        flat_q = c2_q.flatten()
        b3 = np.round(params['b3'] / (c2_scale * q_params['w3_scale'])).astype(np.int32)
        acc3, logits_fp = fc_int(flat_q, q_params['w3'], b3, c2_scale, q_params['w3_scale'])

        pred = int(np.argmax(logits_fp))
        correct += int(pred == y_test[i])

        golden_inputs.append(x_q[0])
        golden_conv1_acc.append(acc1)
        golden_conv2_acc.append(acc2)
        golden_logits.append(acc3)
        golden_preds.append(pred)

    quant_acc = correct / N
    print(f"Quantized (INT8) accuracy on {N}-sample golden set: {quant_acc:.4f}")

    np.savez('golden_vectors.npz',
             inputs=np.array(golden_inputs),
             conv1_acc=np.array(golden_conv1_acc),
             conv2_acc=np.array(golden_conv2_acc),
             logits=np.array(golden_logits),
             preds=np.array(golden_preds),
             labels=y_test[:N])
    print("Saved golden_vectors.npz, model_int8.npz, weights_hex/*.hex")

if __name__ == '__main__':
    main()
