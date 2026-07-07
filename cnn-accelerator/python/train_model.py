"""
Train a tiny CNN (LeNet-scale) on the sklearn 'digits' dataset (8x8 grayscale, 10 classes).
No internet required — dataset ships with sklearn.

This is deliberately small so it maps cleanly onto a systolic MAC array:
  Conv1: 1x8x8 input -> 4 filters, 3x3 kernel -> 4x6x6
  Conv2: 4x6x6 -> 8 filters, 3x3 kernel -> 8x4x4
  FC:    8*4*4=128 -> 10 classes

Run: python3 train_model.py
Outputs: model_fp32.npz (weights), verifies test accuracy.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def conv2d(x, w, b):
    # x: (Cin, H, W), w: (Cout, Cin, kh, kw), b: (Cout,)
    Cout, Cin, kh, kw = w.shape
    _, H, W = x.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    out = np.zeros((Cout, Ho, Wo))
    for co in range(Cout):
        acc = np.zeros((Ho, Wo))
        for ci in range(Cin):
            for i in range(kh):
                for j in range(kw):
                    acc += w[co, ci, i, j] * x[ci, i:i+Ho, j:j+Wo]
        out[co] = acc + b[co]
    return out

def forward(x, params):
    c1 = relu(conv2d(x, params['w1'], params['b1']))          # (4,6,6)
    c2 = relu(conv2d(c1, params['w2'], params['b2']))         # (8,4,4)
    flat = c2.reshape(-1)                                       # (128,)
    logits = flat @ params['w3'] + params['b3']                # (10,)
    return softmax(logits), (x, c1, c2, flat, logits)

def init_params():
    return {
        'w1': (np.random.randn(4, 1, 3, 3) * 0.3).astype(np.float32),
        'b1': np.zeros(4, dtype=np.float32),
        'w2': (np.random.randn(8, 4, 3, 3) * 0.2).astype(np.float32),
        'b2': np.zeros(8, dtype=np.float32),
        'w3': (np.random.randn(128, 10) * 0.1).astype(np.float32),
        'b3': np.zeros(10, dtype=np.float32),
    }

def backward_and_update(x, y, params, lr):
    probs, cache = forward(x, params)
    x_in, c1, c2, flat, logits = cache
    grad_logits = probs.copy()
    grad_logits[y] -= 1  # dL/dlogits for cross-entropy+softmax

    grad_w3 = np.outer(flat, grad_logits)
    grad_b3 = grad_logits
    grad_flat = params['w3'] @ grad_logits
    grad_c2 = grad_flat.reshape(c2.shape)
    grad_c2[c2 <= 0] = 0  # relu backward

    Cout2, Cin2, kh2, kw2 = params['w2'].shape
    Ho2, Wo2 = c2.shape[1], c2.shape[2]
    grad_w2 = np.zeros_like(params['w2'])
    grad_b2 = grad_c2.sum(axis=(1, 2))
    grad_c1 = np.zeros_like(c1)
    for co in range(Cout2):
        for ci in range(Cin2):
            for i in range(kh2):
                for j in range(kw2):
                    grad_w2[co, ci, i, j] = np.sum(grad_c2[co] * c1[ci, i:i+Ho2, j:j+Wo2])
                    grad_c1[ci, i:i+Ho2, j:j+Wo2] += grad_c2[co] * params['w2'][co, ci, i, j]
    grad_c1[c1 <= 0] = 0

    Cout1, Cin1, kh1, kw1 = params['w1'].shape
    Ho1, Wo1 = c1.shape[1], c1.shape[2]
    grad_w1 = np.zeros_like(params['w1'])
    grad_b1 = grad_c1.sum(axis=(1, 2))
    for co in range(Cout1):
        for ci in range(Cin1):
            for i in range(kh1):
                for j in range(kw1):
                    grad_w1[co, ci, i, j] = np.sum(grad_c1[co] * x_in[ci, i:i+Ho1, j:j+Wo1])

    for name, grad in [('w1', grad_w1), ('b1', grad_b1), ('w2', grad_w2),
                        ('b2', grad_b2), ('w3', grad_w3), ('b3', grad_b3)]:
        params[name] -= lr * grad
    return -np.log(probs[y] + 1e-9)

def main():
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0  # normalize to [0,1], shape (N,8,8)
    y = digits.target
    X = X[:, None, :, :]  # add channel dim -> (N,1,8,8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = init_params()
    lr = 0.05
    epochs = 15
    for ep in range(epochs):
        idx = np.random.permutation(len(X_train))
        total_loss = 0.0
        for i in idx:
            total_loss += backward_and_update(X_train[i], y_train[i], params, lr)
        acc = evaluate(X_test, y_test, params)
        print(f"epoch {ep+1:2d}/{epochs}  loss={total_loss/len(idx):.4f}  test_acc={acc:.4f}")

    np.savez('model_fp32.npz', **params)
    print("Saved model_fp32.npz")
    print(f"Final test accuracy: {evaluate(X_test, y_test, params):.4f}")

    # Save test set for later golden/quantized comparison
    np.savez('test_data.npz', X_test=X_test, y_test=y_test)

def evaluate(X, y, params):
    correct = 0
    for i in range(len(X)):
        probs, _ = forward(X[i], params)
        if np.argmax(probs) == y[i]:
            correct += 1
    return correct / len(X)

if __name__ == '__main__':
    main()
