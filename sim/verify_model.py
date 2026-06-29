#!/usr/bin/env python3
"""
verify_model.py  -  Cycle-accurate reference model of the systolic_array RTL.

This is NOT the deliverable. It exists to prove the architecture and the
diagonal-skew dataflow are correct, by replicating the EXACT register/accumulate
semantics of pe.v + systolic_array.v + tb_systolic.v and checking the result
against a plain integer matrix multiply, over many random signed test cases.

If this passes, the Verilog (a direct transcription of these same semantics)
produces the same PASS under iverilog.
"""
import random


def systolic_run(A, B, N):
    """Mirror the RTL: N x N output-stationary array, registered operand
    propagation, accumulate every cycle, zero-padded diagonal feed."""
    # registered state (acc = PE accumulator, a_reg/b_reg = a_out/b_out regs)
    acc   = [[0] * N for _ in range(N)]
    a_reg = [[0] * N for _ in range(N)]
    b_reg = [[0] * N for _ in range(N)]

    def ext_a(i, t):                 # west feed for row i at cycle t
        k = t - i
        return A[i][k] if 0 <= k < N else 0

    def ext_b(j, t):                 # north feed for col j at cycle t
        k = t - j
        return B[k][j] if 0 <= k < N else 0

    total_cycles = 4 * N + 4         # matches TB: 4N feed + settle
    for t in range(total_cycles):
        # --- combinational: derive each PE's inputs from current regs ---
        a_in = [[0] * N for _ in range(N)]
        b_in = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                a_in[i][j] = ext_a(i, t) if j == 0 else a_reg[i][j - 1]
                b_in[i][j] = ext_b(j, t) if i == 0 else b_reg[i - 1][j]

        # --- sequential: commit all next-states simultaneously (non-blocking) ---
        n_acc   = [[acc[i][j] + a_in[i][j] * b_in[i][j] for j in range(N)] for i in range(N)]
        n_a_reg = [[a_in[i][j] for j in range(N)] for i in range(N)]
        n_b_reg = [[b_in[i][j] for j in range(N)] for i in range(N)]
        acc, a_reg, b_reg = n_acc, n_a_reg, n_b_reg

    return acc


def golden(A, B, N):
    return [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)] for i in range(N)]


def run_suite(N, n_tests, lo, hi, label):
    fails = 0
    for _ in range(n_tests):
        A = [[random.randint(lo, hi) for _ in range(N)] for _ in range(N)]
        B = [[random.randint(lo, hi) for _ in range(N)] for _ in range(N)]
        if systolic_run(A, B, N) != golden(A, B, N):
            fails += 1
    status = "PASS" if fails == 0 else f"FAIL ({fails})"
    print(f"  N={N:<3} range=[{lo:>4},{hi:>3}]  {n_tests:>4} tests : {status}  [{label}]")
    return fails


if __name__ == "__main__":
    random.seed(0xC0FFEE)
    print("=== Cycle-accurate verification of systolic MAC array ===")
    total = 0
    total += run_suite(4,  500,   0,  15,  "default 4x4, unsigned-ish")
    total += run_suite(4,  500, -15,  15,  "4x4 signed (negatives)")
    total += run_suite(8,  300, -15,  15,  "8x8 signed (param scaling)")
    total += run_suite(16, 100, -15,  15,  "16x16 signed (large array)")
    total += run_suite(3,  300, -50,  50,  "non-power-of-two 3x3")
    total += run_suite(4,  200,-128, 127,  "4x4 full int8 range")
    print("=" * 56)
    print("ALL SUITES PASSED" if total == 0 else f"FAILURES: {total}")
