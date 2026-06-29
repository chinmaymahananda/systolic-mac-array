// ---------------------------------------------------------------------------
// tb_systolic.v  -  Self-checking testbench for systolic_array
//
// For each random test it:
//   1. fills N x N operand matrices A, B with random signed values,
//   2. computes a golden C = A x B in the testbench,
//   3. streams A and B into the DUT with the correct diagonal skew,
//   4. reads every PE accumulator and compares against the golden result,
//   5. reports PASS / FAIL and exits with a non-zero status on any mismatch.
//
// Run (Icarus Verilog):
//   iverilog -g2012 -o sim/tb rtl/pe.v rtl/systolic_array.v tb/tb_systolic.v
//   vvp sim/tb
//   gtkwave sim/dump.vcd   (optional)
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

module tb_systolic;

    localparam integer N          = 4;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;
    localparam integer NUM_TESTS  = 20;

    integer t, i, j, k, test, errors, total_errors;
    integer seed;

    reg                       clk, rst_n, en, clear;
    reg  [N*DATA_WIDTH-1:0]   a_in_flat, b_in_flat;
    wire [N*N*ACC_WIDTH-1:0]  c_out_flat;

    // operand + golden storage (signed)
    reg signed [DATA_WIDTH-1:0] A [0:N-1][0:N-1];
    reg signed [DATA_WIDTH-1:0] B [0:N-1][0:N-1];
    reg signed [ACC_WIDTH-1:0]  C_gold [0:N-1][0:N-1];
    reg signed [ACC_WIDTH-1:0]  c_hw;

    // DUT
    systolic_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clear(clear),
        .a_in_flat(a_in_flat), .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );

    // 100 MHz clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Generate one random operand pair and its golden product.
    // -----------------------------------------------------------------------
    task gen_operands;
        begin
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    A[i][j] = $random(seed) % 16;   // small range keeps values readable
                    B[i][j] = $random(seed) % 16;
                end
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    C_gold[i][j] = 0;
                    for (k = 0; k < N; k = k + 1)
                        C_gold[i][j] = C_gold[i][j] + A[i][k] * B[k][j];
                end
        end
    endtask

    // -----------------------------------------------------------------------
    // Drive the diagonally-skewed feed for `cyc`, then return zeros.
    //   a-feed for row i at cycle t = A[i][t-i]  when 0 <= t-i < N, else 0
    //   b-feed for col j at cycle t = B[t-j][j]  when 0 <= t-j < N, else 0
    // -----------------------------------------------------------------------
    task drive_cycle(input integer cyc);
        integer idx;
        begin
            a_in_flat = {N*DATA_WIDTH{1'b0}};
            b_in_flat = {N*DATA_WIDTH{1'b0}};
            for (i = 0; i < N; i = i + 1) begin
                idx = cyc - i;
                if (idx >= 0 && idx < N)
                    a_in_flat[i*DATA_WIDTH +: DATA_WIDTH] = A[i][idx];
            end
            for (j = 0; j < N; j = j + 1) begin
                idx = cyc - j;
                if (idx >= 0 && idx < N)
                    b_in_flat[j*DATA_WIDTH +: DATA_WIDTH] = B[idx][j];
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Run one full matrix multiply through the DUT and check the result.
    // -----------------------------------------------------------------------
    task run_test(input integer tnum);
        begin
            errors = 0;

            // reset + clear accumulators
            rst_n = 1'b0; en = 1'b0; clear = 1'b0;
            a_in_flat = 0; b_in_flat = 0;
            @(negedge clk); @(negedge clk);
            rst_n = 1'b1; en = 1'b1; clear = 1'b0;

            // stream operands with skew; valid feed window is 0 .. 2N-2,
            // then keep clocking so every diagonal product is accumulated.
            for (t = 0; t < 4*N; t = t + 1) begin
                @(negedge clk);
                drive_cycle(t);
            end
            // let the final accumulations settle
            repeat (4) @(negedge clk);

            // compare every PE accumulator to the golden matrix
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    c_hw = c_out_flat[(i*N + j)*ACC_WIDTH +: ACC_WIDTH];
                    if (c_hw !== C_gold[i][j]) begin
                        errors = errors + 1;
                        $display("  MISMATCH test %0d C[%0d][%0d]: hw=%0d gold=%0d",
                                 tnum, i, j, c_hw, C_gold[i][j]);
                    end
                end

            if (errors == 0)
                $display("  test %0d: PASS", tnum);
            else begin
                $display("  test %0d: FAIL (%0d mismatches)", tnum, errors);
                total_errors = total_errors + errors;
            end
        end
    endtask

    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("sim/dump.vcd");
        $dumpvars(0, tb_systolic);

        seed         = 32'hC0FFEE;
        total_errors = 0;

        $display("=== systolic_array self-check : %0dx%0d, DATA_WIDTH=%0d ===",
                 N, N, DATA_WIDTH);

        for (test = 1; test <= NUM_TESTS; test = test + 1) begin
            gen_operands;
            run_test(test);
        end

        $display("================================================");
        if (total_errors == 0)
            $display("ALL %0d TESTS PASSED", NUM_TESTS);
        else
            $display("FAILED: %0d total mismatches", total_errors);
        $display("================================================");

        if (total_errors != 0) $fatal;
        $finish;
    end

endmodule

`default_nettype wire
