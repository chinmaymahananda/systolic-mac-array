// ---------------------------------------------------------------------------
// systolic_array.v  -  N x N output-stationary systolic matrix-multiply engine
//
// Computes C = A x B for N x N operands (contraction dimension K = N).
//
//   * Each row i of A is streamed in from the LEFT  (a_in_flat[i]).
//   * Each col j of B is streamed in from the TOP   (b_in_flat[j]).
//   * Operands are skewed (staggered) by the driver so that A[i][k] and
//     B[k][j] arrive at PE(i,j) on the same cycle. Outside the valid window
//     the driver feeds zeros, which contribute nothing to the MAC, so `en`
//     can be held high for the whole run.
//   * After ~3N cycles the accumulator in PE(i,j) holds C[i][j]. The full
//     result is exposed on c_out_flat with PE(i,j) at index (i*N + j).
//
// Fully parameterizable in array size and operand/accumulator width. Built
// from generate loops so it scales to any N without hand-editing.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

module systolic_array #(
    parameter integer N          = 4,    // array is N x N (matrices are N x N)
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32
) (
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             en,
    input  wire                             clear,
    input  wire [N*DATA_WIDTH-1:0]          a_in_flat,   // a row-feed,  i -> [i*DW +: DW]
    input  wire [N*DATA_WIDTH-1:0]          b_in_flat,   // b col-feed,  j -> [j*DW +: DW]
    output wire [N*N*ACC_WIDTH-1:0]         c_out_flat   // result, (i*N+j) -> [.. +: ACC_WIDTH]
);

    genvar i, j;

    // Inter-PE operand nets.
    //   a_h[i][j]   feeds the a-input of PE(i,j); a_h[i][0] is the external feed.
    //   b_v[i][j]   feeds the b-input of PE(i,j); b_v[0][j] is the external feed.
    wire signed [DATA_WIDTH-1:0] a_h [0:N-1][0:N];
    wire signed [DATA_WIDTH-1:0] b_v [0:N][0:N-1];

    // Hook external feeds onto the west/north edges.
    generate
        for (i = 0; i < N; i = i + 1) begin : g_west
            assign a_h[i][0] = a_in_flat[i*DATA_WIDTH +: DATA_WIDTH];
        end
        for (j = 0; j < N; j = j + 1) begin : g_north
            assign b_v[0][j] = b_in_flat[j*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // The PE grid.
    generate
        for (i = 0; i < N; i = i + 1) begin : g_row
            for (j = 0; j < N; j = j + 1) begin : g_col
                wire signed [ACC_WIDTH-1:0] c_ij;

                pe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .en     (en),
                    .clear  (clear),
                    .a_in   (a_h[i][j]),
                    .b_in   (b_v[i][j]),
                    .a_out  (a_h[i][j+1]),   // march east
                    .b_out  (b_v[i+1][j]),   // march south
                    .c_out  (c_ij)
                );

                assign c_out_flat[(i*N + j)*ACC_WIDTH +: ACC_WIDTH] = c_ij;
            end
        end
    endgenerate

endmodule

`default_nettype wire
