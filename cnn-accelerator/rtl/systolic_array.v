// systolic_array.v
// N x N systolic array of MAC units, weight-stationary style:
//   - Weights are pre-loaded into each PE and held for the duration of one
//     output-channel's convolution pass.
//   - Activations stream in from the left, partial sums accumulate locally,
//     row of outputs is read out along the bottom when acc_valid asserts.
//
// This directly extends the original parameterizable MAC array concept
// (chinmaymahananda/systolic-mac-array) into a 2D array wired for conv use:
// each PE(row,col) computes one (output_channel, output_pixel) accumulator
// depending on how the controller streams data in.

module systolic_array #(
    parameter N          = 4,   // NxN array
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                                   clk,
    input  wire                                   rst_n,
    input  wire                                   en,
    input  wire                                   acc_clear,
    input  wire signed [N*DATA_WIDTH-1:0]         a_row_in,   // N activation inputs, one per row
    input  wire signed [N*DATA_WIDTH-1:0]         w_col_in,   // N weight inputs, one per column
    output wire signed [N*N*ACC_WIDTH-1:0]        acc_flat    // flattened NxN accumulator outputs
);

    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : ROW
            for (c = 0; c < N; c = c + 1) begin : COL
                wire signed [DATA_WIDTH-1:0] a_local = a_row_in[(r+1)*DATA_WIDTH-1 : r*DATA_WIDTH];
                wire signed [DATA_WIDTH-1:0] w_local = w_col_in[(c+1)*DATA_WIDTH-1 : c*DATA_WIDTH];
                wire signed [ACC_WIDTH-1:0]  acc_local;

                mac_unit #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .en(en),
                    .acc_clear(acc_clear),
                    .a_in(a_local),
                    .w_in(w_local),
                    .acc_out(acc_local)
                );

                assign acc_flat[((r*N+c)+1)*ACC_WIDTH-1 : (r*N+c)*ACC_WIDTH] = acc_local;
            end
        end
    endgenerate

endmodule
