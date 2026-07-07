// mac_unit.v
// Single INT8 x INT8 -> INT32 accumulate MAC element.
// This is the atomic building block of the systolic array: each PE holds
// one accumulator and performs one multiply-add per cycle when enabled.

module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,        // pipeline enable
    input  wire                         acc_clear, // clear accumulator (start of new output pixel)
    input  wire signed [DATA_WIDTH-1:0] a_in,       // activation/input operand
    input  wire signed [DATA_WIDTH-1:0] w_in,       // weight operand
    output reg  signed [ACC_WIDTH-1:0]  acc_out
);

    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = a_in * w_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= {ACC_WIDTH{1'b0}};
        end else if (acc_clear) begin
            acc_out <= {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
        end else if (en) begin
            acc_out <= acc_out + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
        end
    end

endmodule
