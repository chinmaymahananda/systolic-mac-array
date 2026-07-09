// chain_mem.v
// Additive memory primitives for the chained top_accelerator only.
// Original per-layer modules (conv1_mem.v, conv2_mem.v, fc_mem.v) are left
// untouched so the existing standalone tb_conv1/tb_conv2/tb_fc testbenches
// keep working exactly as before.

// True dual-port accumulator RAM: write side driven by the layer's own
// controller (during compute), read side driven externally by the
// requant_bridge (during hand-off between layers). Decoupling the two
// address ports avoids fighting over a single-port RAM's one address bus.
module acc_ram_dp #(
    parameter ACC_WIDTH = 32,
    parameter DEPTH     = 36,
    parameter AW        = 6
)(
    input  wire                         clk,
    input  wire                         wr_en,
    input  wire [AW-1:0]                wr_addr,
    input  wire signed [ACC_WIDTH-1:0]  wr_data,
    input  wire [AW-1:0]                rd_addr,
    output reg  signed [ACC_WIDTH-1:0]  rd_data
);
    reg signed [ACC_WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule


// Writable image RAM: same registered-read behavior as img_rom_c2/img_rom_fc,
// but with an added write port so requant_bridge can load it from the
// previous layer''s output. Always populated by a bridge in the chained
// design -- no $readmemh here.
module img_ram_w #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 144,
    parameter AW         = 8
)(
    input  wire                          clk,
    input  wire                          wr_en,
    input  wire [AW-1:0]                 wr_addr,
    input  wire signed [DATA_WIDTH-1:0]  wr_data,
    input  wire [AW-1:0]                 rd_addr,
    output reg  signed [DATA_WIDTH-1:0]  rd_data
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule
