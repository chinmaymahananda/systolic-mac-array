// conv1_mem.v
// Simple synthesizable ROM/RAM blocks for the Conv1 accelerator.
// $readmemh is used for simulation; on real FPGA synthesis these infer
// to BRAM automatically in Vivado when written this way (single-port,
// registered read).

module img_rom #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 64
)(
    input  wire                          clk,
    input  wire [6:0]                    addr,
    output reg  signed [DATA_WIDTH-1:0]  data
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh("golden_hex/input_0.hex", mem); // overridden per-sample by testbench $readmemh

    always @(posedge clk) data <= mem[addr];
endmodule


module wgt_rom_tapmajor #(
    parameter NUM_CH     = 4,
    parameter DATA_WIDTH = 8,
    parameter TAPS       = 9
)(
    input  wire                                clk,
    input  wire [5:0]                          addr, // tap index 0..8
    output reg  signed [NUM_CH*DATA_WIDTH-1:0]  data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex/w1_tapmajor.hex", mem);

    always @(posedge clk) data <= mem[addr];
endmodule


module bias_rom #(
    parameter NUM_CH    = 4,
    parameter ACC_WIDTH = 32
)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex/b1.hex", mem);
    assign data = mem[0];
endmodule


module out_acc_ram #(
    parameter ACC_WIDTH = 32,
    parameter DEPTH     = 36  // 6x6 output pixels, one bank per channel
)(
    input  wire                         clk,
    input  wire                         wr_en,
    input  wire [7:0]                   addr,
    input  wire signed [ACC_WIDTH-1:0]  wr_data,
    output reg  signed [ACC_WIDTH-1:0]  rd_data
);
    reg signed [ACC_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en) mem[addr] <= wr_data;
        rd_data <= mem[addr];
    end
endmodule
