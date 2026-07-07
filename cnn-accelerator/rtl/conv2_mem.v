// conv2_mem.v -- memory blocks for Conv2 (8 output channels, 4 input channels)

module img_rom_c2 #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 144  // 4 channels x 36 pixels (6x6)
)(
    input  wire                          clk,
    input  wire [7:0]                    addr,
    output reg  signed [DATA_WIDTH-1:0]  data
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh("golden_hex_v2/conv2_input_0.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule


module wgt_rom_c2 #(
    parameter NUM_CH     = 8,
    parameter DATA_WIDTH = 8,
    parameter TAPS       = 36
)(
    input  wire                                clk,
    input  wire [5:0]                          addr,
    output reg  signed [NUM_CH*DATA_WIDTH-1:0]  data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex_v2/w2_tapmajor.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule


module bias_rom_c2 #(
    parameter NUM_CH    = 8,
    parameter ACC_WIDTH = 32
)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex_v2/b2.hex", mem);
    assign data = mem[0];
endmodule


module out_acc_ram_c2 #(
    parameter ACC_WIDTH = 32,
    parameter DEPTH     = 16  // 4x4 output pixels
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
