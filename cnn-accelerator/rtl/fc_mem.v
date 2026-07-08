// fc_mem.v -- memory blocks for FC layer (128 -> 10)

module img_rom_fc #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 128
)(
    input  wire                          clk,
    input  wire [6:0]                    addr,
    output reg  signed [DATA_WIDTH-1:0]  data
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh("golden_hex_fc/fc_input_0.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule


module wgt_rom_fc #(
    parameter NUM_CH     = 10,
    parameter DATA_WIDTH = 8,
    parameter TAPS       = 128
)(
    input  wire                                clk,
    input  wire [6:0]                          addr,
    output reg  signed [NUM_CH*DATA_WIDTH-1:0]  data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex_fc/w3_tapmajor.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule


module bias_rom_fc #(
    parameter NUM_CH    = 10,
    parameter ACC_WIDTH = 32
)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex_fc/b3.hex", mem);
    assign data = mem[0];
endmodule


module out_reg_ram_fc #(
    parameter ACC_WIDTH = 32
)(
    input  wire                         clk,
    input  wire                         wr_en,
    input  wire signed [ACC_WIDTH-1:0]  wr_data,
    output reg  signed [ACC_WIDTH-1:0]  rd_data
);
    always @(posedge clk) begin
        if (wr_en) rd_data <= wr_data;
    end
endmodule
