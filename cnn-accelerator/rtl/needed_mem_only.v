// needed_mem_only.v -- subset of conv1_mem.v / conv2_mem.v / fc_mem.v that
// top_accelerator.v actually instantiates (excludes out_acc_ram, img_rom_c2,
// img_rom_fc, out_acc_ram_c2 which are superseded by chain_mem.v''s
// acc_ram_dp/img_ram_w). Used only for isolating a build issue -- temporary.

module wgt_rom_tapmajor #(parameter NUM_CH=4, DATA_WIDTH=8, TAPS=9)(
    input wire clk, input wire [5:0] addr, output reg signed [NUM_CH*DATA_WIDTH-1:0] data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex/w1_tapmajor.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule

module bias_rom #(parameter NUM_CH=4, ACC_WIDTH=32)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex/b1.hex", mem);
    assign data = mem[0];
endmodule

module img_rom #(parameter DATA_WIDTH=8, DEPTH=64)(
    input wire clk, input wire [6:0] addr, output reg signed [DATA_WIDTH-1:0] data
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh("golden_hex/input_0.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule

module wgt_rom_c2 #(parameter NUM_CH=8, DATA_WIDTH=8, TAPS=36)(
    input wire clk, input wire [5:0] addr, output reg signed [NUM_CH*DATA_WIDTH-1:0] data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex_v2/w2_tapmajor.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule

module bias_rom_c2 #(parameter NUM_CH=8, ACC_WIDTH=32)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex_v2/b2.hex", mem);
    assign data = mem[0];
endmodule

module wgt_rom_fc #(parameter NUM_CH=10, DATA_WIDTH=8, TAPS=128)(
    input wire clk, input wire [6:0] addr, output reg signed [NUM_CH*DATA_WIDTH-1:0] data
);
    reg [NUM_CH*DATA_WIDTH-1:0] mem [0:TAPS-1];
    initial $readmemh("weights_hex_fc/w3_tapmajor.hex", mem);
    always @(posedge clk) data <= mem[addr];
endmodule

module bias_rom_fc #(parameter NUM_CH=10, ACC_WIDTH=32)(
    output wire signed [NUM_CH*ACC_WIDTH-1:0] data
);
    reg [NUM_CH*ACC_WIDTH-1:0] mem [0:0];
    initial $readmemh("weights_hex_fc/b3.hex", mem);
    assign data = mem[0];
endmodule

module out_reg_ram_fc #(parameter ACC_WIDTH=32)(
    input wire clk, input wire wr_en, input wire signed [ACC_WIDTH-1:0] wr_data,
    output reg signed [ACC_WIDTH-1:0] rd_data
);
    always @(posedge clk) if (wr_en) rd_data <= wr_data;
endmodule
