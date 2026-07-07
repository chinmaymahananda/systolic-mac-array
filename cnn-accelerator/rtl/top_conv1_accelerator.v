// top_conv1_accelerator.v
// Top-level: wires conv1_controller to image ROM, weight ROM, bias ROM,
// and 4 output accumulator RAM banks (one per output channel).

module top_conv1_accelerator #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter NUM_CH     = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire done
);

    wire [6:0] img_addr;
    wire signed [DATA_WIDTH-1:0] img_data;

    wire [5:0] wgt_addr;
    wire signed [NUM_CH*DATA_WIDTH-1:0] wgt_data;

    wire signed [NUM_CH*ACC_WIDTH-1:0] bias_data;

    wire out_wr_en;
    wire [7:0] out_addr;
    wire signed [ACC_WIDTH-1:0] out_ch0, out_ch1, out_ch2, out_ch3;

    img_rom #(.DATA_WIDTH(DATA_WIDTH)) u_img_rom (
        .clk(clk), .addr(img_addr), .data(img_data)
    );

    wgt_rom_tapmajor #(.NUM_CH(NUM_CH), .DATA_WIDTH(DATA_WIDTH)) u_wgt_rom (
        .clk(clk), .addr(wgt_addr), .data(wgt_data)
    );

    bias_rom #(.NUM_CH(NUM_CH), .ACC_WIDTH(ACC_WIDTH)) u_bias_rom (
        .data(bias_data)
    );

    conv1_controller #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .NUM_CH(NUM_CH)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .img_addr(img_addr), .img_data(img_data),
        .wgt_addr(wgt_addr), .wgt_data(wgt_data),
        .bias_data(bias_data),
        .out_wr_en(out_wr_en), .out_addr(out_addr),
        .out_data_ch0(out_ch0), .out_data_ch1(out_ch1),
        .out_data_ch2(out_ch2), .out_data_ch3(out_ch3)
    );

    out_acc_ram #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch0 (
        .clk(clk), .wr_en(out_wr_en), .addr(out_addr), .wr_data(out_ch0), .rd_data()
    );
    out_acc_ram #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch1 (
        .clk(clk), .wr_en(out_wr_en), .addr(out_addr), .wr_data(out_ch1), .rd_data()
    );
    out_acc_ram #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch2 (
        .clk(clk), .wr_en(out_wr_en), .addr(out_addr), .wr_data(out_ch2), .rd_data()
    );
    out_acc_ram #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch3 (
        .clk(clk), .wr_en(out_wr_en), .addr(out_addr), .wr_data(out_ch3), .rd_data()
    );

endmodule
