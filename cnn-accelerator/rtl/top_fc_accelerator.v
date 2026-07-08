// top_fc_accelerator.v -- wires fc_controller to its memories

module top_fc_accelerator #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter NUM_CH     = 10
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire done
);

    wire [6:0] img_addr;
    wire signed [DATA_WIDTH-1:0] img_data;
    wire [6:0] wgt_addr;
    wire signed [NUM_CH*DATA_WIDTH-1:0] wgt_data;
    wire signed [NUM_CH*ACC_WIDTH-1:0] bias_data;
    wire out_wr_en;
    wire signed [ACC_WIDTH-1:0] out_ch0, out_ch1, out_ch2, out_ch3, out_ch4, out_ch5, out_ch6, out_ch7, out_ch8, out_ch9;

    img_rom_fc #(.DATA_WIDTH(DATA_WIDTH)) u_img_rom (.clk(clk), .addr(img_addr), .data(img_data));
    wgt_rom_fc #(.NUM_CH(NUM_CH), .DATA_WIDTH(DATA_WIDTH)) u_wgt_rom (.clk(clk), .addr(wgt_addr), .data(wgt_data));
    bias_rom_fc #(.NUM_CH(NUM_CH), .ACC_WIDTH(ACC_WIDTH)) u_bias_rom (.data(bias_data));

    fc_controller #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .NUM_CH(NUM_CH)) u_ctrl (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .img_addr(img_addr), .img_data(img_data),
        .wgt_addr(wgt_addr), .wgt_data(wgt_data),
        .bias_data(bias_data),
        .out_wr_en(out_wr_en),
        .out_data_ch0(out_ch0), .out_data_ch1(out_ch1), .out_data_ch2(out_ch2), .out_data_ch3(out_ch3), .out_data_ch4(out_ch4),
        .out_data_ch5(out_ch5), .out_data_ch6(out_ch6), .out_data_ch7(out_ch7), .out_data_ch8(out_ch8), .out_data_ch9(out_ch9)
    );

    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch0 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch0), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch1 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch1), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch2 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch2), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch3 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch3), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch4 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch4), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch5 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch5), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch6 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch6), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch7 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch7), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch8 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch8), .rd_data());
    out_reg_ram_fc #(.ACC_WIDTH(ACC_WIDTH)) u_ram_ch9 (.clk(clk), .wr_en(out_wr_en), .wr_data(out_ch9), .rd_data());

endmodule
