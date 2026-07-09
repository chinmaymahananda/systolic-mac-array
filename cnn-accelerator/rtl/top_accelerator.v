// top_accelerator.v
// Chains Conv1 -> requant_bridge -> Conv2 -> requant_bridge -> FC into one
// pipeline with a single start/done handshake, per docs/EXTENDING.md Day 3.
// Reuses the already-verified conv1_controller/conv2_controller/fc_controller
// and their weight/bias ROMs unchanged. Only the per-layer output/input
// buffers are swapped for dual-port/writable variants (chain_mem.v) so the
// requant_bridge can move data between stages without touching the
// controllers themselves.

module top_accelerator #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output wire signed [ACC_WIDTH-1:0] logit0, logit1, logit2, logit3, logit4,
    output wire signed [ACC_WIDTH-1:0] logit5, logit6, logit7, logit8, logit9
);

    // ---------------- Conv1 ----------------
    wire [6:0] c1_img_addr; wire signed [DATA_WIDTH-1:0] c1_img_data;
    wire [5:0] c1_wgt_addr; wire signed [4*DATA_WIDTH-1:0] c1_wgt_data;
    wire signed [4*ACC_WIDTH-1:0] c1_bias_data;
    wire c1_out_wr_en; wire [7:0] c1_out_addr;
    wire signed [ACC_WIDTH-1:0] c1_o0, c1_o1, c1_o2, c1_o3;
    wire c1_start, c1_done;

    img_rom u_img_rom_c1 (.clk(clk), .addr(c1_img_addr), .data(c1_img_data));
    wgt_rom_tapmajor #(.NUM_CH(4)) u_wgt_rom_c1 (.clk(clk), .addr(c1_wgt_addr), .data(c1_wgt_data));
    bias_rom #(.NUM_CH(4)) u_bias_rom_c1 (.data(c1_bias_data));

    conv1_controller #(.NUM_CH(4)) u_c1 (
        .clk(clk), .rst_n(rst_n), .start(c1_start), .done(c1_done),
        .img_addr(c1_img_addr), .img_data(c1_img_data),
        .wgt_addr(c1_wgt_addr), .wgt_data(c1_wgt_data),
        .bias_data(c1_bias_data),
        .out_wr_en(c1_out_wr_en), .out_addr(c1_out_addr),
        .out_data_ch0(c1_o0), .out_data_ch1(c1_o1), .out_data_ch2(c1_o2), .out_data_ch3(c1_o3)
    );

    wire [5:0] br1_src_addr;
    wire signed [ACC_WIDTH-1:0] c1_rd0, c1_rd1, c1_rd2, c1_rd3;
    acc_ram_dp #(.DEPTH(36), .AW(6)) u_c1ram0 (.clk(clk), .wr_en(c1_out_wr_en), .wr_addr(c1_out_addr[5:0]), .wr_data(c1_o0), .rd_addr(br1_src_addr), .rd_data(c1_rd0));
    acc_ram_dp #(.DEPTH(36), .AW(6)) u_c1ram1 (.clk(clk), .wr_en(c1_out_wr_en), .wr_addr(c1_out_addr[5:0]), .wr_data(c1_o1), .rd_addr(br1_src_addr), .rd_data(c1_rd1));
    acc_ram_dp #(.DEPTH(36), .AW(6)) u_c1ram2 (.clk(clk), .wr_en(c1_out_wr_en), .wr_addr(c1_out_addr[5:0]), .wr_data(c1_o2), .rd_addr(br1_src_addr), .rd_data(c1_rd2));
    acc_ram_dp #(.DEPTH(36), .AW(6)) u_c1ram3 (.clk(clk), .wr_en(c1_out_wr_en), .wr_addr(c1_out_addr[5:0]), .wr_data(c1_o3), .rd_addr(br1_src_addr), .rd_data(c1_rd3));

    wire br1_start, br1_done;
    wire br1_dst_wr_en; wire [7:0] br1_dst_addr; wire signed [7:0] br1_dst_data;

    requant_bridge #(.NUM_CH(4), .PIXELS(36), .SHIFT(7), .SRC_AW(6), .DST_AW(8)) u_bridge1 (
        .clk(clk), .rst_n(rst_n), .start(br1_start), .done(br1_done),
        .src_addr(br1_src_addr),
        .src_rd_data_flat({c1_rd3, c1_rd2, c1_rd1, c1_rd0}),
        .dst_wr_en(br1_dst_wr_en), .dst_wr_addr(br1_dst_addr), .dst_wr_data(br1_dst_data)
    );

    // ---------------- Conv2 ----------------
    wire [7:0] c2_img_addr; wire signed [DATA_WIDTH-1:0] c2_img_data;
    wire [5:0] c2_wgt_addr; wire signed [8*DATA_WIDTH-1:0] c2_wgt_data;
    wire signed [8*ACC_WIDTH-1:0] c2_bias_data;
    wire c2_out_wr_en; wire [7:0] c2_out_addr;
    wire signed [ACC_WIDTH-1:0] c2_o0,c2_o1,c2_o2,c2_o3,c2_o4,c2_o5,c2_o6,c2_o7;
    wire c2_start, c2_done;

    img_ram_w #(.DEPTH(144), .AW(8)) u_img_ram_c2 (
        .clk(clk), .wr_en(br1_dst_wr_en), .wr_addr(br1_dst_addr), .wr_data(br1_dst_data),
        .rd_addr(c2_img_addr), .rd_data(c2_img_data)
    );
    wgt_rom_c2 #(.NUM_CH(8)) u_wgt_rom_c2 (.clk(clk), .addr(c2_wgt_addr), .data(c2_wgt_data));
    bias_rom_c2 #(.NUM_CH(8)) u_bias_rom_c2 (.data(c2_bias_data));

    conv2_controller #(.NUM_CH(8)) u_c2 (
        .clk(clk), .rst_n(rst_n), .start(c2_start), .done(c2_done),
        .img_addr(c2_img_addr), .img_data(c2_img_data),
        .wgt_addr(c2_wgt_addr), .wgt_data(c2_wgt_data),
        .bias_data(c2_bias_data),
        .out_wr_en(c2_out_wr_en), .out_addr(c2_out_addr),
        .out_data_ch0(c2_o0), .out_data_ch1(c2_o1), .out_data_ch2(c2_o2), .out_data_ch3(c2_o3),
        .out_data_ch4(c2_o4), .out_data_ch5(c2_o5), .out_data_ch6(c2_o6), .out_data_ch7(c2_o7)
    );

    wire [3:0] br2_src_addr;
    wire signed [ACC_WIDTH-1:0] c2_rd0,c2_rd1,c2_rd2,c2_rd3,c2_rd4,c2_rd5,c2_rd6,c2_rd7;
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram0 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o0), .rd_addr(br2_src_addr), .rd_data(c2_rd0));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram1 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o1), .rd_addr(br2_src_addr), .rd_data(c2_rd1));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram2 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o2), .rd_addr(br2_src_addr), .rd_data(c2_rd2));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram3 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o3), .rd_addr(br2_src_addr), .rd_data(c2_rd3));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram4 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o4), .rd_addr(br2_src_addr), .rd_data(c2_rd4));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram5 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o5), .rd_addr(br2_src_addr), .rd_data(c2_rd5));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram6 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o6), .rd_addr(br2_src_addr), .rd_data(c2_rd6));
    acc_ram_dp #(.DEPTH(16), .AW(4)) u_c2ram7 (.clk(clk), .wr_en(c2_out_wr_en), .wr_addr(c2_out_addr[3:0]), .wr_data(c2_o7), .rd_addr(br2_src_addr), .rd_data(c2_rd7));

    wire br2_start, br2_done;
    wire br2_dst_wr_en; wire [6:0] br2_dst_addr; wire signed [7:0] br2_dst_data;

    requant_bridge #(.NUM_CH(8), .PIXELS(16), .SHIFT(7), .SRC_AW(4), .DST_AW(7)) u_bridge2 (
        .clk(clk), .rst_n(rst_n), .start(br2_start), .done(br2_done),
        .src_addr(br2_src_addr),
        .src_rd_data_flat({c2_rd7,c2_rd6,c2_rd5,c2_rd4,c2_rd3,c2_rd2,c2_rd1,c2_rd0}),
        .dst_wr_en(br2_dst_wr_en), .dst_wr_addr(br2_dst_addr), .dst_wr_data(br2_dst_data)
    );

    // ---------------- FC ----------------
    wire [6:0] fc_img_addr; wire signed [DATA_WIDTH-1:0] fc_img_data;
    wire [6:0] fc_wgt_addr; wire signed [10*DATA_WIDTH-1:0] fc_wgt_data;
    wire signed [10*ACC_WIDTH-1:0] fc_bias_data;
    wire fc_out_wr_en;
    wire signed [ACC_WIDTH-1:0] fc_o0,fc_o1,fc_o2,fc_o3,fc_o4,fc_o5,fc_o6,fc_o7,fc_o8,fc_o9;
    wire fc_start, fc_done;

    img_ram_w #(.DEPTH(128), .AW(7)) u_img_ram_fc (
        .clk(clk), .wr_en(br2_dst_wr_en), .wr_addr(br2_dst_addr), .wr_data(br2_dst_data),
        .rd_addr(fc_img_addr), .rd_data(fc_img_data)
    );
    wgt_rom_fc #(.NUM_CH(10)) u_wgt_rom_fc (.clk(clk), .addr(fc_wgt_addr), .data(fc_wgt_data));
    bias_rom_fc #(.NUM_CH(10)) u_bias_rom_fc (.data(fc_bias_data));

    fc_controller #(.NUM_CH(10)) u_fc (
        .clk(clk), .rst_n(rst_n), .start(fc_start), .done(fc_done),
        .img_addr(fc_img_addr), .img_data(fc_img_data),
        .wgt_addr(fc_wgt_addr), .wgt_data(fc_wgt_data),
        .bias_data(fc_bias_data),
        .out_wr_en(fc_out_wr_en),
        .out_data_ch0(fc_o0), .out_data_ch1(fc_o1), .out_data_ch2(fc_o2), .out_data_ch3(fc_o3), .out_data_ch4(fc_o4),
        .out_data_ch5(fc_o5), .out_data_ch6(fc_o6), .out_data_ch7(fc_o7), .out_data_ch8(fc_o8), .out_data_ch9(fc_o9)
    );

    out_reg_ram_fc u_fcreg0 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o0), .rd_data(logit0));
    out_reg_ram_fc u_fcreg1 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o1), .rd_data(logit1));
    out_reg_ram_fc u_fcreg2 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o2), .rd_data(logit2));
    out_reg_ram_fc u_fcreg3 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o3), .rd_data(logit3));
    out_reg_ram_fc u_fcreg4 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o4), .rd_data(logit4));
    out_reg_ram_fc u_fcreg5 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o5), .rd_data(logit5));
    out_reg_ram_fc u_fcreg6 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o6), .rd_data(logit6));
    out_reg_ram_fc u_fcreg7 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o7), .rd_data(logit7));
    out_reg_ram_fc u_fcreg8 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o8), .rd_data(logit8));
    out_reg_ram_fc u_fcreg9 (.clk(clk), .wr_en(fc_out_wr_en), .wr_data(fc_o9), .rd_data(logit9));

    // ---------------- Top-level sequencer ----------------
    // Each sub-block''s "start" must be a clean pulse that is LOW again by
    // the time it reaches its own done state (matching conv1_controller''s
    // own S_DONE -> S_IDLE transition rule: "if (!start) state <= S_IDLE").
    // Hence the separate *_KICK states: assert start for exactly one cycle,
    // then drop it, then wait for done.
    localparam T_IDLE=0, T_C1_KICK=1, T_C1_WAIT=2, T_BR1_KICK=3, T_BR1_WAIT=4,
               T_C2_KICK=5, T_C2_WAIT=6, T_BR2_KICK=7, T_BR2_WAIT=8,
               T_FC_KICK=9, T_FC_WAIT=10, T_DONE=11;
    reg [3:0] tstate;

    reg c1_start_r, br1_start_r, c2_start_r, br2_start_r, fc_start_r;
    assign c1_start = c1_start_r;
    assign br1_start = br1_start_r;
    assign c2_start = c2_start_r;
    assign br2_start = br2_start_r;
    assign fc_start = fc_start_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tstate <= T_IDLE; done <= 0;
            c1_start_r<=0; br1_start_r<=0; c2_start_r<=0; br2_start_r<=0; fc_start_r<=0;
        end else begin
            done <= 0;
            c1_start_r<=0; br1_start_r<=0; c2_start_r<=0; br2_start_r<=0; fc_start_r<=0;
            case (tstate)
                T_IDLE:     if (start) begin c1_start_r<=1; tstate<=T_C1_KICK; end
                T_C1_KICK:  tstate <= T_C1_WAIT;
                T_C1_WAIT:  if (c1_done) begin br1_start_r<=1; tstate<=T_BR1_KICK; end
                T_BR1_KICK: tstate <= T_BR1_WAIT;
                T_BR1_WAIT: if (br1_done) begin c2_start_r<=1; tstate<=T_C2_KICK; end
                T_C2_KICK:  tstate <= T_C2_WAIT;
                T_C2_WAIT:  if (c2_done) begin br2_start_r<=1; tstate<=T_BR2_KICK; end
                T_BR2_KICK: tstate <= T_BR2_WAIT;
                T_BR2_WAIT: if (br2_done) begin fc_start_r<=1; tstate<=T_FC_KICK; end
                T_FC_KICK:  tstate <= T_FC_WAIT;
                T_FC_WAIT:  if (fc_done) tstate <= T_DONE;
                T_DONE:     begin done <= 1; tstate <= T_IDLE; end
                default:    tstate <= T_IDLE;
            endcase
        end
    end

endmodule
