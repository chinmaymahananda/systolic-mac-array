// conv2_controller.v
// Computes Conv2: 4x6x6 input (Conv1's requantized output) -> 8 output
// channels, 3x3 kernel, valid convolution -> 8x4x4 pre-activation accumulators.
//
// Same verified architecture/timing as conv1_controller.v (8 parallel MAC
// PEs instead of 4; taps now loop over ci*ky*kx = 4*3*3 = 36 instead of 9).
// The 2-cycle mac_en/mac_acc_clear delay (addr_phase_d2/tap_idx_d2) was
// verified bit-exact for Conv1 against 20 golden samples -- reused as-is.

module conv2_controller #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter NUM_CH     = 8,   // output channels
    parameter IN_CH      = 4,   // input channels
    parameter IN_SIZE    = 6,
    parameter K          = 3,
    parameter OUT_SIZE   = IN_SIZE - K + 1,  // 4
    parameter TAPS       = IN_CH * K * K     // 36
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    output reg  [7:0]                    img_addr,   // 0..143 (ci*36 + row*6 + col)
    input  wire signed [DATA_WIDTH-1:0]  img_data,

    output reg  [5:0]                    wgt_addr,   // 0..35 tap index
    input  wire signed [NUM_CH*DATA_WIDTH-1:0] wgt_data,

    input  wire signed [NUM_CH*ACC_WIDTH-1:0] bias_data,

    output reg                            out_wr_en,
    output reg  [7:0]                     out_addr,   // pixel_idx, 0..15 (per-channel base)
    output wire signed [ACC_WIDTH-1:0]    out_data_ch0, out_data_ch1, out_data_ch2, out_data_ch3,
    output wire signed [ACC_WIDTH-1:0]    out_data_ch4, out_data_ch5, out_data_ch6, out_data_ch7
);

    localparam S_IDLE   = 0,
               S_TAP    = 1,
               S_DRAIN  = 5,
               S_DRAIN2 = 6,
               S_BIAS   = 2,
               S_WRITE  = 3,
               S_DONE   = 4;

    reg [3:0] state;
    reg [3:0] oy, ox;         // output pixel coords, 0..3
    reg [2:0] ci;             // input channel, 0..3
    reg [2:0] ky, kx;         // tap coords, 0..2
    reg [6:0] tap_idx;        // 0..35

    wire addr_phase = (state == S_TAP);
    reg  addr_phase_d, addr_phase_d2;
    reg  [6:0] tap_idx_d, tap_idx_d2;

    wire mac_en       = addr_phase_d2;
    wire mac_acc_clear = addr_phase_d2 && (tap_idx_d2 == 0);

    wire signed [DATA_WIDTH-1:0] a_shared = img_data;

    wire signed [NUM_CH*ACC_WIDTH-1:0] mac_acc_flat;

    genvar g;
    generate
        for (g = 0; g < NUM_CH; g = g + 1) begin : PE
            mac_unit #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_mac (
                .clk(clk), .rst_n(rst_n),
                .en(mac_en), .acc_clear(mac_acc_clear),
                .a_in(a_shared),
                .w_in(wgt_data[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH]),
                .acc_out(mac_acc_flat[(g+1)*ACC_WIDTH-1 : g*ACC_WIDTH])
            );
        end
    endgenerate

    reg signed [ACC_WIDTH-1:0] result [0:NUM_CH-1];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; oy <= 0; ox <= 0; ci <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
            done <= 0; out_wr_en <= 0; out_addr <= 0; img_addr <= 0; wgt_addr <= 0;
            addr_phase_d <= 0; tap_idx_d <= 0; addr_phase_d2 <= 0; tap_idx_d2 <= 0;
            for (i = 0; i < NUM_CH; i = i + 1) result[i] <= 0;
        end else begin
            out_wr_en <= 0;
            addr_phase_d <= addr_phase;   addr_phase_d2 <= addr_phase_d;
            tap_idx_d    <= tap_idx;      tap_idx_d2    <= tap_idx_d;
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        oy <= 0; ox <= 0; ci <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
                        img_addr <= 0; wgt_addr <= 0;
                        state <= S_TAP;
                    end
                end

                S_TAP: begin
                    img_addr <= ci * (IN_SIZE*IN_SIZE) + (oy + ky) * IN_SIZE + (ox + kx);
                    wgt_addr <= tap_idx;

                    if (tap_idx == TAPS - 1) begin
                        state <= S_DRAIN;
                    end else begin
                        tap_idx <= tap_idx + 1;
                        if (kx == K-1) begin
                            kx <= 0;
                            if (ky == K-1) begin ky <= 0; ci <= ci + 1; end
                            else ky <= ky + 1;
                        end else kx <= kx + 1;
                    end
                end

                S_DRAIN:  state <= S_DRAIN2;
                S_DRAIN2: state <= S_BIAS;

                S_BIAS: begin
                    for (i = 0; i < NUM_CH; i = i + 1)
                        result[i] <= mac_acc_flat[i*ACC_WIDTH +: ACC_WIDTH] + bias_data[i*ACC_WIDTH +: ACC_WIDTH];
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    out_addr  <= oy * OUT_SIZE + ox;
                    out_wr_en <= 1;
                    if (ox == OUT_SIZE-1) begin
                        ox <= 0;
                        if (oy == OUT_SIZE-1) begin
                            state <= S_DONE;
                        end else begin
                            oy <= oy + 1;
                            ci <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
                            state <= S_TAP;
                        end
                    end else begin
                        ox <= ox + 1;
                        ci <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
                        state <= S_TAP;
                    end
                end

                S_DONE: begin
                    done <= 1;
                    if (!start) state <= S_IDLE;
                end
            endcase
        end
    end

    assign out_data_ch0 = result[0]; assign out_data_ch1 = result[1];
    assign out_data_ch2 = result[2]; assign out_data_ch3 = result[3];
    assign out_data_ch4 = result[4]; assign out_data_ch5 = result[5];
    assign out_data_ch6 = result[6]; assign out_data_ch7 = result[7];

endmodule
