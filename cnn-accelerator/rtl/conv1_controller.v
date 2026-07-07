// conv1_controller.v
// Computes Conv1 of the CNN accelerator: 1x8x8 input -> 4 output channels,
// 3x3 kernel, valid convolution -> 4x6x6 pre-activation accumulators.
//
// Architecture: 4 parallel mac_unit PEs (one per output channel, weight-
// stationary within a tap). For each of the 36 output pixels, the controller
// streams 9 (ky,kx) taps; each PE multiplies the shared activation sample by
// its own per-channel weight and accumulates. After 9 taps, the controller
// adds the per-channel bias (as a separate adder stage, matching the golden
// Python model's acc[co] = sum(w*x) + b order exactly) and latches the
// result into out_acc_mem.
//
// This is intentionally scoped to Conv1 for Week-1 delivery. Conv2 and the
// FC layer reuse this exact same control pattern (only loop bounds change:
// Cin=4 instead of 1, Cout=8 instead of 4, output 4x4 instead of 6x6) --
// documented as the immediate next extension in docs/EXTENDING.md.

module conv1_controller #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter NUM_CH     = 4,   // output channels
    parameter IN_SIZE    = 8,
    parameter K          = 3,
    parameter OUT_SIZE   = IN_SIZE - K + 1  // 6
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // Input activation memory (8x8 int8), read one pixel per cycle
    output reg  [6:0]                    img_addr,   // 0..63
    input  wire signed [DATA_WIDTH-1:0]  img_data,

    // Weight memory (4 channels x 9 taps int8), read one per cycle per channel
    output reg  [5:0]                    wgt_addr,   // 0..35 (co*9 + tap)
    input  wire signed [NUM_CH*DATA_WIDTH-1:0] wgt_data, // all 4 channels' weight for current tap, packed

    // Bias memory (4 x int32)
    input  wire signed [NUM_CH*ACC_WIDTH-1:0] bias_data, // all 4 biases, packed

    // Output accumulator writeback (4 channels x 36 pixels x int32)
    output reg                            out_wr_en,
    output reg  [7:0]                     out_addr,   // co*36 + pixel_idx
    output wire signed [ACC_WIDTH-1:0]    out_data_ch0,
    output wire signed [ACC_WIDTH-1:0]    out_data_ch1,
    output wire signed [ACC_WIDTH-1:0]    out_data_ch2,
    output wire signed [ACC_WIDTH-1:0]    out_data_ch3
);

    localparam S_IDLE     = 0,
               S_TAP      = 1,
               S_DRAIN    = 5,
               S_DRAIN2   = 6,  // extra cycle: lets the LAST tap's ROM read (1-cycle latency) land in the MAC before S_BIAS reads mac_acc_flat
               S_BIAS     = 2,
               S_WRITE    = 3,
               S_DONE     = 4;

    reg [2:0]  state;
    reg [3:0]  oy, ox;        // output pixel coords, 0..5
    reg [3:0]  ky, kx;        // tap coords, 0..2
    reg [5:0]  tap_idx;       // 0..8, linear tap counter for wgt_addr base

    // TWO cycles of delay between "tap_idx says we are addressing tap T" and
    // "img_data/wgt_data actually show tap T"'s value:
    //   1) img_addr/wgt_addr are themselves registered from (ky,kx)/tap_idx,
    //      so they reflect the PREVIOUS cycle's tap, not the current one.
    //   2) img_rom/wgt_rom_tapmajor are registered reads, adding one more
    //      cycle on top of that.
    // A single delay stage was NOT enough (verified against golden vectors,
    // 124/144 mismatched) -- mac_en/mac_acc_clear must lag tap_idx by TWO
    // cycles, not one.
    wire addr_phase = (state == S_TAP);
    reg  addr_phase_d, addr_phase_d2;
    reg  [5:0] tap_idx_d, tap_idx_d2;

    wire mac_en        = addr_phase_d2;
    wire mac_acc_clear  = addr_phase_d2 && (tap_idx_d2 == 0);

    // Shared activation value for current tap, broadcast to all 4 PEs
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

    // Post-accumulation bias adder (one per channel), latched result registers
    reg signed [ACC_WIDTH-1:0] result [0:NUM_CH-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; oy <= 0; ox <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
            done <= 0; out_wr_en <= 0; out_addr <= 0; img_addr <= 0; wgt_addr <= 0;
            for (i = 0; i < NUM_CH; i = i + 1) result[i] <= 0;
            addr_phase_d <= 0; tap_idx_d <= 0; addr_phase_d2 <= 0; tap_idx_d2 <= 0;
        end else begin
            out_wr_en <= 0;
            addr_phase_d <= addr_phase;   addr_phase_d2 <= addr_phase_d;
            tap_idx_d    <= tap_idx;      tap_idx_d2    <= tap_idx_d;
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        oy <= 0; ox <= 0; ky <= 0; kx <= 0; tap_idx <= 0;
                        img_addr <= 0; wgt_addr <= 0;
                        state <= S_TAP;
                    end
                end

                S_TAP: begin
                    // present address for THIS cycle's tap; data arrives combinationally
                    img_addr <= (oy + ky) * IN_SIZE + (ox + kx);
                    wgt_addr <= tap_idx; // controller/wgt_mem packs all 4 channels per tap_idx row

                    if (tap_idx == K*K - 1) begin
                        state <= S_DRAIN;
                    end else begin
                        tap_idx <= tap_idx + 1;
                        if (kx == K-1) begin kx <= 0; ky <= ky + 1; end
                        else kx <= kx + 1;
                    end
                end

                S_DRAIN: begin
                    state <= S_DRAIN2; // need a 2nd drain cycle: two total pipeline stages to clear
                end

                S_DRAIN2: begin
                    state <= S_BIAS;
                end

                S_BIAS: begin
                    for (i = 0; i < NUM_CH; i = i + 1)
                        result[i] <= mac_acc_flat[i*ACC_WIDTH +: ACC_WIDTH] + bias_data[i*ACC_WIDTH +: ACC_WIDTH];
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    out_addr  <= oy * OUT_SIZE + ox; // per-channel base; top wires 4 parallel writes
                    out_wr_en <= 1;
                    // advance to next output pixel
                    if (ox == OUT_SIZE-1) begin
                        ox <= 0;
                        if (oy == OUT_SIZE-1) begin
                            state <= S_DONE;
                        end else begin
                            oy <= oy + 1;
                            ky <= 0; kx <= 0; tap_idx <= 0;
                            state <= S_TAP;
                        end
                    end else begin
                        ox <= ox + 1;
                        ky <= 0; kx <= 0; tap_idx <= 0;
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

    assign out_data_ch0 = result[0];
    assign out_data_ch1 = result[1];
    assign out_data_ch2 = result[2];
    assign out_data_ch3 = result[3];

endmodule
