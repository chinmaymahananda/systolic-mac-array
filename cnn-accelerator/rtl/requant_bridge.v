// requant_bridge.v
// Moves NUM_CH channel banks of raw INT32 accumulator values (from one
// layer''s acc_ram_dp instances) through ReLU -> round -> right-shift ->
// clamp-to-int8, into a single flattened img_ram_w for the next layer, in
// channel-major order (ch*PIXELS + pixel) -- matching the exact flatten
// order used by quantize_and_golden_v2.py (c_q.flatten()) and by
// conv2_controller/fc_controller''s own img_addr indexing (ci*PIXELS + pixel).
//
// Conservative on timing by design: holds each source address for an extra
// settle cycle before trusting rd_data, mirroring the registered-ROM +
// registered-address latency lesson already documented in this repo''s
// Conv1 bring-up (README.md "Day 1 Update"). Correctness first.
module requant_bridge #(
    parameter NUM_CH  = 4,
    parameter PIXELS  = 36,
    parameter SHIFT   = 7,
    parameter SRC_AW  = 6,
    parameter DST_AW  = 8
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    output reg  [SRC_AW-1:0]      src_addr,
    input  wire [NUM_CH*32-1:0]   src_rd_data_flat,

    output reg                    dst_wr_en,
    output reg  [DST_AW-1:0]      dst_wr_addr,
    output reg  signed [7:0]      dst_wr_data
);

    localparam S_IDLE  = 2'd0;
    localparam S_ADDR  = 2'd1;
    localparam S_WAIT  = 2'd2;
    localparam S_WRITE = 2'd3;

    reg [1:0] state;
    reg [$clog2(PIXELS+1)-1:0] pix;
    reg [$clog2(NUM_CH+1)-1:0] ch;

    function signed [7:0] relu_requant(input signed [31:0] acc);
        reg signed [31:0] relu, rounded;
        begin
            relu = (acc[31]) ? 32'sd0 : acc;
            if (SHIFT > 0)
                rounded = (relu + (1 <<< (SHIFT-1))) >>> SHIFT;
            else
                rounded = relu;
            relu_requant = (rounded > 32'sd127) ? 8'sd127 : rounded[7:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; pix <= 0; ch <= 0; done <= 0;
            dst_wr_en <= 0; src_addr <= 0;
        end else begin
            done <= 0;
            dst_wr_en <= 0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        pix <= 0; ch <= 0; src_addr <= 0;
                        state <= S_ADDR;
                    end
                end
                S_ADDR: begin
                    src_addr <= pix;
                    state <= S_WAIT;
                end
                S_WAIT: begin
                    state <= S_WRITE;
                end
                S_WRITE: begin
                    dst_wr_en   <= 1;
                    dst_wr_addr <= ch * PIXELS + pix;
                    dst_wr_data <= relu_requant($signed(src_rd_data_flat[ch*32 +: 32]));
                    if (ch == NUM_CH-1) begin
                        ch <= 0;
                        if (pix == PIXELS-1) begin
                            done  <= 1;
                            state <= S_IDLE;
                        end else begin
                            pix <= pix + 1;
                            state <= S_ADDR;
                        end
                    end else begin
                        ch <= ch + 1;
                    end
                end
            endcase
        end
    end
endmodule
