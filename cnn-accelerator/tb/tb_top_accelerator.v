// tb_top_accelerator.v -- end-to-end check: real image -> Conv1 -> Conv2 ->
// FC (all chained through top_accelerator.v), compared against the true
// end-to-end golden logits for all 20 calibration samples in one run.
// This is Day 4 from docs/EXTENDING.md.
`timescale 1ns/1ps

module tb_top_accelerator;
    reg clk = 0, rst_n = 0, start = 0;
    wire done;
    wire signed [31:0] l0,l1,l2,l3,l4,l5,l6,l7,l8,l9;
    integer sample, errors, total_errors, total_samples, k;
    reg [8*64:0] fname;
    reg signed [31:0] golden_logits [0:9];

    top_accelerator dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .logit0(l0), .logit1(l1), .logit2(l2), .logit3(l3), .logit4(l4),
        .logit5(l5), .logit6(l6), .logit7(l7), .logit8(l8), .logit9(l9)
    );
    always #5 clk = ~clk;

    task load_golden(input integer s);
        reg [31:0] flat_mem [0:9];
        integer k;
        begin
            $sformat(fname, "golden_hex_fc/fc_logits_%0d.hex", s);
            $readmemh(fname, flat_mem);
            for (k = 0; k < 10; k = k + 1) golden_logits[k] = flat_mem[k];
        end
    endtask

    integer got [0:9];

    initial begin
        $display("SIM STARTED at time %0t", $time);
        total_errors = 0; total_samples = 0;
        for (sample = 0; sample < 20; sample = sample + 1) begin
            $sformat(fname, "golden_hex/input_%0d.hex", sample);
            $readmemh(fname, dut.u_img_rom_c1.mem);
            load_golden(sample);

            errors = 0;
            rst_n = 0; start = 0;
            #20 rst_n = 1;
            #10 start = 1;
            wait (done == 1);
            #1;
            got[0]=l0; got[1]=l1; got[2]=l2; got[3]=l3; got[4]=l4;
            got[5]=l5; got[6]=l6; got[7]=l7; got[8]=l8; got[9]=l9;
            #9;
            start = 0;

            for (k = 0; k < 10; k = k + 1) begin
                if (got[k] !== golden_logits[k]) begin
                    errors = errors + 1;
                    $display("  SAMPLE %0d ch%0d MISMATCH: rtl=%0d golden=%0d", sample, k, got[k], golden_logits[k]);
                end
            end

            if (errors == 0)
                $display("SAMPLE %0d: PASS -- all 10 end-to-end logits bit-exact", sample);
            else begin
                $display("SAMPLE %0d: FAIL -- %0d/10 mismatches", sample, errors);
                total_errors = total_errors + 1;
            end
            total_samples = total_samples + 1;
        end

        $display("");
        if (total_errors == 0)
            $display("*** PASS: all %0d/%0d samples bit-exact end-to-end (Conv1->Conv2->FC chained) ***", total_samples, total_samples);
        else
            $display("*** FAIL: %0d/%0d samples had mismatches ***", total_errors, total_samples);

        $finish;
    end
    initial begin
        #1000000;
        $display("WATCHDOG TIMEOUT at time %0t", $time);
        $display("  tstate=%0d c1_done=%0d br1_done=%0d c2_done=%0d br2_done=%0d fc_done=%0d done=%0d",
                  dut.tstate, dut.c1_done, dut.br1_done, dut.c2_done, dut.br2_done, dut.fc_done, dut.done);
        $display("  c1_start=%0d br1_start=%0d c2_start=%0d br2_start=%0d fc_start=%0d",
                  dut.c1_start, dut.br1_start, dut.c2_start, dut.br2_start, dut.fc_start);
        $finish;
    end

endmodule





