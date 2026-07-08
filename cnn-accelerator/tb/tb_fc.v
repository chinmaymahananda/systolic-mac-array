// tb_fc.v -- bit-exact check of FC layer against golden_hex_fc vectors
`timescale 1ns/1ps

module tb_fc;
    reg clk = 0, rst_n = 0, start = 0;
    wire done;
    integer sample, errors;
    reg [8*64:0] fname;
    reg signed [31:0] golden_logits [0:9];

    top_fc_accelerator dut (.clk(clk), .rst_n(rst_n), .start(start), .done(done));
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

    initial begin
        sample = 0;
        if ($value$plusargs("SAMPLE=%d", sample)) ;

        $sformat(fname, "golden_hex_fc/fc_input_%0d.hex", sample);
        $readmemh(fname, dut.u_img_rom.mem);
        load_golden(sample);

        errors = 0;
        rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10 start = 1;
        wait (done == 1);
        #10;

        if (dut.u_ram_ch0.rd_data !== golden_logits[0]) begin errors=errors+1; $display("MISMATCH ch0 rtl=%0d golden=%0d", dut.u_ram_ch0.rd_data, golden_logits[0]); end
        if (dut.u_ram_ch1.rd_data !== golden_logits[1]) begin errors=errors+1; $display("MISMATCH ch1 rtl=%0d golden=%0d", dut.u_ram_ch1.rd_data, golden_logits[1]); end
        if (dut.u_ram_ch2.rd_data !== golden_logits[2]) begin errors=errors+1; $display("MISMATCH ch2 rtl=%0d golden=%0d", dut.u_ram_ch2.rd_data, golden_logits[2]); end
        if (dut.u_ram_ch3.rd_data !== golden_logits[3]) begin errors=errors+1; $display("MISMATCH ch3 rtl=%0d golden=%0d", dut.u_ram_ch3.rd_data, golden_logits[3]); end
        if (dut.u_ram_ch4.rd_data !== golden_logits[4]) begin errors=errors+1; $display("MISMATCH ch4 rtl=%0d golden=%0d", dut.u_ram_ch4.rd_data, golden_logits[4]); end
        if (dut.u_ram_ch5.rd_data !== golden_logits[5]) begin errors=errors+1; $display("MISMATCH ch5 rtl=%0d golden=%0d", dut.u_ram_ch5.rd_data, golden_logits[5]); end
        if (dut.u_ram_ch6.rd_data !== golden_logits[6]) begin errors=errors+1; $display("MISMATCH ch6 rtl=%0d golden=%0d", dut.u_ram_ch6.rd_data, golden_logits[6]); end
        if (dut.u_ram_ch7.rd_data !== golden_logits[7]) begin errors=errors+1; $display("MISMATCH ch7 rtl=%0d golden=%0d", dut.u_ram_ch7.rd_data, golden_logits[7]); end
        if (dut.u_ram_ch8.rd_data !== golden_logits[8]) begin errors=errors+1; $display("MISMATCH ch8 rtl=%0d golden=%0d", dut.u_ram_ch8.rd_data, golden_logits[8]); end
        if (dut.u_ram_ch9.rd_data !== golden_logits[9]) begin errors=errors+1; $display("MISMATCH ch9 rtl=%0d golden=%0d", dut.u_ram_ch9.rd_data, golden_logits[9]); end

        if (errors == 0)
            $display("SAMPLE %0d: PASS -- all 10 FC logits bit-exact match golden model", sample);
        else
            $display("SAMPLE %0d: FAIL -- %0d/10 mismatches", sample, errors);

        $finish;
    end
endmodule
