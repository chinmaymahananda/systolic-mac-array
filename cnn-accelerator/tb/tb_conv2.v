// tb_conv2.v -- bit-exact check of Conv2 against golden_hex_v2 vectors
`timescale 1ns/1ps

module tb_conv2;
    reg clk = 0, rst_n = 0, start = 0;
    wire done;
    integer sample, errors, c, px;
    reg [8*64:0] fname;
    reg signed [31:0] golden_acc [0:7][0:15];

    top_conv2_accelerator dut (.clk(clk), .rst_n(rst_n), .start(start), .done(done));
    always #5 clk = ~clk;

    task load_golden(input integer s);
        reg [31:0] flat_mem [0:127]; // 8 channels x 16 pixels
        integer cc, pp;
        begin
            $sformat(fname, "golden_hex_v2/conv2_acc_%0d.hex", s);
            $readmemh(fname, flat_mem);
            for (cc = 0; cc < 8; cc = cc + 1)
                for (pp = 0; pp < 16; pp = pp + 1)
                    golden_acc[cc][pp] = flat_mem[cc*16 + pp];
        end
    endtask

    initial begin
        sample = 0;
        if ($value$plusargs("SAMPLE=%d", sample)) ;

        $sformat(fname, "golden_hex_v2/conv2_input_%0d.hex", sample);
        $readmemh(fname, dut.u_img_rom.mem);
        load_golden(sample);

        errors = 0;
        rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10 start = 1;
        wait (done == 1);
        #10;

        for (px = 0; px < 16; px = px + 1) begin
            if (dut.u_ram_ch0.mem[px] !== golden_acc[0][px]) begin errors=errors+1; $display("MISMATCH ch0 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch0.mem[px], golden_acc[0][px]); end
            if (dut.u_ram_ch1.mem[px] !== golden_acc[1][px]) begin errors=errors+1; $display("MISMATCH ch1 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch1.mem[px], golden_acc[1][px]); end
            if (dut.u_ram_ch2.mem[px] !== golden_acc[2][px]) begin errors=errors+1; $display("MISMATCH ch2 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch2.mem[px], golden_acc[2][px]); end
            if (dut.u_ram_ch3.mem[px] !== golden_acc[3][px]) begin errors=errors+1; $display("MISMATCH ch3 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch3.mem[px], golden_acc[3][px]); end
            if (dut.u_ram_ch4.mem[px] !== golden_acc[4][px]) begin errors=errors+1; $display("MISMATCH ch4 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch4.mem[px], golden_acc[4][px]); end
            if (dut.u_ram_ch5.mem[px] !== golden_acc[5][px]) begin errors=errors+1; $display("MISMATCH ch5 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch5.mem[px], golden_acc[5][px]); end
            if (dut.u_ram_ch6.mem[px] !== golden_acc[6][px]) begin errors=errors+1; $display("MISMATCH ch6 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch6.mem[px], golden_acc[6][px]); end
            if (dut.u_ram_ch7.mem[px] !== golden_acc[7][px]) begin errors=errors+1; $display("MISMATCH ch7 px%0d rtl=%0d golden=%0d", px, dut.u_ram_ch7.mem[px], golden_acc[7][px]); end
        end

        if (errors == 0)
            $display("SAMPLE %0d: PASS -- all 128 Conv2 accumulator values bit-exact match golden model", sample);
        else
            $display("SAMPLE %0d: FAIL -- %0d/128 mismatches", sample, errors);

        $finish;
    end
endmodule
