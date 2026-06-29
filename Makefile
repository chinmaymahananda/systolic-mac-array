# Systolic MAC Array - simulation makefile (Icarus Verilog)
#
#   make        # compile + run the self-checking testbench
#   make wave   # run, then open the waveform in GTKWave
#   make lint   # lint with Verilator (if installed)
#   make clean

IVERILOG ?= iverilog
VVP      ?= vvp
GTKWAVE  ?= gtkwave
VERILATOR?= verilator

RTL  := rtl/pe.v rtl/systolic_array.v
TB   := tb/tb_systolic.v
OUT  := sim/tb
VCD  := sim/dump.vcd

.PHONY: all sim wave lint clean

all: sim

sim: $(OUT)
	$(VVP) $(OUT)

$(OUT): $(RTL) $(TB)
	$(IVERILOG) -g2012 -o $(OUT) $(RTL) $(TB)

wave: sim
	$(GTKWAVE) $(VCD) &

lint:
	$(VERILATOR) --lint-only -Wall $(RTL)

clean:
	rm -f $(OUT) $(VCD)
