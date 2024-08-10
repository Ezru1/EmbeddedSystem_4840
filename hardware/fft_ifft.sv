
/* verilator lint_off UNUSEDSIGNAL */

module	fft_ifft #(
    parameter   FFT_IWIDTH = 16,
    parameter   FFT_OWIDTH = 21,
    parameter   IFFT_IWIDTH = 21,
    parameter   IFFT_OWIDTH = 26,
    parameter    LGWIDTH = 9
)(
    input               i_clk,  //  Master Clock
    input               i_ce,
    input               i_reset,
    input    [(2 * FFT_IWIDTH - 1): 0] i_sample,

    input    [FFT_IWIDTH - 1: 0] writedata,
    input               write,

    input               read,

    input               address,
    


    output   [FFT_IWIDTH - 1: 0] readdata ,
    output  [(2 * IFFT_OWIDTH - 1): 0]o_result,   //  Output Data (Imag)
    output  o_sync   //  Output Data (Imag)
);


wire fft_o_sync;
reg ifft_i_reset;

wire [(2 * FFT_OWIDTH - 1) : 0] fft_o_result;
wire [(2 * IFFT_IWIDTH - 1) : 0] ifft_i_sample;

wire [IFFT_OWIDTH - 1: 0] out_re;
wire [IFFT_OWIDTH - 1: 0] out_im;

assign out_re = o_result[2 * IFFT_OWIDTH - 1: IFFT_OWIDTH];
assign out_im = o_result[IFFT_OWIDTH - 1: 0];



wire	w_syncd;
reg	    r_syncd;

always @(posedge i_clk) begin

	if (i_reset)
		r_syncd <= 1'b0;
	else
		r_syncd <= r_syncd || fft_o_sync;

	// ifft_i_sample <= fft_o_result
	

end

assign ifft_i_sample = fft_o_result;

assign	w_syncd = r_syncd || fft_o_sync;
 


fftmain fft(
    .i_clk(i_clk),
    .i_reset(i_reset),
    .i_ce(i_ce),
    .i_sample(i_sample),
    .o_result(fft_o_result),
    .o_sync(fft_o_sync));


reg [(LGWIDTH - 1):0] lpf_cutoff;

assign lpf_cutoff = 10'd10; 

wire lpf_o_ce, lpf_o_sync;
wire [(2 * IFFT_IWIDTH - 1) : 0] lpf_o_sample;

LPF #(
    .LGWIDTH(LGWIDTH),
    .LGFILTWIDTH(5)
) lpf (   
    .i_cutoff(lpf_cutoff),
    .i_clk(i_clk),
    .i_reset(fft_o_sync), 
    .i_ce(((i_ce)&&(w_syncd))),
    .i_sample(fft_o_result), 
    .i_sync(fft_o_sync),
    .o_ce(lpf_o_ce),
    .o_sample(lpf_o_sample), 
    .o_sync(lpf_o_sync));
 
ifftmain ifft(
    .i_clk(i_clk),
    .i_reset(i_reset),
    .i_ce(lpf_o_ce),
    .i_sample(lpf_o_sample),
    .o_result(o_result),
    .o_sync(o_sync));
	  
endmodule
 

