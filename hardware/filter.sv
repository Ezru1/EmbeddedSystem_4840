module	filter #(
	
		parameter WIDTH=21,
		parameter LGWIDTH=9,
		parameter LGP1=LGWIDTH+1,
		parameter  LGFILTWIDTH=3,
		parameter COEFFILE="hpf.hex"
		
	) (
		// {{{
		
		input wire	[(LGWIDTH - 1):0] i_cutoff,

		input	wire	i_clk, i_ce,
        input	wire	[(2*WIDTH-1):0]	i_sample,
		input 	wire 	i_sync,

		output  reg 	o_ce,
		output	reg	    [(2*WIDTH-1):0]	o_sample,
	    output	reg				o_sync
		// }}}
	);
 
	reg[(LGWIDTH):0] bin_idx;

	wire signed [WIDTH - 1: 0] sample_re, sample_im;
	reg signed [WIDTH - 1: 0] out_re, out_im;


	wire [LGFILTWIDTH: 0] filter_width; 
	assign filter_width = 'b1 << LGFILTWIDTH;

	wire signed [2 * WIDTH - 1: 0] fixed_point_sample_re;
	wire signed [2 * WIDTH - 1: 0] fixed_point_sample_im;

	wire [LGWIDTH - 1: 0] 	filter_idx;


	reg	signed [(2*WIDTH-1):0]	coeff;

	wire [LGWIDTH - 1:0] symm_bin_idx;


	// Read in coefficients
    reg	[(2*WIDTH-1):0]	cmem [0:((1<<LGWIDTH)-1)];
	initial	$readmemh(COEFFILE,cmem);  

		
	assign sample_re = i_sample[2 * WIDTH - 1 : WIDTH];
	assign sample_im = i_sample[WIDTH - 1: 0];


	// Symmetrical index for real series DFT
	assign symm_bin_idx = (bin_idx) < ('b1 << (LGWIDTH - 1)) ? (filter_width + bin_idx ): ('b1 << (LGWIDTH)) - (bin_idx - filter_width);


	// keep filter index within bounds of cmem
	assign filter_idx = symm_bin_idx > i_cutoff ? (symm_bin_idx < i_cutoff + filter_width ? symm_bin_idx - i_cutoff : filter_width - 1) : 0;

	// assign outputs
	assign fixed_point_sample_re = sample_re * coeff;
	assign fixed_point_sample_im = sample_im * coeff ;

	// assign o_sample[2 * WIDTH - 1 : WIDTH] = fixed_point_sample_re[2*WIDTH - 1: WIDTH];
	// assign o_sample[WIDTH - 1: 0] = fixed_point_sample_im[2*WIDTH - 1: WIDTH];


	always @(posedge i_clk) begin  
  
		o_ce <= i_ce;
		o_sync <= i_sync;
 
		
		if (i_ce) begin 
			if (i_sync) begin 
				bin_idx <= 1 ;
				coeff <= cmem[filter_width > i_cutoff ? (0 < i_cutoff ? filter_width - i_cutoff : filter_width - 1) : 0];
				
			end 

			else begin
				bin_idx <= bin_idx + 1; 
				coeff <= cmem[filter_idx];
				
			end  

			o_sample[2 * WIDTH - 1 : WIDTH] <= fixed_point_sample_re[2*WIDTH - 1: WIDTH];
			o_sample[WIDTH - 1: 0] <= fixed_point_sample_im[2*WIDTH - 1: WIDTH];



		end
  
	end
  

endmodule
