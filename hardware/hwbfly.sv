////////////////////////////////////////////////////////////////////////////////
//
// Filename:	hwbfly.v
// {{{
// Project:	A General Purpose Pipelined FFT Implementation
//
// Purpose:	This routine is identical to the butterfly.v routine found
//		in 'butterfly.v', save only that it uses the verilog
//	operator '*' in hopes that the synthesizer would be able to optimize
//	it with hardware resources.
//
//	It is understood that a hardware multiply can complete its operation in
//	a single clock.
//
// Operation:
//
//	Given two inputs, A (i_left) and B (i_right), and a complex
//	coefficient C (i_coeff), return two outputs, O1 and O2, where:
//
//		O1 = A + B, and
//		O2 = (A - B)*C
//
//	This operation is commonly known as a Decimation in Frequency (DIF)
//	Radix-2 Butterfly.
//	O1 and O2 are rounded before being returned in (o_left) and o_right
//	to OWIDTH bits.  If SHIFT is one, an extra bit is dropped from these
//	values during the rounding process.
//
//	Further, since these outputs will take some number of clocks to
//	calculate, we'll pipe a value (i_aux) through the system and return
//	it with the results (o_aux), so you can synchronize to the outgoing
//	output stream.
//
//
// Creator:	Dan Gisselquist, Ph.D.
//		Gisselquist Technology, LLC
//
////////////////////////////////////////////////////////////////////////////////
// }}}
// Copyright (C) 2015-2024, Gisselquist Technology, LLC
// {{{
// This file is part of the general purpose pipelined FFT project.
//
// The pipelined FFT project is free software (firmware): you can redistribute
// it and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// The pipelined FFT project is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  (It's in the $(ROOT)/doc directory.  Run make
// with no target there if the PDF file isn't present.)  If not, see
// <http://www.gnu.org/licenses/> for a copy.
// }}}
// License:	LGPL, v3, as defined and found on www.gnu.org,
// {{{
//		http://www.gnu.org/licenses/lgpl.html
//
// }}}
////////////////////////////////////////////////////////////////////////////////
//
//
`default_nettype	none
//
module	hwbfly #(
		// {{{
		// Public changeable parameters ...
		//	- IWIDTH, number of bits in each component of the input
		//	- CWIDTH, number of bits in each component of the
		//	twiddle factor
		//	- OWIDTH, number of bits in each component of the
		//	output
		parameter IWIDTH=16,CWIDTH=IWIDTH+4,OWIDTH=IWIDTH+1,
		// SHIFT
		// {{{
		// Drop an additional bit on the output?
		parameter		SHIFT=0,
		// }}}
		// CKPCE
		// {{{
		// The number of clocks per clock enable, 1, 2, or 3.
		parameter	[1:0]	CKPCE=1
		// }}}
		// }}}
	) (
		// {{{
		input	wire	i_clk, i_reset, i_ce,
		input	wire	[(2*CWIDTH-1):0]	i_coef,
		input	wire	[(2*IWIDTH-1):0]	i_left, i_right,
		input	wire	i_aux,
		output	wire	[(2*OWIDTH-1):0]	o_left, o_right,
		output	reg	o_aux
		// }}}
	);

	// Local signal declarations
	// {{{
	reg	[(2*IWIDTH-1):0]	r_left, r_right;
	reg				r_aux, r_aux_2;
	reg	[(2*CWIDTH-1):0]	r_coef;
	wire	signed	[(IWIDTH-1):0]	r_left_r, r_left_i, r_right_r, r_right_i;
	reg	signed	[(CWIDTH-1):0]	ir_coef_r, ir_coef_i;

	reg	signed	[(IWIDTH):0]	r_sum_r, r_sum_i, r_dif_r, r_dif_i;

	reg	[(2*IWIDTH+2):0]	leftv, leftvv;

	wire	signed	[((IWIDTH+1)+(CWIDTH)-1):0]	p_one, p_two;
	wire	signed	[((IWIDTH+2)+(CWIDTH+1)-1):0]	p_three;

	wire	signed	[((IWIDTH+2)+(CWIDTH+1)-1):0]	w_one, w_two;

	wire	aux_s;
	wire	signed	[(IWIDTH+CWIDTH):0]	left_si, left_sr;
	reg		[(2*IWIDTH+2):0]	left_saved;
	(* use_dsp48="no" *)
	reg	signed	[(CWIDTH+IWIDTH+3-1):0]	mpy_r, mpy_i;

	wire	signed	[(OWIDTH-1):0]	rnd_left_r, rnd_left_i, rnd_right_r, rnd_right_i;
	// }}}

	assign	r_left_r  = r_left[ (2*IWIDTH-1):(IWIDTH)];
	assign	r_left_i  = r_left[ (IWIDTH-1):0];
	assign	r_right_r = r_right[(2*IWIDTH-1):(IWIDTH)];
	assign	r_right_i = r_right[(IWIDTH-1):0];

	// Set up the input to the multiply

	// r_aux, r_aux_2
	// {{{
	initial r_aux   = 1'b0;
	initial r_aux_2 = 1'b0;
	always @(posedge i_clk)
	if (i_reset)
	begin
		r_aux <= 1'b0;
		r_aux_2 <= 1'b0;
	end else if (i_ce)
	begin
		// One clock just latches the inputs
		r_aux <= i_aux;
		// Next clock adds/subtracts
		// Other inputs are simply delayed on second clock
		r_aux_2 <= r_aux;
	end
	// }}}

	// r_[left|right|coef], r_[sum|dif]_[r|i], ir_coef_[r|i]
	// {{{
	always @(posedge i_clk)
	if (i_ce)
	begin
		// One clock just latches the inputs
		r_left <= i_left;	// No change in # of bits
		r_right <= i_right;
		r_coef  <= i_coef;
		// Next clock adds/subtracts
		r_sum_r <= r_left_r + r_right_r; // Now IWIDTH+1 bits
		r_sum_i <= r_left_i + r_right_i;
		r_dif_r <= r_left_r - r_right_r;
		r_dif_i <= r_left_i - r_right_i;
		// Other inputs are simply delayed on second clock
		ir_coef_r <= r_coef[(2*CWIDTH-1):CWIDTH];
		ir_coef_i <= r_coef[(CWIDTH-1):0];
	end
	// }}}


	// See comments in the butterfly.v source file for a discussion of
	// these operations and the appropriate bit widths.


	// leftv, leftvv
	// {{{
	initial leftv    = 0;
	initial leftvv   = 0;
	always @(posedge i_clk)
	if (i_reset)
	begin
		leftv <= 0;
		leftvv <= 0;
	end else if (i_ce)
	begin
		// Second clock, pipeline = 1
		leftv <= { r_aux_2, r_sum_r, r_sum_i };

		// Third clock, pipeline = 3
		//   As desired, each of these lines infers a DSP48
		leftvv <= leftv;
	end
	// }}}

	// Core multiply section
	// {{{
	generate if (CKPCE <= 1)
	begin : CKPCE_ONE
		// {{{
		// Local declarations
		// {{{
		// Coefficient multiply inputs
		reg	signed	[(CWIDTH-1):0]	p1c_in, p2c_in;
		// Data multiply inputs
		reg	signed	[(IWIDTH):0]	p1d_in, p2d_in;
		// Product 3, coefficient input
		reg	signed	[(CWIDTH):0]	p3c_in;
		// Product 3, data input
		reg	signed	[(IWIDTH+1):0]	p3d_in;

		reg	signed	[((IWIDTH+1)+(CWIDTH)-1):0]	rp_one, rp_two;
		reg	signed	[((IWIDTH+2)+(CWIDTH+1)-1):0]	rp_three;
		// }}}

		// p[1|2]c_in, p[1|2|3]d_in
		// {{{
		always @(posedge i_clk)
		if (i_ce)
		begin
			// Second clock, pipeline = 1
			p1c_in <= ir_coef_r;
			p2c_in <= ir_coef_i;
			p1d_in <= r_dif_r;
			p2d_in <= r_dif_i;
			p3c_in <= ir_coef_i + ir_coef_r;
			p3d_in <= r_dif_r + r_dif_i;
		end
		// }}}

		// Perform our multiplies

		always @(posedge i_clk)
		if (i_ce)
		begin
			// Third clock, pipeline = 3
			//   As desired, each of these lines infers a DSP48
			rp_one   <= p1c_in * p1d_in;
			rp_two   <= p2c_in * p2d_in;
			rp_three <= p3c_in * p3d_in;
		end

		// }}}

		assign	p_one   = rp_one;
		assign	p_two   = rp_two;
		assign	p_three = rp_three;
		// }}}
	end  endgenerate
	// }}}

	assign	w_one = { {(2){p_one[((IWIDTH+1)+(CWIDTH)-1)]}}, p_one };
	assign	w_two = { {(2){p_two[((IWIDTH+1)+(CWIDTH)-1)]}}, p_two };

	// left_sr, left_si
	// {{{
	// These values are held in memory and delayed during the
	// multiply.  Here, we recover them.  During the multiply,
	// values were multiplied by 2^(CWIDTH-2)*exp{-j*2*pi*...},
	// therefore, the left_x values need to be right shifted by
	// CWIDTH-2 as well.  The additional bits come from a sign
	// extension.
	assign	left_sr = { {2{left_saved[2*(IWIDTH+1)-1]}}, left_saved[(2*(IWIDTH+1)-1):(IWIDTH+1)], {(CWIDTH-2){1'b0}} };
	assign	left_si = { {2{left_saved[(IWIDTH+1)-1]}}, left_saved[((IWIDTH+1)-1):0], {(CWIDTH-2){1'b0}} };
	// }}}
	assign	aux_s = left_saved[2*IWIDTH+2];

	// left_saved, o_aux
	// {{{
	initial left_saved = 0;
	initial o_aux      = 1'b0;
	always @(posedge i_clk)
	if (i_reset)
	begin
		left_saved <= 0;
		o_aux <= 1'b0;
	end else if (i_ce)
	begin
		// First clock, recover all values
		left_saved <= leftvv;

		// Second clock, round and latch for final clock
		o_aux <= aux_s;
	end
	// }}}

	// mpy_r, mpy_i
	// {{{
	always @(posedge i_clk)
	if (i_ce)
	begin
		// These values are IWIDTH+CWIDTH+3 bits wide
		// although they only need to be (IWIDTH+1)
		// + (CWIDTH) bits wide.  (We've got two
		// extra bits we need to get rid of.)

		// These two lines also infer DSP48's.
		// To keep from using extra DSP48 resources,
		// they are prevented from using DSP48's
		// by the (* use_dsp48 ... *) comment above.
		mpy_r <= w_one - w_two;
		mpy_i <= p_three - w_one - w_two;
	end
	// }}}

	// Round the results
	// {{{
	convround #(CWIDTH+IWIDTH+1,OWIDTH,SHIFT+2)
	do_rnd_left_r(i_clk, i_ce, left_sr, rnd_left_r);

	convround #(CWIDTH+IWIDTH+1,OWIDTH,SHIFT+2)
	do_rnd_left_i(i_clk, i_ce, left_si, rnd_left_i);

	convround #(CWIDTH+IWIDTH+3,OWIDTH,SHIFT+4)
	do_rnd_right_r(i_clk, i_ce, mpy_r, rnd_right_r);

	convround #(CWIDTH+IWIDTH+3,OWIDTH,SHIFT+4)
	do_rnd_right_i(i_clk, i_ce, mpy_i, rnd_right_i);
	// }}}

	// o_left, o_right
	// {{{
	// As a final step, we pack our outputs into two packed two's
	// complement numbers per output word, so that each output word
	// has (2*OWIDTH) bits in it, with the top half being the real
	// portion and the bottom half being the imaginary portion.
	assign	o_left = { rnd_left_r, rnd_left_i };
	assign	o_right= { rnd_right_r,rnd_right_i};
// }}}


endmodule
