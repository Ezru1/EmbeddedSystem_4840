/*
 * Avalon memory-mapped peripheral that generates VGA
 *
 * Stephen A. Edwards
 * Columbia University
 */

module vga_pixel(input logic        clk,
	        input logic 	   reset,
		input logic [7:0]  writedata,
		input logic 	   write,
		input 		   chipselect,
		input logic [2:0]  address,

		output logic [7:0] VGA_R, VGA_G, VGA_B,
		output logic 	   VGA_CLK, VGA_HS, VGA_VS,
		                   VGA_BLANK_n,
		output logic 	   VGA_SYNC_n);

   logic [10:0]	   hcount;
   logic [9:0]     vcount;
   logic [10:0]    h, v;
   logic [7:0]     h1, h2, v1, v2;
   logic [7:0] 	   background_r, background_g, background_b;
   logic 	   read_ena = 1, write_ena = 1;
   logic [18:0]    address_read, address_write;
   logic [7:0]    data_in, data_out;
   logic [18:0]   temp_add;
/*
   initial begin
       // Initialize VGAmem with default values
       for (int i = 0; i < 800; i = i + 1) begin
           VGAmem[i] = 24'b0;
       end
   end
*/
	
   vga_counters counters(.clk50(clk), .*);
   memory mem(.*);
   
//   initial begin
//	{VGA_R,VGA_G,VGA_B} <= {8'h0, 8'h0, 8'h80};
//   end

   always_ff @(posedge clk)begin
      if (reset) begin
	background_r <= 8'h0;
	background_g <= 8'h0;
	background_b <= 8'h80;
	
      end else if (chipselect && write)
      	case (address)
	 3'h0 : background_r <= writedata;
	 3'h1 : background_g <= writedata;
	 3'h2 : background_b <= writedata;
         3'h3 : h1 <= writedata;
         3'h4 : h2 <= writedata;
         3'h5 : v1 <= writedata;
         3'h6 : v2 <= writedata;
        endcase
      
        h = (h1<<8) + h2;
        v = (v1<<8) + v2;
        data_in <= background_r;
        address_write <= v * 640 + h;
	address_read <= vcount * 640 + hcount[10:1];
      	{VGA_R,VGA_G,VGA_B} <= {data_out, data_out, data_out};        
	if (VGA_BLANK_n) begin
	     #1000000;
	     //data_in <= {8'hff, 8'h0, 8'hff};
        end
	
   end

	       
endmodule


module vga_counters(
 input logic 	     clk50, reset,
 output logic [10:0] hcount,  // hcount[10:1] is pixel column
 output logic [9:0]  vcount,  // vcount[9:0] is pixel row
 output logic 	     VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_n, VGA_SYNC_n);
/*
 * 640 X 480 VGA timing for a 50 MHz clock: one pixel every other cycle
 * 
 * HCOUNT 1599 0             1279       1599 0
 *             _______________              ________
 * ___________|    Video      |____________|  Video
 * 
 * 
 * |SYNC| BP |<-- HACTIVE -->|FP|SYNC| BP |<-- HACTIVE
 *       _______________________      _____________
 * |____|       VGA_HS          |____|
 */
   // Parameters for hcount
   parameter HACTIVE      = 11'd 1280,
             HFRONT_PORCH = 11'd 32,
             HSYNC        = 11'd 192,
             HBACK_PORCH  = 11'd 96,   
             HTOTAL       = HACTIVE + HFRONT_PORCH + HSYNC +
                            HBACK_PORCH; // 1600
   
   // Parameters for vcount
   parameter VACTIVE      = 10'd 480,
             VFRONT_PORCH = 10'd 10,
             VSYNC        = 10'd 2,
             VBACK_PORCH  = 10'd 33,
             VTOTAL       = VACTIVE + VFRONT_PORCH + VSYNC +
                            VBACK_PORCH; // 525

   logic endOfLine;
   
   always_ff @(posedge clk50 or posedge reset)
     if (reset)          hcount <= 0;
     else if (endOfLine) hcount <= 0;
     else  	         hcount <= hcount + 11'd 1;

   assign endOfLine = hcount == HTOTAL - 1;
       
   logic endOfField;
   
   always_ff @(posedge clk50 or posedge reset)
     if (reset)          vcount <= 0;
     else if (endOfLine)
       if (endOfField)   vcount <= 0;
       else              vcount <= vcount + 10'd 1;

   assign endOfField = vcount == VTOTAL - 1;

   // Horizontal sync: from 0x520 to 0x5DF (0x57F)
   // 101 0010 0000 to 101 1101 1111
   assign VGA_HS = !( (hcount[10:8] == 3'b101) &
		      !(hcount[7:5] == 3'b111));
   assign VGA_VS = !( vcount[9:1] == (VACTIVE + VFRONT_PORCH) / 2);

   assign VGA_SYNC_n = 1'b0; // For putting sync on the green signal; unused
   
   // Horizontal active: 0 to 1279     Vertical active: 0 to 479
   // 101 0000 0000  1280	       01 1110 0000  480
   // 110 0011 1111  1599	       10 0000 1100  524
   assign VGA_BLANK_n = !( hcount[10] & (hcount[9] | hcount[8]) ) &
			!( vcount[9] | (vcount[8:5] == 4'b1111) );

   /* VGA_CLK is 25 MHz
    *             __    __    __
    * clk50    __|  |__|  |__|
    *        
    *             _____       __
    * hcount[0]__|     |_____|
    */
   assign VGA_CLK = hcount[0]; // 25 MHz clock: rising edge sensitive
   
endmodule


module memory(
	input logic clk, reset,
 	input logic [18:0] address_read, address_write,
 	input logic [7:0] data_in,
 	output logic [7:0] data_out,
 	input logic read_ena, write_ena);

 	reg [7:0] mem [307199:0];
/*		
	initial begin
		for (int i = 0; i < 4000; i = i + 1) begin
        	   mem[i] = 8'b11111111;
	       end
	end	
*/
 	always_ff@(posedge clk) begin
        	if (write_ena)
                	mem[address_write] = data_in;
       	 	if (read_ena)
                	data_out = mem[address_read];
 	end
endmodule
