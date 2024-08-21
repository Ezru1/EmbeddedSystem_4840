/*
 * Avalon memory-mapped peripheral that generates VGA
 *
 * Stephen A. Edwards
 * Columbia University
 */



module vga_pixel(input logic        clk,
	        input logic 	   reset,
		input logic [31:0]  writedata,
		input logic 	   write,
		input 		   chipselect,
		input logic [7:0]  address,
		
		
		output logic [7:0] VGA_R, VGA_G, VGA_B,
		output logic 	   VGA_CLK, VGA_HS, VGA_VS,
		                   VGA_BLANK_n,
		output logic 	   VGA_SYNC_n);

   logic [10:0]	   hcount;
   logic [9:0]     vcount;
   logic [7:0] 	   background_r;
   logic [15:0]     h,v;
   logic 	   read_ena = 0, write_ena = 1;
   logic [16:0]    address_read, address_write;
   logic [31:0]    data_in;
   logic [7:0] 	  data_out;
   logic [7:0]    vgalum;
   logic [18:0]   temp_add;
   logic [7:0]    lum[3:0];
   logic [1:0]    bias;
	
   vga_counters counters(.clk50(clk), .*);
   
   
   memory mem(.*);
   
            
   always_ff @(posedge clk)begin	
      if (chipselect && write)
      	case (address)
         4'h0 : begin
         	data_in <= writedata;
         	write_ena <= 1;
         end
         4'h1 : begin
         	h <= writedata[31:16];
         	v <= writedata[15:0];
         end
        endcase
        address_write <= (v * 640 + h) >> 2;
        if (write_ena == 1) write_ena <= 0;
   end
   reg [31:0] temp;
   
   always_ff @(posedge clk)begin
	address_read = (vcount * 640 + hcount[10:1]) >> 2;
	bias = hcount[10:1] % 4;
	read_ena = (vcount * 640 + hcount[10:1] < 256000 && hcount[10:1] > 0 && hcount[10:1] < 632) ? 1 : 0;
	{VGA_R,VGA_G,VGA_B} <= {data_out, data_out, data_out};  
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
 	input logic [16:0] address_read, address_write,
 	input logic [31:0] data_in,
 	input logic [1:0] bias,
 	output logic [7:0] data_out,
 	input logic read_ena, write_ena);

 	reg [31:0] mem [65535:0];
 	reg [31:0] A;
 	
 	always_ff@(posedge clk) begin
        	if (write_ena) begin
                	mem[address_write] = data_in;
                end
       	 	if (read_ena) begin
       	 		A = mem[address_read];
                	data_out <= {A[(bias<<3)+7],A[(bias<<3)+6],A[(bias<<3)+5],A[(bias<<3)+4],A[(bias<<3)+3],A[(bias<<3)+2],A[(bias<<3)+1],A[(bias<<3)]};
               	end else
                  data_out <= 0;
                
                
 	end	
 	
endmodule

