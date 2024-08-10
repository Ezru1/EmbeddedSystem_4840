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
		
		output logic [6:0]        HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, 
		output logic [7:0] VGA_R, VGA_G, VGA_B,
		output logic 	   VGA_CLK, VGA_HS, VGA_VS,
		                   VGA_BLANK_n,
		output logic 	   VGA_SYNC_n);

   logic [10:0]	   hcount;
   logic [9:0]     vcount;
   logic [7:0] 	   background_r;
   logic [15:0]     h,v;
   logic 	   read_ena = 0, write_ena = 1;
   logic [19:0]    address_read, address_write;
   logic [7:0]    data_out;
   logic [7:0]    vgalum;
   logic [31:0]   data_in;
   logic [18:0]   temp_add;
   logic [31:0]    lum;
	
   vga_counters counters(.clk50(clk), .*);
   memory mem(.*);
   
   hex7seg h5( .a(h[11:8]),.y(HEX5) ), // left digit
            h4( .a(h[7:4]),.y(HEX4) ), 
            h3( .a(h[3:0]),.y(HEX3) ), 
            h2( .a(v[11:8]),.y(HEX2) ),
            h1( .a(v[7:4]),.y(HEX1) ),
            h0( .a(v[3:0]),.y(HEX0) );    
            
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
        //address_write <= v * 640 + h;
        address_write <= v * 640 + h;
        if (write_ena == 1) write_ena <= 0;
   end
   
   always_ff @(posedge clk)begin
	address_read <= vcount * 640 + hcount[10:1];
	read_ena <= (vcount * 640 + hcount[10:1] < 256000 && hcount[10:1] > 0 && hcount[10:1] < 630) ? 1 : 0;
	vgalum <= data_out;
   	{VGA_R,VGA_G,VGA_B} <= {vgalum, vgalum, vgalum};  
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

module hex7seg(input logic  [3:0] a,
               output logic [6:0] y);

   /* Replace this comment and the code below it with your solution */
    always_comb
        case (a)        //      gfe_dcba
            4'h0:        y = 7'b100_0000;
            4'h1:        y = 7'b111_1001;
            4'h2:        y = 7'b010_0100;
            4'h3:        y = 7'b011_0000;
            4'h4:        y = 7'b001_1001;
            4'h5:        y = 7'b001_0010;
            4'h6:        y = 7'b000_0010;
            4'h7:        y = 7'b111_1000;
            4'h8:        y = 7'b000_0000;
            4'h9:        y = 7'b001_0000;
            4'hA:        y = 7'b000_1000;
            4'hB:        y = 7'b000_0011;
            4'hC:        y = 7'b100_0110;
            4'hD:        y = 7'b010_0001;
            4'hE:        y = 7'b000_0110;
            4'hF:        y = 7'b000_1110;
            default:     y = 7'b111_1111;
        endcase
endmodule


module memory(
	input logic clk, reset,
 	input logic [19:0] address_read, address_write,
 	input logic [31:0] data_in,
 	output logic [7:0] data_out,
 	input logic read_ena, write_ena);

 	reg [7:0] mem [257199:0];
 	
 	always_ff@(posedge clk) begin
        	if (write_ena) begin
                	//mem[(address_write << 2) + 3] <= data_in[15:12];
                	//mem[(address_write << 2) + 2] <= data_in[11:8];
                	//mem[(address_write << 2) + 1] <= data_in;
                	mem[(address_write << 0) + 0] <= data_in;
                end
       	 	if (read_ena)
                	data_out <= mem[address_read];
            else
                  data_out <= 0;
 	end
endmodule

