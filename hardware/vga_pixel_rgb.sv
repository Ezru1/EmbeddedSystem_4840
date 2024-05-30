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
   logic [15:0]	   pos;
   logic 	   read_ena, write_ena;
   logic [16:0]    address_read, address_write;
   logic [23:0]    data_in, data_out;
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
      if (VGA_BLANK_n) begin
	      write_ena <= 1;
	     // data_in <= {8'hff, 8'h0, 8'hff};
	      data_in <= {background_r, background_g, background_b};
	      address_write <= v * 640 + h;
             // address_write <= ((v >> 3) * 80 + (h >> 3)) >= 800 ? 0 : (v >> 3) * 80 + (h >> 3);
	      if (write_ena) 
		      write_ena <= 0;
      end
      read_ena <= 1;
      address_read <= vcount * 640 + hcount[10:1];
     // address_read <= ((vcount >> 3) * 80 + (hcount[10:1] >> 3)) >= 800 ? 0 : (vcount >> 3) * 80 + (hcount[10:1] >> 3);
      {VGA_R,VGA_G,VGA_B} <= data_out;
      //read_ena <= 0;
   end 
/*   
   always_comb begin	 
	h 
   end
*/
/*
   always_comb begin
	h = (h1<<8) + h2;
	v = (v1<<8) + v2;
	{VGA_R,VGA_G,VGA_B} = 24'b0;
	if (VGA_BLANK_n) begin
	    if ((hcount[10:1] >> 3 == h >> 3) && (vcount >> 3 == v >> 3)) begin
		//VGAmem[(h >> 3) * (v >> 3)] = {background_r,background_g,background_b};
		{VGA_R,VGA_G,VGA_B} = {8'h0, 8'h0, 8'hff};
 	    end
	    //{VGA_R,VGA_G,VGA_B} = VGAmem[pos];
	end
   end 
*/
/*   
   always_comb begin
	h = (h1<<8) + h2;
	v = (v1<<8) + v2;
	if ((h >> 3) * (v >> 3) >= 800)
	   (h >> 3) * (v >> 3) = 0;
	{VGA_R,VGA_G,VGA_B} = VGAmem[(h >> 3) * (v >> 3)];
   end 
*/
   
/*   always_comb begin
	h = (h1<<8) + h2;
	v = (v1<<8) + v2;
	if (!VGA_BLANK_n) begin
	    {VGA_R,VGA_G,VGA_B} = VGAmem[(h >> 3) * (v >> 3)];
        end
	else begin
	    if ((hcount[10:1] == h) && (vcount == v )) begin
		{VGA_R,VGA_G,VGA_B} = {background_r,background_g,background_b};
		VGAmem[(h >> 3) * (v >> 3)] = {background_r,background_g,background_b};
 	    end
            else begin
                {VGA_R,VGA_G,VGA_B} = VGAmem[(h >> 3) * (v >> 3)];
            end
        end
        
     end */
	       
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
 	input logic [23:0] data_in,
 	output logic [23:0] data_out,
 	input logic read_ena, write_ena);

 	reg [23:0] mem [307199:0];

 	always_ff@(posedge clk) begin
        	if (write_ena)
                	mem[address_write] = data_in;
       	 	if (read_ena)
                	data_out = mem[address_read];
 	end
endmodule

