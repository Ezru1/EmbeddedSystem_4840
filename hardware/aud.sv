/*
 * Avalon memory-mapped peripheral that handles audio inputs
 * Utilizes a audio driver that handles Altera driver files.
 * Functions to both store 1.5 seconds of 48khz audio data into BRAM for the Avalon bus to read out
 * Also can send to software the readout the result of the detector moudule.
 * Alex Yu, Sienna Brent, Rajat Tyagi, Riona westphal
 * Columbia University
 */

`include "global_variables.sv"
`include "./AudioCodecDrivers/audio_driver.sv"

//`define RAM_ADDR_BITS 5'd16
//`define RAM_WORDS 16'd48000

// 7-Seg dispaly for debugging

module hex7seg(input logic  [3:0] a,
               output logic [6:0] y);
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

module audio_control( 
        input logic [3:0]         KEY, // Pushbuttons; KEY[0] is rightmost
        // 7-segment LED displays; HEX0 is rightmost
        //output logic [6:0]        HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, 
        
        //Audio pin assignments
        //Used because Professor Scott Hauck and Kyle Gagner
        output logic              FPGA_I2C_SCLK,
        inout                     FPGA_I2C_SDAT,
        output logic              AUD_XCK,
        input logic               AUD_ADCLRCK,
        input logic               AUD_DACLRCK,
        input logic               AUD_BCLK,
        input logic               AUD_ADCDAT,
        output logic              AUD_DACDAT,
        
        //Driver IO ports
        input logic               clk,
        input logic               reset,
        input logic [31:0]        writedata,
        input logic               write,
        input logic               read,
        input                     chipselect,
        input logic [15:0]        address,
        output logic [31:0]       readdata,     

	output logic [6:0]        HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, 
        //Bram controls
        output logic [15:0]       bram_wa,
        output logic [15:0]       bram_ra,
        output logic              bram_write = 0,
        output logic [23:0]       bram_data_in,
        input logic [23:0]        bram_data_out

        );

    //Audio Controller
    reg [23:0]      dac_left_in;
    reg [23:0]      dac_right_in;
    logic [23:0]     adc_left_out;
    logic [23:0]     adc_right_out;
    
    reg [20:0]      sqr;
    // wire advance;
    
    //Device drivers from Altera modified by Professor Scott Hauck and Kyle Gagner in Verilog
    audio_driver aDriver(
        .CLOCK_50(clk), 
        .reset(reset), 
        .dac_left(dac_left_in), 
        .dac_right(dac_right_in), 
        .adc_left(adc_left_out), 
        .adc_right(adc_right_out), 
        .advance(advance), 
        .FPGA_I2C_SCLK(FPGA_I2C_SCLK), 
        .FPGA_I2C_SDAT(FPGA_I2C_SDAT), 
        .AUD_XCK(AUD_XCK), 
        .AUD_DACLRCK(AUD_DACLRCK), 
        .AUD_ADCLRCK(AUD_ADCLRCK), 
        .AUD_BCLK(AUD_BCLK), 
        .AUD_ADCDAT(AUD_ADCDAT), 
        .AUD_DACDAT(AUD_DACDAT)
        );

    //Instantiate hex decoders
    logic [23:0]    hexout_buffer;
  
    
    reg [15:0] ra = 0,wa = 0;
    reg [23:0] rd,wd;
    reg        wena = 0,rena = 0;
    logic      clk_2;

    twoportbram Bram(
	 .clk(clk_2),
	 .reset(reset),
	 .ra(ra),
  	 .wa(wa),
	 .write(wena),
	 .read(rena),
	 .data_in(wd),
	 .data_out(rd)
    );
    
    hex7seg h5( .a(angle[5:4]),.y(HEX5) ), // left digit
            h4( .a(angle[3:0]),.y(HEX4) ), 
            h3( .a(0),.y(HEX3) ), 
            h2( .a(0),.y(HEX2) ),
            h1( .a(0),.y(HEX1) ),
            h0( .a(0),.y(HEX0) );   
            
        	
	
    //Convert stereo input to mono        
    logic [23:0]     adc_mono;
    logic  [23:0]    buffer;
    logic [25:0]     CC = 26'b0;
    logic [31:0]     CC2 = 32'b1;
    logic [5:0]      angle = 6'd45;
    //debounce
    always_ff @(posedge clk)begin
   	logic ena0 = 1,ena1 = 1;
   	logic [31:0] presstime0,presstime1;
	CC2 <= CC2 + 1;
	if (CC2 == 32'hffffffff) CC2 <= 1;
   	if (KEY[0] == 0 && ena0 == 1) begin
   	    presstime0 <= CC2;
   	    if (angle < 6'd60) angle <= angle + 1;
   	end
   	if (CC2 < presstime0 + 1 || KEY[0] == 0) begin
   	    ena0 <= 0;
   	end else begin 
   	    ena0 <= 1;
   	    presstime0 <= 0;
   	end
   	if (KEY[1] == 0 && ena1 == 1) begin
   	    presstime1 <= CC2;
   	    if (angle > 6'd30) angle <= angle - 1;
   	end
   	if (CC2 < presstime1 + 1 || KEY[1] == 0) begin
   	    ena1 <= 0;
   	end else begin 
   	    ena1 <= 1;
   	    presstime1 <= 0;
   	end
    end
    
    logic [3:0]     bram_input_ctrl;
    logic [23:0]    result_buffer;
    logic [23:0]    adc_out_buffer;
    logic           write_clk;
    logic 	    left_fft_ce;	
    logic 	    left_fft_o_sync;
    logic [41:0]    o_result;
    logic[15:0]     inp;

    reg [23:0]      Ctst = 0;
    logic cena = 1;
    
    assign left_fft_ce = 1;
    assign clk_2 = CC[13];
    assign wd = {3'b0, sqr};

    fftmain left_fft(
		.i_clk(clk_2),
		.i_reset(reset),
		.i_ce(left_fft_ce),
		.i_sample({adc_mono[23:8], 16'b0}),
		.o_result(o_result),
		.o_sync(left_fft_o_sync));  
    function [23:0] A;
	      	input [23:0] num;  //declare input
	      
		//intermediate signals.
	      	reg [23:0] absn;
		    	begin
		      	if (num[23] == 0)
			  absn = {1'b0,num[23:1]};
			else
			  absn = {1'b1,num[23:1]};
			A = absn;
		end
    endfunction //end of Function
    function [20:0] sqrt;
      	input [20:0] num1;  //declare input
      	input [20:0] num2;
      
        //intermediate signals.
      	reg [20:0] absn1,absn2;
      	logic [41:0] n1 = 0,n2 = 0;
      	reg [41:0] a;
      	reg [23:0] q;
      	reg [25:0] left,right,r;   
    	begin
      	//do square
      	if (num1[20] == 0)
          absn1 = num1;
        else
          absn1 = -num1;
      	if (num2[20] == 0)
          absn2 = num2;
        else
          absn2 = -num2;
          for(integer i = 0; i < 20; i++)begin
          if (absn1[i])
            n1 += absn1 << i;
          if (absn2[i])
            n2 += absn2 << i;
        end
        //initialize all the variables.
        a = n1 + n2;
        q = 0;
        left = 0;   //input to adder/sub
        right = 0;  //input to adder/sub
        r = 0;  //remainder
        //run the calculations for 16 iterations.
      	for(integer i = 0; i < 21; i++) begin 
        	right = {q,r[25],1'b1};
          	left = {r[23:0],a[41:40]};
          	a = {a[39:0],2'b00};    //left shift by 2 bits.
        	if (r[25] == 1) //add if r is negative
                r = left + right;
            else    //subtract if r is positive
                r = left - right;
        	q = {q[22:0],!r[25]};       
        end
        sqrt = q;   //final assignment of output.
    end
    endfunction //end of Function

    always_comb begin
	sqr = sqrt(o_result[20:0],o_result[41:21]);
        buffer = rd;
        //buffer = A(adc_mono);
    end

    //Determine when the driver is in the middle of pulling a sample
    //by default dont use the BRAM module
    logic           bram_writing = 0;
    logic           bram_reading = 0;
    logic [31:0]    driverReading = 31'd0;
    logic [15:0]    limit;

    always_ff @(posedge clk) begin
    	adc_mono <= A(adc_right_out + adc_left_out);
	//adc_mono <= adc_left_out;
	CC += 1;
    end
   
    always_ff @(posedge clk_2) begin
	if(wena) wa += 1;
    end

    always_ff @(posedge clk) begin : IOcalls
        // ioread recieved
        //adc_mono <= adc_left_out;
        if (chipselect && read) begin
            case (address)
                16'h0002: begin
                    readdata[31:6] <= 0;
                    readdata[5:0] <= angle;
                end
                16'h0003 : begin
		    rena <= 1;
		    wena <= 1;
		    ra <= ra + 1;
                    // return padded buffer
                    if (buffer[23] == 1) begin 
                        readdata[23:0] <= buffer[23:0];
                        readdata[31:24] <= 8'b11111111;
                    end
                    else if (buffer[23] == 0) begin
                        readdata[23:0] <= buffer[23:0];
                        readdata[31:24] <= 8'b00000000;
                    end
		    
                end
            endcase
        end
    end

    wire sampleBeingTaken;
    assign sampleBeingTaken = driverReading[0];
    
    //Map timer(Sample) counter output
    parameter readOutSize = 16'hffff;
    //Sample inputs/Audio passthrough

endmodule
