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

        //Bram controls
        output logic [15:0]       bram_wa,
        output logic [15:0]       bram_ra,
        output logic              bram_write = 0,
        output logic [23:0]       bram_data_in,
        input logic [23:0]        bram_data_out

/*
        //goertzel control
        output logic [23:0]       adc_out_buffer,
        output logic              advance,
        
        //detector return
        input logic [2:0]         result,
        input logic [2:0]         overall_result,
        input logic               flag
*/
        );

    //Audio Controller
    reg [23:0]      dac_left_in;
    reg [23:0]      dac_right_in;
    wire [23:0]     adc_left_out;
    wire [23:0]     adc_right_out;
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
    /*hex7seg h5( .a(wa[15:12]),.y(HEX5) ), // left digit
            h4( .a(wa[11:8]),.y(HEX4) ), 
            h3( .a(wa[7:4]),.y(HEX3) ), 
            h2( .a(ra[11:8]),.y(HEX2) ),
            h1( .a(ra[7:4]),.y(HEX1) ),
            h0( .a(ra[3:0]),.y(HEX0) );    */
  
    
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

    //Convert stereo input to mono        
    logic [23:0]    audioInMono;
    logic  [23:0]    buffer;
    logic [25:0]     CC = 26'b0;
    
    
    logic [3:0]     bram_input_ctrl;
    logic [23:0]    result_buffer;
    logic [23:0]    adc_out_buffer;
    logic           write_clk;
    logic 	    left_fft_ce;	
    logic 	    left_fft_o_sync;
    logic [41:0]    left_fft_o_result;
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
		.i_sample({adc_right_out[23:8], 16'b0}),
		.o_result(left_fft_o_result),
		.o_sync(left_fft_o_sync));  

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
	sqr = sqrt(left_fft_o_result[20:0],left_fft_o_result[41:21]);
        buffer = rd;
        //buffer = (adc_right_out[23] == 1)  ? 0 : adc_right_out;
    end

    //Determine when the driver is in the middle of pulling a sample
    //by default dont use the BRAM module
    logic           bram_writing = 0;
    logic           bram_reading = 0;
    logic [31:0]    driverReading = 31'd0;
    logic [15:0]    limit;

    always_ff @(posedge clk) begin
	CC += 1;
    end
   
    always_ff @(posedge clk_2) begin
	if(wena) wa += 1;
    end

    always_ff @(posedge clk) begin : IOcalls
        // ioread recieved
        if (chipselect && read) begin
            case (address)
                16'h0002 : begin
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
