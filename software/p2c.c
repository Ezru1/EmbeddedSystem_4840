#include <stdio.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480

int main() {
    FILE *fp_in, *fp_out;
    unsigned char *image_data;
    int i;

    // Open the input image file in binary mode
    fp_in = fopen("sword.png", "rb");
    if (fp_in == NULL) {
        printf("Error opening input image file\n");
        return 1;
    }

    // Allocate memory for the image data array
    image_data = (unsigned char *)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    if (image_data == NULL) {
        printf("Error allocating memory for image data\n");
        fclose(fp_in);
        return 1;
    }

    // Read pixel values from the input image file
    fread(image_data, sizeof(unsigned char), WIDTH * HEIGHT, fp_in);

    // Close the input image file
    fclose(fp_in);

    // Open the output text file to write the array
    fp_out = fopen("sword.coe", "w");
    if (fp_out == NULL) {
        printf("Error opening output file\n");
        free(image_data);
        return 1;
    }

    // Write the pixel values to the output text file
    for (i = 0; i < WIDTH * HEIGHT; i++) {
        fprintf(fp_out, "%d ", image_data[i]);
    }

    // Close the output text file
    fclose(fp_out);

    // Free the allocated memory
    free(image_data);

    printf("Array written to image_data.txt\n");

    return 0;
}

