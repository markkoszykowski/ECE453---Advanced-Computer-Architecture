#include <png.h>
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void compare(char *blurred_image, char *unblurred_image, char *original_image, int size) {
	double blurred_diff = 0;
	double unblurred_diff = 0;
	double blurred_total = 0;
	double unblurred_total = 0;
	for(int i = 0; i < size; i++) {
		blurred_diff += fabs(blurred_image[i] - original_image[i])/255;
		unblurred_diff += fabs(unblurred_image[i] - original_image[i])/255;
	}
	blurred_total = (blurred_diff/size) * 100;
	unblurred_total = (unblurred_diff/size) * 100;
	printf("Percent difference between blurred and original: %f\n", blurred_total);
	printf("Percent difference between unblurred and original: %f\n", unblurred_total);
}

int main(int argc, char *argv[]) {
	int height, width, row_width, size;
	
	if(argc != 4) {
		printf("Usage: ./compare BlurredImage.png UnblurredImage.png Original.jpg\n");
		return 1;
	}

	char *blurred_image = NULL;
	char *unblurred_image = NULL;
	char *original_image = NULL;

	//Read the blulrred iamge and convert it to a workable format
	FILE *blurred = fopen(argv[1], "rb");
	if(!blurred) {
		printf("%s coud not be opened.\n", argv[1]);
		return -1;
	}

	png_structp png_ptr_1 = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png_ptr_1) {
		printf("Could not make read structure for %s.\n", argv[1]);
		return -2;
	}

	png_infop info_ptr_1 = png_create_info_struct(png_ptr_1);
	if(!info_ptr_1) {
		printf("Could not make info structure for %s.\n", argv[1]);
		return -3;
	}

	png_init_io(png_ptr_1, blurred);
	png_read_info(png_ptr_1, info_ptr_1);

	png_set_strip_alpha(png_ptr_1);

	height = png_get_image_height(png_ptr_1, info_ptr_1);
	width = png_get_image_width(png_ptr_1, info_ptr_1);
	row_width = 3 * width;
	size = row_width * height;

	blurred_image = (char*)calloc(row_width, height);

	for (int i = 0; i < height; i++) {
		png_read_row(png_ptr_1, (png_byte*)blurred_image+i*row_width, NULL);
	}

	fclose(blurred);

	png_destroy_read_struct(&png_ptr_1, &info_ptr_1, NULL);

	//Read the unblurred image and convert it to workable format
	FILE *unblurred = fopen (argv[2], "rb");
	if(!unblurred) {
		printf("%s could not be opened.\n", argv[2]);
		return -4;
	}

	png_structp png_ptr_2 = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png_ptr_2) {
		printf("Could not make read structure for %s.\n", argv[2]);
		return -5;
	}

	png_infop info_ptr_2 = png_create_info_struct(png_ptr_2);
	if(!info_ptr_2) {
		printf("Could not mafe info structure for %s.\n", argv[2]);
		return -6;
	}

	png_init_io(png_ptr_2, unblurred);
	png_read_info(png_ptr_2, info_ptr_2);

	png_set_strip_alpha(png_ptr_2);

	unblurred_image = (char*)calloc(row_width, height);

	for(int i = 0; i < height; i++) {
		png_read_row(png_ptr_2, (png_byte*)unblurred_image+i*row_width, NULL);
	}

	fclose(unblurred);

	png_destroy_read_struct(&png_ptr_2, &info_ptr_2, NULL);

	//Read the original image and convert it to a workable format
	FILE *original	= fopen(argv[3], "rb");
	if(!original) {
		printf("%s could not be opened.\n", argv[3]);
		return -7;
	}

	struct jpeg_decompress_struct info;
	struct jpeg_error_mgr err;

	info.err = jpeg_std_error(&err);
	jpeg_create_decompress(&info);

	jpeg_stdio_src(&info, original);
	jpeg_read_header(&info, TRUE);
	jpeg_start_decompress(&info);

	if(info.num_components != 3) {
		printf("Original JPEG is not in RGB channel configuration.\n");
		return -8;
	}

	original_image = (char*)malloc(size);

	while(info.output_scanline < info.output_height) {
		unsigned char *image_array[1];
		image_array[0] = (unsigned char*)original_image + info.output_scanline*row_width;
		jpeg_read_scanlines(&info, image_array, 1);
	}

	jpeg_finish_decompress(&info);
	jpeg_destroy_decompress(&info);

	fclose(original);
        
        //Call to compare function
	compare(blurred_image, unblurred_image, original_image, size);

        //Free memory
	free(blurred_image);
	free(unblurred_image);
	free(original_image);

	return 0;
}
