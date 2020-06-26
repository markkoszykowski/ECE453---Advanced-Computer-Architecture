#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

//Filter used to sharpen the image (Gaussian Unblurring)
__device__
__constant__
int deblur_mask[5][5] {
	{-1, -4, -6, -4, -1},
	{-4, -16, -24, -16, -4},
	{-6, -24, 476, -24, -6},
	{-4, -16, -24, -16, -4},
	{-1, -4, -6, -4, -1}
};

//CUDA Kernel Function
__global__
void deblur(char *blurred_image, char *unblurred_image, int height, int width) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	
	int red = 0;
	int green = 0;
	int blue = 0;

	if(i >= 2 && i < width - 2 && j >= 2 && j < height - 2) {
		for(int a = -2; a < 3; a++) {
			for(int b = -2; b < 3; b++) {
				red += deblur_mask[2+a][2+b]*blurred_image[((j+a)*(width)+(i+b))*3];
				green += deblur_mask[2+a][2+b]*blurred_image[((j+a)*(width)+(i+b))*3 + 1];
				blue += deblur_mask[2+a][2+b]*blurred_image[((j+a)*(width)+(i+b))*3 + 2];
			}
		}
		red = red / 256;
		green = green / 256;
		blue = blue / 256;
		if(red < 0) {
			red = 0;
		}
		if(green < 0) {
			green = 0;
		}
		if(blue < 0) {
			blue = 0;
		}
		unblurred_image[(j*width+i)*3] = red;
		unblurred_image[(j*width+i)*3+1] = green;
		unblurred_image[(j*width+i)*3+2] = blue;
	}
}

int main(int argc, char *argv[]) {
	int height, width, row_width, size;

	//Declare host arrays
	char *h_blurred_image = NULL;
	char *h_unblurred_image = NULL;

	cudaError_t err = cudaSuccess;

	if(argc != 3) {
		printf("Usage: ./Project1 BlurredFile.png DesiredOutput.png\n");
		return 1;
	}
	
	//Start opening the image and convert it workable format
	FILE *png = fopen(argv[1], "rb");
	if(!png) {
		printf("%s could not be opened. \n", argv[1]);
		return -1;
	}

	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png_ptr) {
		printf("Coud not make read structure for %s. \n", argv[1]);
		return -2;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if(!info_ptr) {
		printf("Could not make info structure for %s. \n", argv[1]);
		return -3;
	}

	png_init_io(png_ptr, png);
	png_read_info(png_ptr, info_ptr);

	//Remove alpha value(transparency) for each pixel
	png_set_strip_alpha(png_ptr);
	
	//Save image parameters	
	height = png_get_image_height(png_ptr, info_ptr);
	width = png_get_image_width(png_ptr, info_ptr);
	row_width = 3 * width;
	size = height * row_width;

	printf("Image width: %d \nImage height: %d \n", width, height);

	//Allocate space on host for arrays
	h_blurred_image = (char*)calloc(row_width, height);
	h_unblurred_image = (char*)calloc(row_width, height);
	
	//Read data from blurred PNG
	for(int i = 0; i < height; i++) {
		png_read_row(png_ptr, (png_byte*)h_blurred_image + i*row_width, NULL);
	}

	fclose(png);

	//Set the unblurred image data equal to the blurred image data
	for(int i = 0; i < size; i++) {
		h_unblurred_image[i] = h_blurred_image[i];
	}

	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	
	//Declare device arrays
	char *d_blurred_image = NULL;
	char *d_unblurred_image = NULL;

	//Allocate space on device for arrays
	err = cudaMalloc((void**)&d_blurred_image, size);
	if(err != cudaSuccess) {
		printf("Failed to allocate space on device for blurred image array.\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)&d_unblurred_image, size);
	if(err != cudaSuccess) {
		printf("Failed to allocate space on device for unblurred image array.\n");
		exit(EXIT_FAILURE);
	}

	//Copy Memory from host to device
	err = cudaMemcpy(d_blurred_image, h_blurred_image, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Failed to copy blurred image array to device.\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_unblurred_image, h_unblurred_image, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Failed to copy unblurred image array to device.\n");
		exit(EXIT_FAILURE);
	}

	//Set Block and Grid dimenstions
	dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
	dim3 dimBlock(16, 16, 1);
	
	clock_t begin = clock();

	//Launch Kernel
	deblur<<<dimGrid, dimBlock>>>(d_blurred_image, d_unblurred_image, height, width);

	clock_t end = clock();

	//Determine length of execution
	double time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("The deblurring algorithm took: %f seconds when run on the GPU.\n", time);

	err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Copy Memory from device to host
	err = cudaMemcpy(h_unblurred_image, d_unblurred_image, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Failed to copy unblurred image array to host.\n");
		exit(EXIT_FAILURE);
	}

	//Free space on device
	err = cudaFree(d_blurred_image);
	if(err != cudaSuccess) {
		printf("Failed to free device memory for blurred image array.\n");
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_unblurred_image);
	if(err != cudaSuccess) {
		printf("Failred to free device memory for unblurred image array.\n");
		exit(EXIT_FAILURE);
	}

	//Copy the contents of the 1D array to a 2D array
	png_byte **row_pointers = NULL;
	row_pointers = (png_byte**)malloc(size);
	for(int i = 0; i < height; i++) {
		row_pointers[i] = (png_byte*)malloc(row_width);
		row_pointers[i] = (png_byte*)h_unblurred_image+i*row_width;
	}

	free(h_blurred_image);

	//Start creating the new image
	FILE *output = fopen(argv[2], "wb");
	if(!output) {
		printf("%s could not be made.\n", argv[2]);
		return -4;
	}

	png_structp png_ptr_new = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png_ptr_new) {
		printf("Could not make write structure for %s. \n", argv[2]);
		return -5;
	}

	png_infop info_ptr_new = png_create_info_struct(png_ptr_new);
	if(!info_ptr_new) {
		printf("Could not make info structure for %s. \n", argv[2]);
		return -6;
	}

	png_init_io(png_ptr_new, output);

	//Set parameters for the unblurred output
	png_set_IHDR(
			png_ptr_new,
			info_ptr_new,
			width, height, 8,
			PNG_COLOR_TYPE_RGB,
			PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_DEFAULT,
			PNG_FILTER_TYPE_DEFAULT
	);

	//Write data to image
	png_write_info(png_ptr_new, info_ptr_new);
	png_write_image(png_ptr_new, row_pointers);
	png_write_end(png_ptr_new, NULL);
	fclose(output);

	//Free space on host
	free(h_unblurred_image);
	free(row_pointers);

	return 0;

}
