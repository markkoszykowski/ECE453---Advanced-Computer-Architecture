#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
//#include <cooperative_groups.h>

void read_file(FILE *mat_file, double *matrix, int size) {
	int row, col;

	for(row = 0; row < size; row++) {
		for(col = 0; col < (size+1); col++) {
			fscanf(mat_file, "%lf", &matrix[row*(size+1) + col]);
		}
	}
}

int check_mat(double *matrix, int size) {
	int check = size;

	for(int i = 0; i < size; i++) {
		if(matrix[i*(size+2)] ==0) {
			check = i;
			break;
		}
	}

	return check;
}

void rearrange_mat(double *matrix, int size, int err_col) {
	for(int j = 0; j < size; j++) {
		if(matrix[j*(size+1) + err_col] != 0 && matrix[err_col*(size+1) + j] != 0) {
			for(int i = 0; i < (size+1); i++) {
				double temp = matrix[j*(size+1) +  i];
				matrix[j*(size+1) + i] = matrix[err_col*(size+1) + i];
				matrix[err_col*(size+1) + i] = temp;
			}
			break;
		}
	}
}

__global__
void gaussian_elim(double *d_matrix, int size, int current) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//grid_group grid = this_grid();

	if(row < size && row != current && col < (size+1) && col != current) {
		double factor = d_matrix[row*(size+1) + current] / d_matrix[current*(size+1) + current];
		d_matrix[row*(size+1) + col] -= factor * d_matrix[current*(size+1) + col];
	}

	__syncthreads();
	//grid.sync();

	if(row < size && row != current && col == current) {
		double factor = d_matrix[row*(size+1) + current] / d_matrix[current*(size+1) + current];
		d_matrix[row*(size+1) + col] -= factor * d_matrix[current*(size+1) + col];
	}
}

__global__
void normalize(double *d_matrix, int size, int current) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//grid_group grid = this_grid();

	if(row == current && col < (size+1) && col != current) {
		double factor = 1 / d_matrix[current*(size+1) + current];
		d_matrix[row*(size+1) + col] *= factor;
	}

	__syncthreads();
	//grid.sync();

	if(row == current && col == current) {
		double factor = 1 / d_matrix[current*(size+1) + current];
		d_matrix[row*(size+1) + col] *= factor;
	}
}

int check_sol(double *matrix, int size) {
	int total_zero = 0;
	int check = 0;

	for(int row = 0; row < size; row++) {
		total_zero = 0;
		for(int col = 0; col < (size+1); col++) {
			if(matrix[row*(size+1) + col] == 0) {
				total_zero += 1;
			}
		}
		if(total_zero == (size+1)) {
			check = 1;
			break;
		}
		else if(total_zero == size && matrix[row*(size+1) + size] != 0) {
			check = 2;
			break;
		}
	}

	return check;
}

int main() {
	char mat_add[50];
	int size = 0;

	do {
		printf("Please enter the height of the matrix: ");
		scanf("%d", &size);
		while(getchar() != '\n');
	} while(size == 0);

	FILE *mat_file;

	do {
		printf("Please enter the name of the file containing the system coefficients: ");
		scanf("%s", mat_add);
		while(getchar() != '\n');
		mat_file = fopen(mat_add, "r");
	} while(mat_file == NULL);

	double *matrix;
	int total_size = size*(size+1)*sizeof(double);

	matrix = (double*)malloc(total_size);

	read_file(mat_file, matrix, size);

	fclose(mat_file);

	printf("\nInputted Matrix: \n");
	for(int j = 0; j < size; j++) {
		for(int i = 0; i < (size+1); i++) {
			printf("%lf  ", matrix[j*(size+1) + i]);
		}
		printf("\n");
	}

	int err_col = check_mat(matrix, size);

	int zeros = 0;

	while(err_col != size) {
		rearrange_mat(matrix, size, err_col);
		err_col = check_mat(matrix, size);
		zeros++;
		if(zeros >= size) {
			printf("\nOverdefined Matrix!\n");
			return 0;
		}
	}

	/*
	printf("\nAfter Shuffling: \n");

	for(int j = 0; j < size; j++) {
		for(int i = 0; i < (size+1); i++) {
			printf("%Lf  ", matrix[j*(size+1) + i]);
		}
		printf("\n");
	}
	*/

	double *d_matrix;
	
	cudaError_t err = cudaSuccess;

	err = cudaMalloc((void**)&d_matrix, total_size);
	if(err != cudaSuccess) {
		printf("Failed to allocate space on device for matrix.\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_matrix, matrix, total_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Failed to copy matrix from host to device.\n");
		exit(EXIT_FAILURE);
	}

	int thread_x = 0;
        int thread_y = 0;
	int block_x = 0;
	int block_y = 0;

	if(size < 32) {
		thread_x = size+1;
		thread_y = size;

		block_x = 1;
		block_y = 1;
	}
	else {
		thread_x = 16;
		thread_y = 16;

		block_x = ceil((size+1)/16);
		block_y = ceil(size/16);
	}



	dim3 threadsPerBlock(thread_x, thread_y, 1);
	dim3 numBlocks(block_x, block_y, 1);

	int check = 0;

	clock_t begin = clock();

	for(int n = 0; n < size; n++) {
		gaussian_elim<<<numBlocks, threadsPerBlock>>>(d_matrix, size, n);
	
		err = cudaDeviceSynchronize();
		if(err != cudaSuccess) {
			printf("Failed to synchronize device.");
			exit(EXIT_FAILURE);
		}
		
		err = cudaMemcpy(matrix, d_matrix, total_size, cudaMemcpyDeviceToHost);
		if(err != cudaSuccess) {
			printf("Failed to copy matrix from device to host.\n");
			exit(EXIT_FAILURE);
		}

		check = check_sol(matrix, size);
		if(check == 1 || check == 2) {
			break;
		}

		normalize<<<numBlocks, threadsPerBlock>>>(d_matrix, size, n);
		
		err = cudaDeviceSynchronize();
		if(err != cudaSuccess) {
			printf("Failed to synchronize device.");
			exit(EXIT_FAILURE);
		}
		
		err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}

	clock_t end = clock();

	double time = (double)(end - begin) / CLOCKS_PER_SEC;

	err = cudaMemcpy(matrix, d_matrix, total_size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Failed to copy matrix from device to host.\n");
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_matrix);
	if(err != cudaSuccess) {
		printf("Failed to free matrix from device memory.\n");
		exit(EXIT_FAILURE);
	}

	if(check == 1) {
		printf("\nInfinitely many solutions!\n");
	}
	else if(check == 2) {
		printf("\nNo solutions!\n");
	}
	else {
		printf("\nSolution Matrix: \n");
		
		for(int j = 0; j < size; j++) {
			for(int i = 0; i < (size+1); i++) {
				printf("%lf  ", matrix[j*(size+1) + i]);
			}
			printf("\n");
		}
		
	}

	printf("This program took %f seconds to execute the mathematical solutions.\n", time);

	free(matrix);

	return 0;

}
