#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void read_file(FILE *mat_file, long double *matrix, int size) {
        int row, col;

        for(row = 0; row < size; row++) {
                for(col = 0; col < (size+1); col++) {
                        fscanf(mat_file, "%Lf", &matrix[row*(size+1) + col]);
                }
        }
}

int check_mat(long double *matrix, int size) {
        int check = size;
        for(int i = 0; i < size; i++) {
                if(matrix[i*(size+2)] == 0) {
                        check = i;
                        break;
                }
        }
        return check;
}

void rearrange_mat(long double *matrix, int size, int err_col) {
        for(int j = 0; j < size; j++) {
                if(matrix[j*(size+1) + err_col] != 0 && matrix[err_col*(size+1) + j] != 0) {
                        for(int i = 0; i < (size+1); i++) {
                                long double temp = matrix[j*(size+1) + i];
                                matrix[j*(size+1) + i] = matrix[err_col*(size+1) + i];
                                matrix[err_col*(size+1) + i] = temp;
                        }
                        break;
                }
        }
}

void gaussian_elim(long double *matrix, int size, int current) {
        for(int row = 0; row < size; row++) {
                if(row == current) {
                        continue;
                }
                else {
                        for(int col = 0; col < (size+1); col++) {
                                if(col == current) {
                                        continue;
                                }
                                else {
                                        long double factor = matrix[row*(size+1) + current] / matrix[current*(size+1) + current];
                                        matrix[row*(size+1) + col] -= factor * matrix[current*(size+1) + col];
                                }
                        }
                        int col = current;
                        long double factor = matrix[row*(size+1) + current] / matrix[current*(size+1) + current];
                        matrix[row*(size+1) + col] -= factor * matrix[current*(size+1) +col];
                }
        }
}

void normalize(long double *matrix, int size, int current) {
        for(int row = 0; row < size; row++) {
                if(row != current) {
                        continue;
                }
                else {
                        for(int col = 0; col < (size+1); col++) {
                                if(col == current) {
                                        continue;
                                }
                                else {
                                        long double factor = 1 / matrix[current*(size+1) + current];
                                        matrix[row*(size+1) + col] *= factor;
                                }
                        }
                        int col = current;
                        long double factor = 1 / matrix[current*(size+1) + current];
                        matrix[row*(size+1) + col] *= factor;
                }
        }
}

int check_sol(long double *matrix, int size) {
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
                printf("Please enter the height of the square matrix: ");
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

        long double *matrix;

        matrix = malloc(size*(size+1)*sizeof(long double));

        read_file(mat_file, matrix, size);

        fclose(mat_file);

	printf("\nInputted Matrix:\n");
        for(int j = 0; j < size; j++) {
                for(int i = 0; i < (size+1); i++) {
                        printf("%Lf  ", matrix[j*(size+1) + i]);
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
        printf("\nafter shuffling: \n");

        for(int j = 0; j < size; j++) {
                for(int i = 0; i < (size+1); i++) {
                        printf("%Lf  ", matrix[j*(size+1) + i]);
                }
                printf("\n");
        }
	*/

        int check = 0;

        clock_t begin = clock();

        for(int n = 0; n < size; n++) {
                gaussian_elim(matrix, size, n);
                check = check_sol(matrix, size);
                if(check == 1 || check ==2) {
                        break;
                }
		normalize(matrix, size, n);
        }

        clock_t end = clock();

        double time = (double)(end - begin) / CLOCKS_PER_SEC;

        if(check == 1) {
                printf("\nInfinitely many solutions!\n");
        }
        else if(check == 2) {
                printf("\nNo solutions!\n");
        }
        else{
                printf("\nSolution Matrix: \n");

                for(int j = 0; j < size; j++) {
                        for(int i = 0; i < (size+1); i++) {
                                printf("%Lf  ", matrix[j*(size+1) + i]);
                        }
                        printf("\n");
                }
        }

        printf("\nThis program took %f seconds the execute the mathematical solutions.\n\n", time);

        free(matrix);

        return 0;
}

