This project will contain 3 files of code:

-project2.c
-project2.cu
-Makefile

project1.c will contain the C code that is used to solve the system of linear equations on the CPU.

project2.cu will contain the CUDA C code that is used to solve the system of linear equations on the GPU.

The two programs use the same algorithm and output the total execution time that was required to just solve the system.

To compile project2.c, simply run "gcc -o project2 project2.c".

To execute project2.c, simple run "./project2" and follow the on screen instructions.

To compile project2.cu, simply run "make".

to execult project2.cu, simply run "./Project2" and follow the on screen instructions.

Please ensure that when entering the matrix height, the proper height is entered as this code does not perform error checking.

Also, please ensure that the .txt file containing the system coefficients is in the same path as the executable files.

The text file containing the system coefficients should me in the following format:

a00 a01 a02 ... a0n+1
a10 a11 a12 ... a1n+1
.
.
.
an0 an1 an2 ... an(n+1)

NOTE:

The CUDA C program will only work with matrices up to heights of 31 (31x32) because a thread block can only contain up to 1024 threads. Furthermore, grid synchronization is not available on the Jetson's architecture.
