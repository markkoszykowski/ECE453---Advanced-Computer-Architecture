This Project will contain 4 files of code:

-p1.c
-p1.cu
-Makefile
-comparison.c

p1.c will contain the C code that is used to deblur an image on the CPU.

p1.cu will contain the CUDA C code that is used to deblur an image on the GPU.

The two programs use the same algorithm and output the total execution time that was required to just deblur the image.

To compile p1.c, simply run "gcc -o project1 p1.c -lpng"

To execute p1.c, simply run "./project1 BlurredImage.png DesiredOutput.png" after compilation.

To compile p1.cu, simply run "make TARGET_ARCH=aarch64 TARGET_OS=linux SMS="30 32 53 61 62 70 72 75""

To execute p1.cu, simply run "./Project1 BlurredImage.png DesiredOutput.png" after compilation.

Please ensure that the BlurredImage.png is in the same path as the executable file. The DesiredOutput.png will be save to the current directory of the executable.

This code does not perform image format checking so please input images properly.

The comparison.c code is used to compare the blurred and unblurred photos to the original reference photo.

To compile comparison.c, simply run "gcc -o compare comparison.c -lpng -ljpeg"

To execute comparison.c, simply run "./compare BlurredImage.png UnblurredImage.png Original.jpg" after compilation.

Please ensure that the BlurredImage.png, UnblurredImage.png, and Original.jpg files are in the same path as the executable file.

This code does not perform image format checking so please input images properly.
