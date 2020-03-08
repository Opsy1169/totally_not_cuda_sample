# Simple program for matrix multiplication using cuda
# System
| Name  | Value |
| ------------- | ------------- |
| CPU | Intel core i5-6200U 2.3GHz with Turbo boost up to 2.8GHz  |
| GPU  | GTX 950M 2GB  |
| RAM | 6GB |
| OS | Windows 10 64 | 

#Time measurments for square matrices. Average for 5 measurments of each size

| Matrix size | Time CPU, msec | Time GPU, msec | CPU/GPU ratio|
|-------------|----------|----------|------------|
|256x256| 252| 11| 22.9|
|512x512 | 1801 | 84 | 21.5|
|1024x1024| 38014 | 660 | 57.6|
|2048x2048| 292045 | 5283| 55.3|
