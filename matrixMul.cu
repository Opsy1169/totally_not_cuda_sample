#include <stdio.h>
#include <assert.h>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
	float *B, int wA, int wB) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ind_x = bx * BLOCK_SIZE + tx;
	int ind_y = by * BLOCK_SIZE + ty;
	int ind_c = ind_y * wA + ind_x;
	C[ind_c] = 0;
	for (int i = 0; i < wA; ++i) {
		C[ind_c] += A[ind_y*wA + i] * B[ind_x + i * wA];
	}
}

void ConstantInit(float *data, int size, float val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}


float* MatrixMultiply(const int block_size, const dim3 &dimsA,
	const dim3 &dimsB, float *hA, float *hB) {

	// Calculate allocated host memory for matrices A and B to allocate same amount of memory on device
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;


	// Allocate host matrix C
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *)(malloc(mem_size_C));

	if (h_C == NULL) {
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	checkCudaErrors(cudaMalloc((void **)(&d_A), mem_size_A));
	checkCudaErrors(cudaMalloc((void **)(&d_B), mem_size_B));
	checkCudaErrors(cudaMalloc((void **)(&d_C), mem_size_C));

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_A, hA, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, hB, mem_size_B, cudaMemcpyHostToDevice));

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);


	// Record the start event
	checkCudaErrors(cudaEventRecord(start));


	// Perform computations

	MatrixMulCUDA<32> << <grid, threads >> > (d_C, d_A, d_B,
		dimsA.x, dimsB.x);

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	printf("GPU multiplication time for block %u: %.2f \n", block_size, msecTotal);

	// Copy result back to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));


	// Free gpu resources
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	return h_C;
}

float* cpu_mat_mul(float *A, float *B, const dim3 &dimsA,
	const dim3 &dimsB) {
	float *cpu_C = (float *)(malloc(dimsB.x*dimsA.y * sizeof(float)));
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < dimsA.y; ++i) {
		for (int j = 0; j < dimsB.x; ++j) {
			cpu_C[i*dimsA.x + j] = 0;
			for (int k = 0; k < dimsB.y; ++k) {
				cpu_C[i*dimsA.x + j] += A[i*dimsA.x + k] * B[j + k * dimsB.x];
			}
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	printf("CPU multiplication time %u msec", elapsed);
	return cpu_C;
}


void check_correctness(float *cpu, float *gpu, dim3 dimC) {
	double eps = 1.0e-6;
	for (int i = 0; i < dimC.x*dimC.y; ++i) {
		if (abs(gpu[i] - cpu[i]) > eps) {
			printf("Big error for %u", i);
		}
	}
}


int main(int argc, char **argv) {

	int width = 256;
	const int block_size = 32;

	dim3 dimsA(width, width, 1);
	dim3 dimsB(width, width, 1);

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
		dimsB.x, dimsB.y);
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)(malloc(mem_size_A));
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)(malloc(mem_size_B));
	const float valB = 0.01f;
	ConstantInit(h_A, size_A, 1.0f);
	ConstantInit(h_B, size_B, valB);

	float* cuda_result = MatrixMultiply(block_size, dimsA, dimsB, h_A, h_B);
	float* cpu_result = cpu_mat_mul(h_A, h_B, dimsA, dimsB);
	check_correctness(cuda_result, cpu_result, dimsA);
	free(h_A);
	free(h_B);
	free(cuda_result);
	free(cpu_result);

	exit(0);
}

