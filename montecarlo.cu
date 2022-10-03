#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#define Combine(x, y) x##y
#define Com(x, y) Combine(x, y)
#define CudaCheckError() cudaError_t Com(e, __LINE__) = cudaGetLastError(); \
                         if (Com(e, __LINE__) != cudaSuccess) { \
                            fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(Com(e, __LINE__))); \
                         }

#define MIN_G 20.f
#define MAX_G 30.f
#define MIN_H 10.f
#define MAX_H 40.f
#define MIN_D 10.f
#define MAX_D 20.f
#define MIN_V 30.f
#define MAX_V 50.f
#define MIN_THETA 70.f
#define MAX_THETA 80.f
#define TOL 5.f
#define GRAVITY -9.81f

#define NUM_TRIES 16

float RandF(float low, float high) {
	float r = (float) rand();
	float t = r / (float) RAND_MAX;
	return low + t * (high - low);
}

__host__ float Radians(float d) {
    return (M_PI / 180.0f) * d;
}

typedef struct monte_carlo_trial {
	float g;
	float h;
	float d;
	float vx;
	float vy;
} mc_trial_t;

void TimeOfDaySeed() {
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
	srand(seed);
}

void InitializeTrials(mc_trial_t* start, bool* hits, size_t length) {
	TimeOfDaySeed();
	for (size_t i = 0; i < length; i++) {
        mc_trial_t* trial = &start[i];
		trial->g = RandF(MIN_G, MAX_G);
		trial->h = RandF(MIN_H, MAX_H);
		trial->d = RandF(MIN_D, MAX_D);
		float v = RandF(MIN_V, MAX_V);
		float theta = Radians(RandF(MIN_THETA, MAX_THETA));
		trial->vx = (float) cos(theta) * v;
		trial->vy = (float) sin(theta) * v;
		hits[i] = false;
	}
}

__global__ void MonteCarlo(mc_trial_t* trials, bool* hits, size_t length) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    mc_trial_t* trial = &trials[gid];

    float tx = -trial->vy / (0.5f * GRAVITY);
    float x = trial->vx * tx;
    float tg = trial->g / trial->vx;
    float y = trial->vy * tg + (.5f * GRAVITY * tg * tg);
    float td = (-trial->vy - sqrtf((trial->vy * trial->vy) + (2 * GRAVITY * trial->h))) / (GRAVITY);
    float x2 = fabsf((trial->vx * td) - trial->g - trial->d);
    hits[gid] = x > trial->g && y > trial->h && x2 <= TOL;
}

int main(int argc, char* args[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: <blocksize> <num_blocks>");
        return 0;
    }

    int blockSize = atoi(args[1]);
    char* extra;
    size_t length = strtoull(args[2], &extra, 10);

    mc_trial_t* trials = new mc_trial_t[length];
    bool* hits = new bool[length];
    InitializeTrials(trials, hits, length);

    mc_trial_t* deviceTrials = NULL;
    cudaMalloc(&deviceTrials, sizeof(mc_trial_t) * length);
    bool* deviceHits;
    cudaMalloc(&deviceHits, sizeof(bool) * length);
    CudaCheckError();

    cudaMemcpy(deviceTrials, trials, sizeof(mc_trial_t) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHits, hits, sizeof(bool) * length, cudaMemcpyHostToDevice);
    CudaCheckError();

    dim3 grid( length / blockSize, 1, 1 );
    dim3 threads( blockSize, 1, 1 );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    CudaCheckError();

    cudaDeviceSynchronize();
    cudaEventRecord(start, NULL);
    CudaCheckError();
    MonteCarlo<<< grid, threads >>>(deviceTrials, deviceHits, length);
    cudaEventRecord(stop, NULL);
    CudaCheckError();

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    CudaCheckError();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    CudaCheckError();

    // Compute performance.
    double megaTrials = (double) length / ((double) msecTotal / 1000.) / 1000000.;

    cudaMemcpy(hits, deviceHits, sizeof(bool) * length, cudaMemcpyDeviceToHost);
    CudaCheckError();

    // Calculate the percentage.
    double hitCount = 0;
    for (int i = 0; i < length; i++) {
        hitCount += (double) ((int) hits[i]);
    }

    fprintf(stderr, "%d, %lu, %.4lf, %.4lf\n", blockSize, length, megaTrials, (hitCount / (double) length) * 100.);

    delete[] trials;
    delete[] hits;

    cudaFree(deviceTrials);
    cudaFree(deviceHits);
    CudaCheckError();

    return 0;
}
