// Minimal CUDA smoke test: vector add + device props. Not part of benchmarks.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("device=%s sm=%d.%d SMs=%d globalMem=%.1f GiB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0);
    const int n = 1 << 20;
    float *a, *b, *c, *ha = new float[n], *hb = new float[n], *hc = new float[n];
    for (int i = 0; i < n; i++) { ha[i] = i; hb[i] = 1.0f; }
    cudaMalloc(&a, n * 4); cudaMalloc(&b, n * 4); cudaMalloc(&c, n * 4);
    cudaMemcpy(a, ha, n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, n * 4, cudaMemcpyHostToDevice);
    vadd<<<n / 256, 256>>>(a, b, c, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL_FAIL %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hc, c, n * 4, cudaMemcpyDeviceToHost);
    double err = 0; for (int i = 0; i < n; i++) err += fabs(hc[i] - (ha[i] + 1.0f));
    printf("smoke_ok err=%g\n", err);
    return err == 0 ? 0 : 1;
}
