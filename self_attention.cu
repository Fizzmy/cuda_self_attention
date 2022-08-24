

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 32
#define EE float(2.71828182845904523536)
void launch_self_attention(float* input,float* wq,float* wk, float* wv,int len,int input_size,int output_size,float* output);
__global__ void MatrixMulKernel(float* M,float* N, float* P, int m ,int n, int k)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    // Loop over the M and N tiles required to compute the P element
    //if (bx==4) printf("%d %d\n",Row,Col);

    for (int p = 0; p < (k+TILE_WIDTH-1)/TILE_WIDTH; ++p)
    {
        // Collaborative loading of M and N tiles into shared memory
        if (p*TILE_WIDTH+tx<k)
            ds_M[ty][tx] = M[Row*k + p*TILE_WIDTH+tx];
        if (p*TILE_WIDTH+ty<k)
            ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*n + Col];
        __syncthreads();
        if (Row<m && Col<n)
        {
            for (int i = 0; i < TILE_WIDTH; ++i)
                if (p*TILE_WIDTH+i<k)
                    Pvalue += ds_M[ty][i] * ds_N[i][tx];
            //printf("%d %d %.3f\n",Row,Col,Pvalue);
        }
        __syncthreads();
    }
    if (Row<m && Col<n)
    {
        P[Row*n+Col] = Pvalue;
        //printf("%d %d %.3f\n",Row,Col,Pvalue);
    }
}
// n * k    k * m
void launch_matrixmul2(float* c,
                       float* a,
                       float* b,
                 int m , int n , int k) {
    dim3 grid((n+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    MatrixMulKernel<<<grid, block>>>(a, b, c, m, n, k);
}

__global__ void MatrixTKernel(float *input,float *output,int m,int n) // m*n
{
    //__shared__ float ds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    if (Row<m && Col<n)
    {
        //printf("%d %d %d %d\n",Row,Col,n*Row+Col,Col*m+Row);
        output[Col*m+Row]=input[n*Row+Col];
    }
}
void launch_matrixT(float *input,float *output,int m,int n) {
    dim3 grid((n+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    MatrixTKernel<<<grid, block>>>(input,output,m,n);
}

__global__ void ReduceMaxKernel(float *input,float *output,int m) // m*n
{
    __shared__ float ds[TILE_WIDTH*TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    //unsigned int i = threadIdx.x + blockIdx.x * m; 
    unsigned int base = blockIdx.x * m;
    for (int j=tid;j<m;j+=blockDim.x)
    {
        if (j < blockDim.x)
            ds[tid] = input[ base + j ];
        else if (input[base+j] > ds[tid])
            ds[tid] = input[ base + j ];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < m){
            if (ds[tid] < ds[tid + s])
                ds[tid] = ds[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = ds[0];
    }
}
__global__ void BroadCastSubExpKernel(float *input,float *reduceMax,float *output,int m)
{
    __shared__ float maxn;
    unsigned int tid = threadIdx.x;
    if (tid==0) maxn=reduceMax[blockIdx.x];
    __syncthreads();
    for (int j=tid;j<m;j+=blockDim.x)
        output[blockIdx.x*m+j] = powf(EE,input[blockIdx.x*m+j]-maxn);
}
__global__ void ReduceSumKernel(float *input,float *output,int m) // m*n
{
    __shared__ float ds[TILE_WIDTH*TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    //unsigned int i = threadIdx.x + blockIdx.x * m; 
    unsigned int base = blockIdx.x * m;
    for (int j=tid;j<m;j+=blockDim.x)
    {
        if (j < blockDim.x)
            ds[tid] = input[ base + j ];
        else ds[tid] += input[ base + j ];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < m){
            ds[tid] += ds[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = ds[0];
    }
}
__global__ void DivKernel(float *input,float *reduceSum,float *output,int m)
{
    __shared__ float sum;
    unsigned int tid = threadIdx.x;
    if (tid==0) sum=reduceSum[blockIdx.x];
    __syncthreads();
    for (int j=tid;j<m;j+=blockDim.x)
        output[blockIdx.x*m+j] = input[blockIdx.x*m+j]/sum;
}
void launch_softmax(float *input,float *output,int m,int n){

    dim3 grid(m);
    dim3 block(TILE_WIDTH*TILE_WIDTH);
    float *reduceMax,*exp,*reduceSum;
    cudaMalloc((void**)&reduceMax, m * sizeof(float));
    cudaMalloc((void**)&exp, m * n * sizeof(float));
    cudaMalloc((void**)&reduceSum, m * sizeof(float));
    ReduceMaxKernel<<<grid,block>>>(input,reduceMax,n);
    cudaThreadSynchronize();
    BroadCastSubExpKernel<<<grid,block>>>(input,reduceMax,exp,n);
    cudaThreadSynchronize();
    cudaFree(reduceMax);
    ReduceSumKernel<<<grid,block>>>(exp,reduceSum,n);
    cudaThreadSynchronize();
    DivKernel<<<grid,block>>>(exp,reduceSum,output,n);
    cudaThreadSynchronize();
    cudaFree(exp);
    cudaFree(reduceSum);
}

void launch_self_attention(float* input,float* wq,float* wk, float* wv,int len,int input_size,int output_size,float* output)
{
    float *Q,*K,*V,*KT,*QK,*softmaxQK;
    cudaMalloc((void**)&Q, len * output_size * sizeof(float));
	cudaMalloc((void**)&K, len * output_size * sizeof(float));
	cudaMalloc((void**)&V, len * output_size * sizeof(float));

    launch_matrixmul2(Q,input,wq,len,output_size,input_size);
    launch_matrixmul2(K,input,wk,len,output_size,input_size);
    launch_matrixmul2(V,input,wv,len,output_size,input_size);
    
    cudaThreadSynchronize();

    cudaMalloc((void**)&KT, len * output_size * sizeof(float));

    launch_matrixT(K,KT,len,output_size);

    cudaThreadSynchronize();

    cudaFree(K);

    cudaMalloc((void**)&QK, len * len * sizeof(float));
    launch_matrixmul2(QK,Q,KT,len,len,output_size);
    cudaThreadSynchronize();
    cudaFree(Q);
    cudaFree(KT);

    cudaMalloc((void**)&softmaxQK, len * len * sizeof(float));
    launch_softmax(QK,softmaxQK,len,len);
    cudaThreadSynchronize();
    cudaFree(QK);

    launch_matrixmul2(output,softmaxQK,V,len,output_size,len);
    cudaThreadSynchronize();
    cudaFree(V);
    cudaFree(softmaxQK);
}