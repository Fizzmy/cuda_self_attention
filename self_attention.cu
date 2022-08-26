

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
    int baseM = blockIdx.x * m * k;
    int baseN = blockIdx.x * k * n;
    int baseP = blockIdx.x * m * n;
    int bx = blockIdx.y; int by = blockIdx.z;
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
            ds_M[ty][tx] = M[baseM + Row*k + p*TILE_WIDTH+tx];
        if (p*TILE_WIDTH+ty<k)
            ds_N[ty][tx] = N[baseN + (p*TILE_WIDTH+ty)*n + Col];
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
        P[baseP + Row*n+Col] = Pvalue;
        //printf("%d %d %.3f\n",Row,Col,Pvalue);
    }
}
// m * k    k * n
void launch_matrixmul2(float* c,
                       float* a,
                       float* b,
                 int batch_size, int m , int n , int k) {
    dim3 grid(batch_size,(n+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    MatrixMulKernel<<<grid, block>>>(a, b, c, m, n, k);
}

__global__ void MatrixTKernel(float *input,float *output,int m,int n) // m*n
{
    //__shared__ float ds[TILE_WIDTH][TILE_WIDTH];
    int base=blockIdx.x * m * n;
    int bx = blockIdx.y; int by = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    if (Row<m && Col<n)
    {
        //printf("%d %d %d %d\n",Row,Col,n*Row+Col,Col*m+Row);
        output[base+ Col * m + Row]=input[base + n * Row + Col];
    }
}
void launch_matrixT(float *input,float *output,int batch_size,int m,int n) {
    dim3 grid(batch_size,(n+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    MatrixTKernel<<<grid, block>>>(input,output,m,n);
}

__global__ void ReduceMaxKernel(float *input,float *output,int m,int n) // m*n
{
    __shared__ float ds[TILE_WIDTH*TILE_WIDTH];

    int tid = threadIdx.x;
    int base = blockIdx.y * n + blockIdx.x * m * n;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    for (int j=tid;j<n;j+=blockDim.x)
    {
        if (j < blockDim.x)
        {
            ds[tid] = input[ base + j ];
            //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
        }
        else if (input[base+j] > ds[tid])
            ds[tid] = input[ base + j ];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < n){
            if (ds[tid] < ds[tid + s])
                ds[tid] = ds[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x * m + blockIdx.y] = ds[0];
    }
}
__global__ void BroadCastSubExpKernel(float *input,float *reduceMax,float *output,int m,int n)
{
    __shared__ float maxn;
    int tid = threadIdx.x;
    int base = blockIdx.y * n + blockIdx.x * m * n;
    if (tid==0) maxn=reduceMax[blockIdx.x * m + blockIdx.y];
    __syncthreads();
    for (int j=tid;j<n;j+=blockDim.x)
        output[base +j] = powf(EE,input[base + j]-maxn);
}
__global__ void ReduceSumKernel(float *input,float *output,int m,int n) // m*n
{
    __shared__ float ds[TILE_WIDTH*TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * n * m + blockIdx.y * n;
    for (int j=tid;j<n;j+=blockDim.x)
    {
        if (j < blockDim.x)
            ds[tid] = input[ base + j ];
        else ds[tid] += input[ base + j ];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < n){
            ds[tid] += ds[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x * m + blockIdx.y] = ds[0];
    }
}
__global__ void DivKernel(float *input,float *reduceSum,float *output,int m,int n)
{
    __shared__ float sum;
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * n * m + blockIdx.y * n;
    if (tid==0) sum=reduceSum[blockIdx.x * m + blockIdx.y];
    __syncthreads();
    for (int j=tid;j<n;j+=blockDim.x)
        output[base+j] = input[base+j]/sum;
}

__global__ void PrintKernel(float *input,int m,int n) // m*n
{
    //__shared__ float ds[TILE_WIDTH][TILE_WIDTH];
    int base=blockIdx.x * m * n;
    int bx = blockIdx.y; int by = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    if (Row<m && Col<n)
        printf("%d %d %d %.3f\n",blockIdx.x,Row,Col,input[base + n * Row + Col]);
}
void print(float *c,int batch_size,int m,int n)
{
    dim3 grid(batch_size,(n+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH);
    PrintKernel<<<grid, block>>>(c,m,n);
}

void launch_softmax(float *input,float *output,int batch_size, int m,int n){

    dim3 grid(batch_size,m);
    dim3 block(TILE_WIDTH*TILE_WIDTH);
    float *reduceMax,*exp,*reduceSum;
    cudaMalloc((void**)&reduceMax,batch_size * m * sizeof(float));
    cudaMalloc((void**)&exp,batch_size * m * n * sizeof(float));
    cudaMalloc((void**)&reduceSum,batch_size * m * sizeof(float));
    ReduceMaxKernel<<<grid,block>>>(input,reduceMax,m,n);
    cudaThreadSynchronize();
    //print(input,batch_size,m,n);
    //print(reduceMax,batch_size,m,1);
    BroadCastSubExpKernel<<<grid,block>>>(input,reduceMax,exp,m,n);
    cudaThreadSynchronize();
    cudaFree(reduceMax);
    ReduceSumKernel<<<grid,block>>>(exp,reduceSum,m,n);
    cudaThreadSynchronize();
    DivKernel<<<grid,block>>>(exp,reduceSum,output,m,n);
    cudaThreadSynchronize();
    cudaFree(exp);
    cudaFree(reduceSum);
}
__global__ void BroadcastKernel(float *input,float *batch_input,int m,int n)
{
    int base=blockIdx.x * m * n;
    int bx = blockIdx.y; int by = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = bx * blockDim.x + tx;
    int Col = by * blockDim.y + ty;
    if (Row<m && Col<n)
        batch_input[base + Row *n + Col] = input[ Row * n + Col];
}
void launch_broadcast(float *input,float *batch_input,float batch_size,int m,int n)
{
    dim3 grid(batch_size,(m+TILE_WIDTH-1)/TILE_WIDTH,(n+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block(TILE_WIDTH,TILE_WIDTH); 
    BroadcastKernel<<<grid,block>>>(input,batch_input,m,n);
}


void launch_self_attention(float* input,float* wq,float* wk, float* wv,int batch_size,int len,int input_size,int output_size,float* output)
{
    float *bq,*bk,*bv,*Q,*K,*V,*KT,*QK,*softmaxQK;
    cudaMalloc((void**)&bq, batch_size * input_size * output_size * sizeof(float));
    cudaMalloc((void**)&bk, batch_size * input_size * output_size * sizeof(float));
    cudaMalloc((void**)&bv, batch_size * input_size * output_size * sizeof(float));
    launch_broadcast(wq,bq,batch_size,input_size,output_size);
    launch_broadcast(wk,bk,batch_size,input_size,output_size);
    launch_broadcast(wv,bv,batch_size,input_size,output_size);

    
    cudaThreadSynchronize();
    

    cudaMalloc((void**)&Q, batch_size * len * output_size * sizeof(float));
	cudaMalloc((void**)&K, batch_size * len * output_size * sizeof(float));
	cudaMalloc((void**)&V, batch_size * len * output_size * sizeof(float));

    launch_matrixmul2(Q,input,bq,batch_size,len,output_size,input_size);
    launch_matrixmul2(K,input,bk,batch_size,len,output_size,input_size);
    launch_matrixmul2(V,input,bv,batch_size,len,output_size,input_size);
    
    cudaThreadSynchronize();
    cudaFree(bq);cudaFree(bk);cudaFree(bv);
    //print(K,batch_size,len,output_size);

    cudaMalloc((void**)&KT,batch_size * len * output_size * sizeof(float));
    launch_matrixT(K,KT,batch_size,len,output_size);
    cudaThreadSynchronize();
    //print(KT,batch_size,output_size,len);

    cudaFree(K);

    cudaMalloc((void**)&QK, batch_size * len * len * sizeof(float));
    launch_matrixmul2(QK,Q,KT,batch_size,len,len,output_size);
    cudaThreadSynchronize();
    cudaFree(Q);
    cudaFree(KT);

    cudaMalloc((void**)&softmaxQK, batch_size * len * len * sizeof(float));
    launch_softmax(QK,softmaxQK,batch_size,len,len);
    cudaThreadSynchronize();
    cudaFree(QK);

    launch_matrixmul2(output,softmaxQK,V,batch_size,len,output_size,len);
    cudaThreadSynchronize();
    cudaFree(V);
    cudaFree(softmaxQK);
}