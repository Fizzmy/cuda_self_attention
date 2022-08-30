

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>
#define TILE_WIDTH 32
#define MAX_THREADS 1024
#define EE float(2.71828182845904523536)
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

__global__ void SoftmaxKernel(float *input,int tgt_len,float *output,float scale) // m*n
{
    __shared__ float ds[MAX_THREADS];

    int tid = threadIdx.x;
    int base = blockIdx.x * tgt_len;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    float outp;
    if (tid < tgt_len)
    {
        outp = input[ base + tid ] * scale;
        ds[tid] = outp;
        //printf("%.3f %.3f\n",scale,input[ base + tid ] * scale);
    }
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < tgt_len){
            if (ds[tid] < ds[tid + s])
                ds[tid] = ds[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }
    
    float maxn = ds[0];
    
    __syncthreads();

    if (tid < tgt_len)
    {
        //printf("%f %f %f\n",maxn,expf(input[ base + tid ] * scale - maxn ),ds[tid]);
        outp = expf(outp-maxn);
        ds[tid] = outp;
    }
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < tgt_len){
            ds[tid] += ds[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }

    maxn = ds[0];
    
    if (tid < tgt_len)
    {
        //printf("%f\n",maxn);
        output[ base + tid ] = outp/maxn;
    }

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

void launch_softmax(float *input,int batch_size,int tgt_len,float *output,float scale){

    dim3 grid(batch_size);
    dim3 block(MAX_THREADS);
    SoftmaxKernel<<<grid,block>>>(input,tgt_len,output,scale);
}

__device__ int dim5calc(int id_0,int id_1,int id_2,int id_3,int id_4,int dim_1,int dim_2,int dim_3,int dim_4)
{
    int nw = id_0;
    nw = nw * dim_1 + id_1;
    nw = nw * dim_2 +id_2;
    nw = nw * dim_3 +id_3;
    nw = nw * dim_4 +id_4;
    return nw;
}

__device__ int dim4calc(int id_0,int id_1,int id_2,int id_3,int dim_1,int dim_2,int dim_3)
{
    int nw = id_0;
    nw = nw * dim_1 + id_1;
    nw = nw * dim_2 +id_2;
    nw = nw * dim_3 +id_3;
    return nw;
}

__device__ int dim3calc(int id_0,int id_1,int id_2,int dim_1,int dim_2)
{
    int nw = id_0;
    nw = nw * dim_1 + id_1;
    nw = nw * dim_2 +id_2;
    return nw;
}

__global__ void Transform20314Kernel(float *input, int dim_3, int dim_4, float *output)
{
    int id_0 = blockIdx.x;
    int id_1 = blockIdx.y;
    int id_2 = blockIdx.z;
    // int id_34 = threadIdx.x;
    int dim_0 = gridDim.x;
    int dim_1 = gridDim.y;
    int dim_2 = gridDim.z;
    int dim_34 = dim_3 * dim_4;
    int srcBase = dim4calc(id_0,id_1,id_2,0,dim_1,dim_2,dim_34);
    int trgBase = dim5calc(id_2,id_0,0,id_1,0,dim_0,dim_3,dim_1,dim_4);

    for (int i=threadIdx.x; i<dim_34; i+=blockDim.x)
    {
        int id_3 = i / dim_4;
        int id_4 = i % dim_4;
        int newBase = dim3calc(id_3, 0, id_4, dim_1, dim_4);
        output[trgBase + newBase] = input[srcBase + i];
        //printf("%d %d %d %d %f\n",trgBase,newBase,srcBase,i, input[srcBase + i ]);
    }

}

void launch_transform_20314(float *input,int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,float *output)
{
    dim3 grid(dim_0,dim_1,dim_2);
    dim3 block(MAX_THREADS);
    Transform20314Kernel<<<grid,block>>>(input,dim_3,dim_4,output);
}

__global__ void Transform021Kernel(float *input, int dim_2 , float *output)
{
    int id_0 = blockIdx.x;
    int id_1 = blockIdx.y;
    // int id_2 = threadIdx.x;
    int dim_0 = gridDim.x;
    int dim_1 = gridDim.y;
    int srcBase = dim3calc(id_0,id_1,0,dim_1,dim_2);
    int trgBase = dim3calc(id_0,0,id_1,dim_2,dim_1);

    for (int i=threadIdx.x; i<dim_2; i+=blockDim.x)
    {
        int newBase = i * dim_1;
        output[trgBase + newBase] = input[srcBase + i];
    }

}

__global__ void Transform0213Kernel(float *input, int dim_3, float *output)
{
    int id_0 = blockIdx.x;
    int id_1 = blockIdx.y;
    int id_2 = blockIdx.z;
    // int id_3 = threadIdx.x;
    int dim_0 = gridDim.x;
    int dim_1 = gridDim.y;
    int dim_2 = gridDim.z;
    int srcBase = dim4calc(id_0,id_1,id_2,0,dim_1,dim_2,dim_3);
    int trgBase = dim4calc(id_0,id_2,id_1,0,dim_2,dim_1,dim_3);

    for (int i=threadIdx.x; i<dim_3; i+=blockDim.x)
    {
        output[trgBase + i] = input[srcBase + i];
        //printf("%d %d %d %d %f\n",trgBase,newBase,srcBase,i, input[srcBase + i ]);
    }

}

void launch_transform_0213(float *input,int dim_0, int dim_1, int dim_2, int dim_3,float *output)
{
    dim3 grid(dim_0,dim_1,dim_2);
    dim3 block(MAX_THREADS);
    Transform0213Kernel<<<grid,block>>>(input,dim_3,output);
}

void launch_transform_021(float *input,int dim_0, int dim_1, int dim_2,float *output)
{
    dim3 grid(dim_0,dim_1);
    dim3 block(MAX_THREADS);
    Transform021Kernel<<<grid,block>>>(input,dim_2,output);
}

void launch_multi_head_attention(float* input,float* qkv,float* o,int batch_size,int tgt_len,int head_num,int hidden_size,float* output)
{
    float *QKV,*QKVT,*softmax_input,*KT,*softmax_output,*softmax_T;
    int size = batch_size * tgt_len * hidden_size;
    int output_size = hidden_size / head_num;
    cudaMalloc((void**)&QKV, 3 * size * sizeof(float));
    cudaMalloc((void**)&QKVT, 3 * size * sizeof(float));
    cudaMalloc((void**)&softmax_input, batch_size * head_num * tgt_len * tgt_len * sizeof(float));
    cudaMalloc((void**)&KT, size * sizeof(float));

    launch_matrixmul2(QKV,input,qkv,1, batch_size * tgt_len, 3 * hidden_size, hidden_size);
    cudaThreadSynchronize();
    
    
    launch_transform_20314(QKV,batch_size,tgt_len,3,head_num,output_size,QKVT);
    cudaThreadSynchronize();


    // QKVT : 3  * batch_size * head_num * tgt_len * output_size
    float *Q = QKVT;
    float *K = QKVT + batch_size * tgt_len * hidden_size;
    float *V = QKVT + 2 * batch_size * tgt_len * hidden_size;

    launch_transform_021(K,batch_size* head_num,tgt_len,output_size,KT);
    cudaThreadSynchronize();

    launch_matrixmul2(softmax_input,Q,KT,batch_size * head_num, tgt_len, tgt_len, output_size);
    cudaThreadSynchronize();
    cudaFree(KT);
    
    if (tgt_len>1024) throw std::runtime_error("Sequence length greater than 1024 is currently not supported");
    cudaMalloc((void**)&softmax_output, batch_size * head_num * tgt_len * tgt_len * sizeof(float));
    launch_softmax(softmax_input ,batch_size * head_num * tgt_len , tgt_len, softmax_output, sqrt((float)1.0/output_size));
    cudaThreadSynchronize();
    cudaFree(softmax_input);

    launch_matrixmul2(output,softmax_output,V,batch_size * head_num, tgt_len, output_size, tgt_len);
    cudaThreadSynchronize();
    cudaFree(QKVT);

    // print(output , batch_size *head_num, tgt_len , output_size);
    // cudaThreadSynchronize();
   
    cudaMalloc((void**)&softmax_T, batch_size * tgt_len * hidden_size * sizeof(float));
    launch_transform_0213(output,batch_size,head_num,tgt_len,output_size,softmax_T);
    cudaThreadSynchronize();

    
    
    launch_matrixmul2(output,softmax_T,o,1,batch_size * tgt_len, hidden_size,  hidden_size);
    cudaThreadSynchronize();
    cudaFree(softmax_T);
}