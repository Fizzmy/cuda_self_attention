#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>
#include <stddef.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "multi_head_attention.h"
#define TILE_WIDTH 32
#define MAX_THREADS 1024
#define EPS 1e-8f

void launch_matrixmul(cublasHandle_t handle,float* c,
                       float* a,
                       float* b,
                 int batch_size, int m , int n , int k, bool trans_A, bool trans_B, int scaler)
{
    //printf("%.3f %.3f\n",a[0],b[0]);
    if (scaler==0)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (!trans_A && !trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, n*k, a, k, m*k, &beta, c, n, m*n, batch_size);
        else if (!trans_A && trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, n*k, a, k, m*k, &beta, c, n, m*n, batch_size);
        else if (trans_A && !trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, n, n*k, a, m, m*k, &beta, c, n, m*n, batch_size);
        else if (trans_A && trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b, k, n*k, a, m, m*k, &beta, c, n, m*n, batch_size);
    }
    else
    {
        const float alpha = sqrt((float)1.0/scaler);
        const float beta  = 0.0f;
        if (!trans_A && !trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, n*k, a, k, m*k, &beta, c, n, m*n, batch_size);
        else if (!trans_A && trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, n*k, a, k, m*k, &beta, c, n, m*n, batch_size);
        else if (trans_A && !trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, n, n*k, a, m, m*k, &beta, c, n, m*n, batch_size);
        else if (trans_A && trans_B)
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b, k, n*k, a, m, m*k, &beta, c, n, m*n, batch_size);
    }
}
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
                /*
                 float y,t;
                    y = ds_M[ty][i] * ds_N[i][tx] - c;
                    t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                */
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

__global__ void PrintKernel(float *input,int m,int n,int kk) // m*n
{
    //__shared__ float ds[TILE_WIDTH][TILE_WIDTH];
    for (int i=0;i<m;i++)
    {
        printf("[");
        for (int j=0;j<n;j++)
        {
            printf("[");
            for (int k=0;k<kk;k++)
                printf("%.5f ",input[i * n * kk + j * kk + k]);
            printf("]\n");
        }
        printf("]\n");
    }
}
void print(float *c,int batch_size,int m,int n)
{
    dim3 grid(1);
    dim3 block(1);
    PrintKernel<<<grid, block>>>(c,batch_size,m,n);
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

__global__ void Transform13024Kernel(float *input, int dim_3, int dim_4, float *output)
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
    int trgBase = dim5calc(id_1,0,id_0,id_2,0,dim_3,dim_0,dim_2,dim_4);

    for (int i=threadIdx.x; i<dim_34; i+=blockDim.x)
    {
        int id_3 = i / dim_4;
        int id_4 = i % dim_4;
        int newBase = dim4calc(id_3, 0, 0, id_4, dim_0, dim_2, dim_4);
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

void launch_transform_13024(float *input,int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,float *output)
{
    dim3 grid(dim_0,dim_1,dim_2);
    dim3 block(MAX_THREADS);
    Transform13024Kernel<<<grid,block>>>(input,dim_3,dim_4,output);
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

const unsigned int WARP_REDUCE_MASK = 0xffffffff;

__global__ void SoftmaxKernel(float *input,bool *mask,int tgt_len,float* output,bool is_pre) // m*n
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int len = blockIdx.z;
    int tid = threadIdx.x;
    int base = dim4calc(batch, head, len, 0, gridDim.y, gridDim.z , tgt_len);
    int mask_base = batch * tgt_len;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    float val;
    float outp;
    //for (int j = 0; j < tgt_len; j++,base+=tgt_len)
    {
        outp = -1e9;
        if (tid < tgt_len)
        {
            if (mask [mask_base + tid] == 0)
            {
                if (!is_pre || tid <= len) 
                    outp = input[base + tid];
            }    
            //ds[tid] = outp;
            //printf("%.3f %.3f\n",scale,input[ base + tid ] * scale);
        }
        val = outp;
            //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
        __syncthreads();

        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32));
        __syncthreads();
        __shared__ float reduce[TILE_WIDTH];
        if ((tid & 0x1f) == 0)
        {
            reduce[tid>>5] = outp;
        }
        __syncthreads();
        if (tid < (blockDim.x>>5))
            outp = reduce[tid];
        else
            outp = -1e9;
        __syncthreads();

        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32));
        outp = max(outp, __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32));
        __syncthreads();
        
        __shared__ float maxn;
        if (tid==0)
            maxn = outp;
        
        __syncthreads();

        if (tid < tgt_len)
        {
            //printf("%f %f %f\n",maxn,expf(input[ base + tid ] * scale - maxn ),ds[tid]);
            val = __expf(val-maxn);
        }
        else val = 0.0f;
        outp = val;
            //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
        __syncthreads();

        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
        __syncthreads();
        if ((tid & 0x1f) == 0)
        {
            reduce[tid>>5] = outp;
        }
        __syncthreads();
        if (tid < (blockDim.x>>5))
            outp = reduce[tid];
        else
            outp = 0.0f;
        __syncthreads();

        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
        __syncthreads();

        
        if (tid==0)
        {
            maxn = fdividef ( 1.0f , outp);
        }
        __syncthreads();


        if (tid < tgt_len)
        {
            //printf("%f\n",maxn);
            output[ base + tid ] = val * maxn;
        }
    }

}

void launch_softmax(float *input,bool* mask,int batch_size,int head_num,int tgt_len,float *output,bool is_pre)
{
    dim3 grid(batch_size,head_num,tgt_len);
    dim3 block(MAX_THREADS);
    SoftmaxKernel<<<grid,block>>>(input,mask,tgt_len,output,is_pre);
}

__global__ void SoftmaxBwKernel(float *input,float *input_grad,int tgt_len,float *output,float scale)
{

    int tid = threadIdx.x;
    int base = blockIdx.x * tgt_len;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    float outp=0.0f;
    if (tid < tgt_len)
        outp = input[base + tid] * input_grad[base + tid];
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    __shared__ float reduce[TILE_WIDTH];
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
    __syncthreads();
    if ((tid & 0x1f) == 0)
    {
        reduce[tid>>5] = outp;
    }
    __syncthreads();
    if (tid < (blockDim.x>>5))
        outp = reduce[tid];
    else
        outp = 0.0f;
    __syncthreads();

    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
    outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
    __syncthreads();

    __shared__ float sum;
    if (tid==0)
        sum = outp;
    __syncthreads();

    if (tid<tgt_len)
        output[base + tid] = scale * input[base + tid] * (input_grad[base + tid] - sum);

}

void launch_softmax_bw(float *input,float *input_grad,int batch_size,int tgt_len,float *output,float scale)
{
    dim3 grid(batch_size);
    dim3 block(MAX_THREADS);
    SoftmaxBwKernel<<<grid,block>>>(input,input_grad,tgt_len,output,scale);
}
__global__ void LayernormKernel(float *input,int batch_size,int hidden_size,float *input_hat,float *output,float *input_mean,float *input_std,float *normw,float *normb)
{
    __shared__ float mean[MAX_THREADS],mean_sqare[MAX_THREADS];
    int tid = threadIdx.x;
    int base = blockIdx.x * hidden_size;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    mean[tid]=mean_sqare[tid]=0;
    for (int i = tid; i < hidden_size; i += blockDim.x)
    {
        mean[tid] += input[ base + i ];
        mean_sqare[tid] += input[ base + i ] * input[ base + i ];
        //printf("%.3f %.3f\n",scale,input[ base + tid ] * scale);
    }
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < hidden_size && tid + s < blockDim.x)
        {
            mean[tid] += mean[tid + s];
            mean_sqare[tid] += mean_sqare[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }

    __shared__ float tot_mean, tot_std;
    if (tid == 0)
    {
        tot_mean = mean[0] / hidden_size;
        input_mean[ blockIdx.x ] = tot_mean;
        tot_std = mean_sqare[0] / hidden_size - tot_mean * tot_mean + EPS;
        //printf("%f\n",tot_std);
        tot_std = rsqrtf(tot_std);
        input_std[ blockIdx.x ] = tot_std;
        //printf("%f %f\n",tot_mean,tot_std);
    }
    __syncthreads();

    float hat;
    for (int i = tid; i < hidden_size; i += blockDim.x)
    {
        hat = ( input[ base + i ] - tot_mean ) * tot_std;
        output[ base + i ] = hat * normw[i] + normb[i];
        input_hat[ base + i ] = hat;
    }
}
void launch_layernorm(float *input,int batch_size,int hidden_size,float *input_hat,float *output,float *input_mean,float *input_std,float *normw,float *normb)
{
    dim3 grid(batch_size);
    dim3 block(MAX_THREADS);
    LayernormKernel<<<grid,block>>>(input,batch_size,hidden_size,input_hat,output,input_mean,input_std,normw,normb);
}

__global__ void LayernormBwKernel_wb(float *input_grad,float *input_hat,int batch_size,int hidden_size,float *normw_grad,float *normb_grad)
{
    __shared__ float dw[MAX_THREADS],db[MAX_THREADS];
    int base = blockIdx.x; // hidden_size
    int tid = threadIdx.x; // batch_size
    int nw;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    dw[tid]=db[tid]=0;
    for (int i = tid; i < batch_size; i += blockDim.x)
    {
        nw = i * hidden_size + base;
        dw[tid] += input_grad[ nw ] * input_hat[ nw ];
        db[tid] += input_grad[ nw ];
        //printf("%.3f %.3f\n",scale,input[ base + tid ] * scale);
    }
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < batch_size && tid + s < blockDim.x)
        {
            dw[tid] += dw[tid + s];
            db[tid] += db[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        normw_grad[ blockIdx.x ] = dw[0];
        normb_grad[ blockIdx.x ] = db[0];
        //printf("%f %f\n",tot_mean,tot_std);
    }
}
__global__ void LayernormBwKernel_input(float *input_grad,float *input_hat,float *input_std,float *normw,int batch_size,int hidden_size,float *output)
{
    __shared__ float yw[MAX_THREADS],ywx[MAX_THREADS];
    int tid = threadIdx.x;
    int base = blockIdx.x * hidden_size;
    //int baseInput = ;
    //int baseOutput = blockIdx.x * m ;
    yw[tid]=ywx[tid]=0;
    for (int i = tid; i < hidden_size; i += blockDim.x)
    {
        yw[tid] += input_grad[ base + i ] * normw[ i ];
        ywx[tid] += input_grad[ base + i ] * normw[ i ] * input_hat[ base + i ];
        //printf("%.3f %.3f\n",scale,input[ base + tid ] * scale);
    }
        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   
        if(tid < s && tid + s < hidden_size && tid + s < blockDim.x)
        {
            yw[tid] += yw[tid + s];
            ywx[tid] += ywx[tid + s];
            //printf("%d %.3f\n",s,ds[tid]);
        }
        __syncthreads();
    }

    __shared__ float tot_yw, tot_ywx;
    if (tid == 0)
    {
        tot_yw = yw[0];
        tot_ywx = ywx[0];
    }
    __syncthreads();
    for (int i = tid; i < hidden_size; i += blockDim.x)
    {
        output[ base + i ] += (input_grad[ base + i ] * normw[ i ] - (tot_yw + input_hat[ base + i ] * tot_ywx) / hidden_size) * input_std[ blockIdx.x ];
    }
}
void launch_layernorm_bw(float *input_grad,int batch_size,int hidden_size,float *input_hat,float *input_mean,float *input_std,float *normw,float *output,float *normw_grad,float *normb_grad)
{
    dim3 grid(hidden_size);
    dim3 block(MAX_THREADS);
    //print(input_grad,1,batch_size,hidden_size);
    //print(input_hat,1,batch_size,hidden_size);
    LayernormBwKernel_wb<<<grid,block>>>(input_grad,input_hat,batch_size,hidden_size,normw_grad,normb_grad);
    //print(normw_grad,1,1,hidden_size);
    dim3 grid2(batch_size);
    dim3 block2(MAX_THREADS);
    LayernormBwKernel_input<<<grid2,block2>>>(input_grad,input_hat,input_std,normw,batch_size,hidden_size,output);
}

__global__ void DroupoutKernel(float *input,int tot_size,float *output,bool *mask,float ratio,int seed)
{
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    if (base >= tot_size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, base, 0, &state);
    float scale = 1.0f / (1.0f - ratio);
    bool pos;
    pos = (curand_uniform(&state) > ratio);
    mask[base] = pos;
    output[base] = pos * input[base] * scale;
}

void launch_dropout(float *input,int batch_size,int hidden_size,float *output,bool *mask,float ratio)
{
    dim3 grid((batch_size * hidden_size + MAX_THREADS - 1 )/ MAX_THREADS);
    dim3 block(MAX_THREADS);
    //print(input_grad,1,batch_size,hidden_size);
    //print(input_hat,1,batch_size,hidden_size);
    DroupoutKernel<<<grid,block>>>(input,batch_size * hidden_size,output,mask,ratio,0);
}

__global__ void DroupoutBwKernel(float *input,int tot_size,float *output,bool *mask,float ratio)
{
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    if (base >= tot_size) return;
    float scale = 1.0f / (1.0f - ratio);
    output[base] = mask[base] * input[base] * scale;
}

void launch_dropout_bw(float *input,int tot_size,float *output,bool *mask,float ratio)
{
    dim3 grid((tot_size + MAX_THREADS - 1 )/ MAX_THREADS);
    dim3 block(MAX_THREADS);
    //print(input_grad,1,batch_size,hidden_size);
    //print(input_hat,1,batch_size,hidden_size);
    DroupoutBwKernel<<<grid,block>>>(input,tot_size,output,mask,ratio);
}

__global__ void DroupoutResidualKernel(float *input,float *residual,int tot_size,float *output,bool *mask,float ratio,int seed)
{
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    if (base >= tot_size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, base, 0, &state);
    float scale = 1.0f / (1.0f - ratio);
    bool pos;
    pos = (curand_uniform(&state) > ratio);
    mask[base] = pos;
    output[base] = pos * input[base] * scale + residual[base];
}

void launch_dropout_res(float *input, float *residual, int tot_size, float *output,bool *mask,float ratio)
{
    dim3 grid((tot_size + MAX_THREADS - 1 )/ MAX_THREADS);
    dim3 block(MAX_THREADS);
    //print(input_grad,1,batch_size,hidden_size);
    //print(input_hat,1,batch_size,hidden_size);
    DroupoutResidualKernel<<<grid,block>>>(input,residual,tot_size,output,mask,ratio,0);
}

__global__ void DroupoutResidualBwKernel(float *output_grad, bool *mask, int tot_size, float *output,float *input_grad, float ratio)
{
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    if (base >= tot_size) return;

    float scale = 1.0f / (1.0f - ratio);
    output[base] = mask[base] * output_grad[base] * scale;
    input_grad[base] = output_grad[base];
}

void launch_dropout_res_bw(float *output_grad, bool *mask, int tot_size, float *output,float *input_grad,float ratio)
{
    dim3 grid((tot_size + MAX_THREADS - 1 )/ MAX_THREADS);
    dim3 block(MAX_THREADS);
    //print(input_grad,1,batch_size,hidden_size);
    //print(input_hat,1,batch_size,hidden_size);
    DroupoutResidualBwKernel<<<grid,block>>>(output_grad, mask, tot_size, output, input_grad, ratio);
}


float *cuda_malloc(size_t ele_num)
{
    size_t byte_size = ele_num * sizeof(float);
    float *a = NULL;    
    cudaMalloc((void**)&a, byte_size * sizeof(float));
    return a;
}

bool *cuda_malloc2(size_t ele_num)
{
    size_t byte_size = ele_num * sizeof(bool);
    bool *a = NULL;    
    cudaMalloc((void**)&a, byte_size * sizeof(bool));
    return a;
}

void cuda_free(float *x)
{
    cudaFree(x);
}


void cuda_synchronize()
{
    cudaThreadSynchronize();
}

void gpuMemReport()
{
    size_t avail = 0;
    size_t total = 0;
    size_t free = 0;
    char tstring[32] = { '\0' };
    cudaMemGetInfo(&avail, &total);  
    printf("%s Memory avaliable: Free: %zu, Total: %zu, %s: %zu \n", tstring, avail, total,"Allocated:\0" , (total - avail) );
}

