//void launch_multi_head_attention(float* input,float* qkv,float* o,int batch_size,int tgt_len,int head_num,int hidden_size,float* output);
#include <stddef.h>
#include <cublas_v2.h>

float *cuda_malloc(size_t ele_num);
bool *cuda_malloc2(size_t ele_num);

void launch_matrixmul2(float* c,
                       float* a,
                       float* b,
                 int batch_size, int m , int n , int k);

void launch_matrixmul(cublasHandle_t handle,float* c,
                       float* a,
                       float* b,
                 int batch_size, int m , int n , int k, bool trans_A, bool trans_B,int scaler=0);

void launch_matrixT(float *input,float *output,int batch_size,int m,int n);
void launch_layernorm(float *input,int batch_size,int hidden_size,float *input_hat,float *output,float *input_mean,float *input_std,float *normw,float *normb);
void launch_layernorm_bw(float *input_grad,int batch_size,int hidden_size,float *input_hat,float *input_mean,float *input_std,float *normw,float *output,float *normw_grad,float *normb_grad);
void launch_softmax(float *input,bool *mask,int batch_size,int head_num,int tgt_len,float *output,bool is_pre);
void launch_softmax_bw(float *input,float *input_grad,int batch_size,int tgt_len,float *output,float scale);
void launch_transform_20314(float *input,int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,float *output);
void launch_transform_13024(float *input,int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,float *output);
void launch_transform_0213(float *input,int dim_0, int dim_1, int dim_2, int dim_3,float *output);
void launch_transform_021(float *input,int dim_0, int dim_1, int dim_2,float *output);

void launch_dropout(float *input,int batch_size,int hidden_size,float *output,bool *mask,float ratio);
void launch_dropout_bw(float *input,int tot_size,float *output,bool *mask,float ratio);
void launch_dropout_res(float *input, float *residual, int tot_size, float *output,bool *mask,float ratio);
void launch_dropout_res_bw(float *output_grad, bool *mask, int tot_size, float *output,float *input_grad,float ratio);

void cuda_synchronize();
void print(float *c,int batch_size,int m,int n);
void cuda_free(float *x);

void gpuMemReport();