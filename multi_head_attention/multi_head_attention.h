//void launch_multi_head_attention(float* input,float* qkv,float* o,int batch_size,int tgt_len,int head_num,int hidden_size,float* output);

float *cuda_malloc(size_t ele_num);

void launch_matrixmul2(float* c,
                       float* a,
                       float* b,
                 int batch_size, int m , int n , int k);

void launch_matrixT(float *input,float *output,int batch_size,int m,int n);
void launch_softmax(float *input,int batch_size,int tgt_len,float *output,float scale);
void launch_transform_20314(float *input,int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,float *output);
void launch_transform_0213(float *input,int dim_0, int dim_1, int dim_2, int dim_3,float *output);
void launch_transform_021(float *input,int dim_0, int dim_1, int dim_2,float *output);
void cuda_synchronize();