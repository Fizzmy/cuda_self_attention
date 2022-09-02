#include "multi_head_layer.h"
#include "multi_head_attention.h"
#include <stdexcept>

MultiHeadLayer::MultiHeadLayer(){}

MultiHeadLayer::~MultiHeadLayer(){}

void MultiHeadLayer::SetConfig(int bs,int tl,int hn,int hs)
{
    batch_size=bs;
    tgt_len=tl;
    head_num=hn;
    hidden_size=hs;
    output_size=hs/hn;
}

void MultiHeadLayer::Forward(float* input,float* qkv,float* o,float* output)
{
    int size = batch_size * tgt_len * hidden_size;
    int output_size = hidden_size / head_num;
    QKV=cuda_malloc(3 * size);
    QKVT=cuda_malloc(3 * size);
    softmax_input=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    KT=cuda_malloc(size);

    launch_matrixmul2(QKV,input,qkv,1, batch_size * tgt_len, 3 * hidden_size, hidden_size);
    cuda_synchronize();

    launch_transform_20314(QKV,batch_size,tgt_len,3,head_num,output_size,QKVT);
    cuda_synchronize();

    // QKVT : 3  * batch_size * head_num * tgt_len * output_size
    float *Q = QKVT;
    float *K = QKVT + batch_size * tgt_len * hidden_size;
    float *V = QKVT + 2 * batch_size * tgt_len * hidden_size;

    launch_transform_021(K,batch_size* head_num,tgt_len,output_size,KT);
    cuda_synchronize();

    launch_matrixmul2(softmax_input,Q,KT,batch_size * head_num, tgt_len, tgt_len, output_size);
    cuda_synchronize();
    
    if (tgt_len>1024) throw std::runtime_error("Sequence length greater than 1024 is currently not supported");
    softmax_output=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    launch_softmax(softmax_input ,batch_size * head_num * tgt_len , tgt_len, softmax_output, sqrt((float)1.0/output_size));
    cuda_synchronize();

    launch_matrixmul2(output,softmax_output,V,batch_size * head_num, tgt_len, output_size, tgt_len);
    cuda_synchronize();

    // print(output , batch_size *head_num, tgt_len , output_size);
    // cudaThreadSynchronize();
   
    softmax_T=cuda_malloc( batch_size * tgt_len * hidden_size * sizeof(float));
    launch_transform_0213(output,batch_size,head_num,tgt_len,output_size,softmax_T);
    cuda_synchronize();

    launch_matrixmul2(output,softmax_T,o,1,batch_size * tgt_len, hidden_size,  hidden_size);
    cuda_synchronize();
}


void MultiHeadLayer::Backward(float* grad,float* input_grad,float* qkv_grad,float* o_grad)
{

}