#include <torch/extension.h>
#include "multi_head_layer.h"
#include "multi_head_attention.h"
#include <math.h>
#include <stdexcept>
#include <cublas_v2.h>

MultiHeadLayer::MultiHeadLayer(){}

MultiHeadLayer::~MultiHeadLayer(){}


int MultiHeadLayer::getBatchSize()
{
    return batch_size;
}
int MultiHeadLayer::getTgtLen()
{
    return tgt_len;
}
int MultiHeadLayer::getHeadNum()
{
    return head_num;
}
int MultiHeadLayer::getHiddenSize()
{
    return hidden_size;
}
int MultiHeadLayer::getOutputSize()
{
    return output_size;
}
void MultiHeadLayer::SetConfig(int bs,int tl,int hn,int hs,float adr,float odr)
{
    batch_size=bs;
    tgt_len=tl;
    head_num=hn;
    hidden_size=hs;
    output_size=hs/hn;
    atten_dropout_radio = adr;
    output_dropout_radio = odr;
    cublasCreate(&handle);
}
void MultiHeadLayer::SetMalloc()
{
    int size = batch_size * tgt_len * hidden_size;
    int output_size = hidden_size / head_num;
    QKV=cuda_malloc(3 * size);
    QKVT=cuda_malloc(3 * size);
    softmax_input=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    input_norm=cuda_malloc(size);
    input_mean=cuda_malloc(batch_size * tgt_len);
    input_std=cuda_malloc(batch_size * tgt_len);
    input_hat=cuda_malloc(batch_size * tgt_len * hidden_size);
    Q = QKVT;
    K = QKVT + batch_size * tgt_len * hidden_size;
    V = QKVT + 2 * batch_size * tgt_len * hidden_size;
    softmax_output=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    softmax_T=cuda_malloc( batch_size * tgt_len * hidden_size);

    softmax_T_grad = cuda_malloc( batch_size * tgt_len * hidden_size );
    softmax_T_grad_T = cuda_malloc(batch_size*tgt_len*hidden_size);
    QKV_grad = cuda_malloc(3 * batch_size * tgt_len * hidden_size);
    softmax_output_grad = cuda_malloc(batch_size*head_num*tgt_len*tgt_len);
    Q_grad = QKV_grad;
    K_grad = QKV_grad + batch_size * tgt_len * hidden_size;
    V_grad = QKV_grad + 2 * batch_size * tgt_len * hidden_size;
    softmax_input_grad = cuda_malloc(batch_size*head_num*tgt_len*tgt_len);
    QKV_grad_T = cuda_malloc(3 * batch_size * tgt_len * hidden_size);
    norm_grad = cuda_malloc( batch_size * tgt_len * hidden_size);

    softmax_dropout = cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    output_dropout = cuda_malloc(batch_size * tgt_len * hidden_size);
    atten_mask = cuda_malloc2(batch_size * head_num * tgt_len * tgt_len);
    output_mask = cuda_malloc2(batch_size * tgt_len * hidden_size);
}
void MultiHeadLayer::SetGrad()
{
    input_grad = torch::empty({batch_size, tgt_len, hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    normw_grad = torch::empty({hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    normb_grad = torch::empty({hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    qkv_grad = torch::empty({hidden_size, 3*hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    o_grad = torch::empty({hidden_size, hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    output = torch::empty({batch_size,tgt_len,hidden_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}
void MultiHeadLayer::SetWeight(float *input_w,float *norm_w,float *norm_b,float *qkv_w,float *o_w,bool *mask_w, bool is_pre, bool is_training)
{
    qkv=qkv_w;
    mask=mask_w;
    o=o_w;
    input=input_w;
    normw=norm_w;
    normb=norm_b;
    isPre=is_pre;
    isTraining = is_training;
}
void MultiHeadLayer::FreeAll(){
    cuda_free(QKVT);
    cuda_free(softmax_output);
    cuda_free(softmax_T);
    cuda_free(input_norm);
    cuda_free(input_mean);
    cuda_free(input_std);
    cuda_free(input_hat);
    // float *QKV,*QKVT,*softmax_input,*KT,*softmax_output,*softmax_T;
    //     float *qkv,*o,*input;
}

void MultiHeadLayer::Forward(float* output)
{
    int size = batch_size * tgt_len * hidden_size;
    int output_size = hidden_size / head_num;
    
    launch_layernorm(input,batch_size * tgt_len, hidden_size, input_hat, input_norm, input_mean, input_std, normw, normb);
    cuda_synchronize();

    launch_matrixmul(handle,QKV,input_norm,qkv,1, batch_size * tgt_len, 3 * hidden_size, hidden_size, 0, 0);
    cuda_synchronize();

    launch_transform_20314(QKV,batch_size,tgt_len,3,head_num,output_size,QKVT);
    cuda_synchronize();
    
    launch_matrixmul(handle,softmax_input,Q,K,batch_size * head_num, tgt_len, tgt_len, output_size, 0, 1, output_size);
    cuda_synchronize();
    
    // if (tgt_len>1024) throw std::runtime_error("Sequence length greater than 1024 is currently not supported");
    float *softmax_dropout_input = NULL;
    if (isTraining)
    {
        //printf("begin\n");
        launch_dropout(softmax_input, batch_size * head_num * tgt_len, tgt_len, softmax_dropout, atten_mask, atten_dropout_radio);
        cuda_synchronize();
        //printf("end\n");
        softmax_dropout_input = softmax_dropout;
    }
    else
        softmax_dropout_input = softmax_input;

    //print(softmax_dropout_input,batch_size*head_num,tgt_len,tgt_len);
    launch_softmax(softmax_dropout_input, mask, batch_size, head_num, tgt_len, softmax_output, isPre);
    cuda_synchronize();

    launch_matrixmul(handle,output,softmax_output,V,batch_size * head_num, tgt_len, output_size, tgt_len, 0, 0);
    cuda_synchronize();

    // print(output , batch_size *head_num, tgt_len , output_size);
    // cudaThreadSynchronize();
   
    launch_transform_0213(output,batch_size,head_num,tgt_len,output_size,softmax_T);
    cuda_synchronize();

    launch_matrixmul(handle,output_dropout,softmax_T,o,1,batch_size * tgt_len, hidden_size,  hidden_size, 0, 0);
    cuda_synchronize();

    if (isTraining)
        launch_dropout_res(output_dropout, input, batch_size * tgt_len * hidden_size, output, output_mask, output_dropout_radio);
    else
        launch_dropout_res(output_dropout, input, batch_size * tgt_len * hidden_size, output, output_mask, 0.0);
    cuda_synchronize();

}
void MultiHeadLayer::Backward(float* grad, float* input_grad,float* normw_grad,float *normb_grad,float* qkv_grad,float* o_grad)
{
    launch_dropout_res_bw(grad, output_mask, batch_size * tgt_len * hidden_size, output_dropout, input_grad, output_dropout_radio);
    cuda_synchronize();

    launch_matrixmul(handle,o_grad, softmax_T, output_dropout , 1, hidden_size, hidden_size, batch_size * tgt_len, 1, 0);
    launch_matrixmul(handle,softmax_T_grad, output_dropout, o ,1, batch_size * tgt_len, hidden_size, hidden_size, 0, 1);
    cuda_synchronize();
    
    launch_transform_0213(softmax_T_grad,batch_size,tgt_len,head_num,output_size,softmax_T_grad_T);
    cuda_synchronize();
    launch_matrixmul(handle,V_grad, softmax_output, softmax_T_grad_T, batch_size * head_num, tgt_len, output_size, tgt_len, 1, 0);
    launch_matrixmul(handle,softmax_output_grad, softmax_T_grad_T , V, batch_size*head_num, tgt_len, tgt_len, output_size, 0, 1);
    cuda_synchronize();
    
    launch_softmax_bw(softmax_output,softmax_output_grad,batch_size*head_num*tgt_len, tgt_len, softmax_input_grad, sqrt((float)1.0/output_size));
    cuda_synchronize();

    float *softmax_dropout_grad = NULL;
    if (isTraining)
    {
        launch_dropout_bw(softmax_input_grad,batch_size * head_num * tgt_len * tgt_len,softmax_dropout,atten_mask,atten_dropout_radio);
        softmax_dropout_grad = softmax_dropout;
    }
    else softmax_dropout_grad = softmax_input_grad;

    launch_matrixmul(handle,Q_grad, softmax_dropout_grad, K , batch_size * head_num , tgt_len, output_size, tgt_len, 0, 0); 
    launch_matrixmul(handle,K_grad, softmax_dropout_grad, Q, batch_size * head_num , tgt_len, output_size, tgt_len, 1, 0);
    cuda_synchronize();


    launch_transform_13024(QKV_grad, 3, batch_size, head_num, tgt_len, output_size, QKV_grad_T);
    cuda_synchronize();

    launch_matrixmul(handle,qkv_grad, input_norm, QKV_grad_T, 1, hidden_size, 3*hidden_size, batch_size*tgt_len, 1, 0);
    launch_matrixmul(handle,norm_grad, QKV_grad_T, qkv, 1, batch_size * tgt_len, hidden_size, 3 * hidden_size, 0, 1);
    cuda_synchronize();
    
    launch_layernorm_bw(norm_grad,batch_size*tgt_len,hidden_size,input_hat,input_mean,input_std,normw,input_grad,normw_grad,normb_grad);
    cuda_synchronize();
    

}