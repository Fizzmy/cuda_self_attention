#include "multi_head_layer.h"
#include "multi_head_attention.h"
#include <math.h>
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
void MultiHeadLayer::SetWeight(float *input_w,float *qkv_w,float *o_w)
{
    qkv=qkv_w;
    o=o_w;
    input=input_w;
}
void MultiHeadLayer::FreeAll(){
    cuda_free(QKVT);
    cuda_free(softmax_output);
    cuda_free(softmax_T);
    // float *QKV,*QKVT,*softmax_input,*KT,*softmax_output,*softmax_T;
    //     float *qkv,*o,*input;
}

void MultiHeadLayer::Forward(float* output)
{
    int size = batch_size * tgt_len * hidden_size;
    int output_size = hidden_size / head_num;
    float *QKV=cuda_malloc(3 * size);
    QKVT=cuda_malloc(3 * size);
    float *softmax_input=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    float *KT=cuda_malloc(size);

    launch_matrixmul2(QKV,input,qkv,1, batch_size * tgt_len, 3 * hidden_size, hidden_size);
    cuda_synchronize();

    launch_transform_20314(QKV,batch_size,tgt_len,3,head_num,output_size,QKVT);
    cuda_synchronize();
    cuda_free(QKV);

    // QKVT : 3  * batch_size * head_num * tgt_len * output_size
    float *Q = QKVT;
    float *K = QKVT + batch_size * tgt_len * hidden_size;
    float *V = QKVT + 2 * batch_size * tgt_len * hidden_size;

    launch_transform_021(K,batch_size* head_num,tgt_len,output_size,KT);
    cuda_synchronize();

    launch_matrixmul2(softmax_input,Q,KT,batch_size * head_num, tgt_len, tgt_len, output_size);
    cuda_synchronize();
    cuda_free(KT);
    
    if (tgt_len>1024) throw std::runtime_error("Sequence length greater than 1024 is currently not supported");
    softmax_output=cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    launch_softmax(softmax_input ,batch_size * head_num * tgt_len , tgt_len, softmax_output, sqrt((float)1.0/output_size));
    cuda_synchronize();
    cuda_free(softmax_input);

    launch_matrixmul2(output,softmax_output,V,batch_size * head_num, tgt_len, output_size, tgt_len);
    cuda_synchronize();

    // print(output , batch_size *head_num, tgt_len , output_size);
    // cudaThreadSynchronize();
   
    softmax_T=cuda_malloc( batch_size * tgt_len * hidden_size);
    launch_transform_0213(output,batch_size,head_num,tgt_len,output_size,softmax_T);
    cuda_synchronize();

    launch_matrixmul2(output,softmax_T,o,1,batch_size * tgt_len, hidden_size,  hidden_size);
    cuda_synchronize();
}


void MultiHeadLayer::Backward(float* grad, float* input_grad,float* qkv_grad,float* o_grad)
{
    float *softmax_T021 = cuda_malloc( batch_size * tgt_len * hidden_size );
    float *o_T = cuda_malloc( hidden_size * hidden_size ); 
    launch_transform_021(softmax_T, 1, batch_size * tgt_len, hidden_size, softmax_T021);
    launch_transform_021(o, 1, hidden_size, hidden_size, o_T);
    cuda_synchronize();
    float *softmax_T_grad = cuda_malloc( batch_size * tgt_len * hidden_size );
    launch_matrixmul2(o_grad, softmax_T021, grad , 1, hidden_size, hidden_size, batch_size * tgt_len);
    launch_matrixmul2(softmax_T_grad, grad, o_T ,1, batch_size * tgt_len, hidden_size, hidden_size);
    cuda_synchronize();
    cuda_free(softmax_T021);
    cuda_free(o_T);
    
    float *softmax_T_grad_T = cuda_malloc(batch_size*tgt_len*hidden_size);
    float *softmax_output_T = cuda_malloc(batch_size*head_num*tgt_len*tgt_len);
    float *QKV_grad = cuda_malloc(3 * batch_size * tgt_len * hidden_size);
    float *VT = cuda_malloc( batch_size * tgt_len * hidden_size);
    float *softmax_output_grad = cuda_malloc(batch_size*head_num*tgt_len*tgt_len);
    
    float *Q_grad = QKV_grad;
    float *K_grad = QKV_grad + batch_size * tgt_len * hidden_size;
    float *V_grad = QKV_grad + 2 * batch_size * tgt_len * hidden_size;
    float *Q = QKVT;
    float *K = QKVT + batch_size * tgt_len * hidden_size;
    float *V = QKVT + 2 * batch_size * tgt_len * hidden_size;
    launch_transform_0213(softmax_T_grad,batch_size,tgt_len,head_num,output_size,softmax_T_grad_T);
    launch_transform_021(softmax_output,batch_size*head_num,tgt_len,tgt_len,softmax_output_T);
    launch_transform_021(V, batch_size*head_num, tgt_len, output_size, VT);
    cuda_synchronize();
    launch_matrixmul2(V_grad, softmax_output_T, softmax_T_grad_T, batch_size * head_num, tgt_len, output_size, tgt_len);
    launch_matrixmul2(softmax_output_grad, softmax_T_grad_T , VT, batch_size*head_num, tgt_len, tgt_len, output_size);
    cuda_synchronize();
    cuda_free(softmax_T_grad);
    cuda_free(softmax_T_grad_T);
    cuda_free(softmax_output_T);
    cuda_free(VT);
    
    float *softmax_input_grad = cuda_malloc(batch_size*head_num*tgt_len*tgt_len);
    launch_softmax_bw(softmax_output,softmax_output_grad,batch_size*head_num*tgt_len, tgt_len, softmax_input_grad, sqrt((float)1.0/output_size));
    cuda_synchronize();
    cuda_free(softmax_output_grad);

    float *softmax_input_grad_T = cuda_malloc(batch_size * head_num * tgt_len * tgt_len);
    launch_transform_021(softmax_input_grad, batch_size * head_num , tgt_len, tgt_len, softmax_input_grad_T);
    cuda_synchronize();
    launch_matrixmul2(Q_grad, softmax_input_grad, K , batch_size * head_num , tgt_len, output_size, tgt_len); 
    launch_matrixmul2(K_grad, softmax_input_grad_T, Q, batch_size * head_num , tgt_len, output_size, tgt_len);
    cuda_synchronize();
    cuda_free(softmax_input_grad);
    cuda_free(softmax_input_grad_T);

    float *QKV_grad_T = cuda_malloc(3 * batch_size * tgt_len * hidden_size);
    launch_transform_13024(QKV_grad, 3, batch_size, head_num, tgt_len, output_size, QKV_grad_T);
    cuda_synchronize();
    cuda_free(QKV_grad);
    
    float *input_T = cuda_malloc( batch_size * tgt_len * hidden_size);
    float *qkv_T = cuda_malloc( 3 * batch_size * tgt_len * hidden_size);
    launch_transform_021(input, 1, batch_size * tgt_len, hidden_size, input_T);
    launch_transform_021(qkv , 1, hidden_size, 3 * hidden_size, qkv_T);
    cuda_synchronize();
    launch_matrixmul2(qkv_grad, input_T, QKV_grad_T, 1, hidden_size, 3*hidden_size, batch_size*tgt_len);
    launch_matrixmul2(input_grad, QKV_grad_T, qkv_T, 1, batch_size * tgt_len, hidden_size, 3 * hidden_size);
    cuda_synchronize();

}