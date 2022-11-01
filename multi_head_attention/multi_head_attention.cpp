#include <torch/extension.h>
#include "multi_head_attention.h"
#include "multi_head_layer.h"
#include <vector>

MultiHeadLayer* layer=new MultiHeadLayer();

void init(int batch_size,int tgt_len,int head_num,int hidden_size,float atten_dropout_radio,float output_dropout_radio)
{
    layer->SetConfig(batch_size,tgt_len,head_num,hidden_size,atten_dropout_radio,output_dropout_radio);
    layer->SetMalloc(); 
    layer->SetGrad();

}
torch::Tensor torch_launch_multi_head_attention(torch::Tensor & input,torch::Tensor & mask,torch::Tensor &weight,torch::Tensor &bias,torch::Tensor &qkv,torch::Tensor &o,bool isPre,bool isTraining)
{
    // printf("-------------------------------------------------\n");
    // gpuMemReport();
    layer->SetWeight((float* )input.data_ptr(),(float* )weight.data_ptr(),(float* )bias.data_ptr(),(float*) qkv.data_ptr(),(float*) o.data_ptr(),(bool*)mask.data_ptr(),isPre,isTraining);
    layer->Forward((float*) layer->output.data_ptr());
    return layer->output;
    // gpuMemReport();
}
//grad,inp,qkv,o,input_grad,qkv_grad,o_grad
std::vector<torch::Tensor> torch_launch_multi_head_attention_bw(torch::Tensor & grad)
{
    //gpuMemReport();
    //gpuMemReport();
    layer->Backward((float* )grad.data_ptr(),(float* )layer->input_grad.data_ptr(),(float* )layer->normw_grad.data_ptr(),(float* )layer->normb_grad.data_ptr(),(float* )layer->qkv_grad.data_ptr(),(float* )layer->o_grad.data_ptr());
    // layer->FreeAll();
    //gpuMemReport();
    return {layer->input_grad,layer->normw_grad,layer->normb_grad,layer->qkv_grad,layer->o_grad};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init",
        &init,
        "init layer");
    m.def("torch_launch_multi_head_attention",
          &torch_launch_multi_head_attention,
          "self attention forward kernel");
    m.def("torch_launch_multi_head_attention_bw",
          &torch_launch_multi_head_attention_bw,
          "self attention backward kernel");
}