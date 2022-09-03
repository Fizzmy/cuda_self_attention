#include <torch/extension.h>
#include "multi_head_attention.h"
#include "multi_head_layer.h"

MultiHeadLayer* layer=new MultiHeadLayer();



void torch_launch_multi_head_attention(torch::Tensor & input,torch::Tensor &qkv,torch::Tensor &o,int batch_size,int tgt_len,int head_num,int hidden_size,torch::Tensor &output)
{
    layer->SetConfig(batch_size,tgt_len,head_num,hidden_size);
    layer->SetWeight((float* )input.data_ptr(),(float*) qkv.data_ptr(),(float*) o.data_ptr());
    layer->Forward((float*) output.data_ptr());
}
//grad,inp,qkv,o,input_grad,qkv_grad,o_grad
void torch_launch_multi_head_attention_bw(torch::Tensor & grad,torch::Tensor &input_grad,torch::Tensor &qkv_grad,torch::Tensor &o_grad)
{
    layer->Backward((float* )grad.data_ptr(),(float* )input_grad.data_ptr(),(float* )qkv_grad.data_ptr(),(float* )o_grad.data_ptr());
    layer->FreeAll();
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_multi_head_attention",
          &torch_launch_multi_head_attention,
          "self attention forward kernel");
    m.def("torch_launch_multi_head_attention_bw",
          &torch_launch_multi_head_attention_bw,
          "self attention backward kernel");
}