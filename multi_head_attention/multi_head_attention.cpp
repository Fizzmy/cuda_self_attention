#include <torch/extension.h>
#include "multi_head_attention.h"
#include "multi_head_layer.h"
#include <vector>

MultiHeadLayer* layer=new MultiHeadLayer();

void torch_launch_multi_head_attention(torch::Tensor & input,torch::Tensor &weight,torch::Tensor &bias,torch::Tensor &qkv,torch::Tensor &o,int batch_size,int tgt_len,int head_num,int hidden_size,torch::Tensor &output)
{
    // printf("-------------------------------------------------\n");
    // gpuMemReport();
    layer->SetConfig(batch_size,tgt_len,head_num,hidden_size);
    layer->SetWeight((float* )input.data_ptr(),(float* )weight.data_ptr(),(float* )bias.data_ptr(),(float*) qkv.data_ptr(),(float*) o.data_ptr());
    layer->Forward((float*) output.data_ptr());
    // gpuMemReport();
}
//grad,inp,qkv,o,input_grad,qkv_grad,o_grad
std::vector<torch::Tensor> torch_launch_multi_head_attention_bw(torch::Tensor & grad)
{
    //gpuMemReport();
    auto input_grad = torch::empty({layer->getBatchSize(),layer->getTgtLen(),layer->getHiddenSize()}, torch::TensorOptions().dtype(torch::kFloat32).device(grad.device()));
    auto normw_grad = torch::empty({layer->getHiddenSize()}, torch::TensorOptions().dtype(torch::kFloat32).device(grad.device()));
    auto normb_grad = torch::empty({layer->getHiddenSize()}, torch::TensorOptions().dtype(torch::kFloat32).device(grad.device()));
    auto qkv_grad = torch::empty({layer->getHiddenSize(),3*layer->getHiddenSize()}, torch::TensorOptions().dtype(torch::kFloat32).device(grad.device()));
    auto o_grad = torch::empty({layer->getHiddenSize(),layer->getHiddenSize()}, torch::TensorOptions().dtype(torch::kFloat32).device(grad.device()));
    //gpuMemReport();
    layer->Backward((float* )grad.data_ptr(),(float* )input_grad.data_ptr(),(float* )normw_grad.data_ptr(),(float* )normb_grad.data_ptr(),(float* )qkv_grad.data_ptr(),(float* )o_grad.data_ptr());
    layer->FreeAll();
    //gpuMemReport();
    return {input_grad,normw_grad,normb_grad,qkv_grad,o_grad};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_multi_head_attention",
          &torch_launch_multi_head_attention,
          "self attention forward kernel");
    m.def("torch_launch_multi_head_attention_bw",
          &torch_launch_multi_head_attention_bw,
          "self attention backward kernel");
}