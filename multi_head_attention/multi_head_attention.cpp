#include <torch/extension.h>
#include "multi_head_attention.h"
#include "multi_head_layer.h"

MultiHeadLayer* layer=new MultiHeadLayer();



void torch_launch_multi_head_attention(torch::Tensor & input,torch::Tensor &qkv,torch::Tensor &o,int batch_size,int tgt_len,int head_num,int hidden_size,torch::Tensor &output)
{
    layer->SetConfig(batch_size,tgt_len,head_num,hidden_size);
    layer->Forward((float* )input.data_ptr(),(float*) qkv.data_ptr(),(float*) o.data_ptr(),(float*) output.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_multi_head_attention",
          &torch_launch_multi_head_attention,
          "self attention kernel");
}