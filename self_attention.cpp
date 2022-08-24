#include <torch/extension.h>
#include "self_attention.h"


void torch_launch_self_attention(torch::Tensor & input,torch::Tensor &wq,torch::Tensor &wk, torch::Tensor &wv,int len,int input_size,int output_size,torch::Tensor &output)
{
    launch_self_attention((float* )input.data_ptr(),(float*) wq.data_ptr(),(float*) wk.data_ptr(), (float*) wv.data_ptr(),len,input_size,output_size,(float*) output.data_ptr());

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_self_attention",
          &torch_launch_self_attention,
          "self attention kernel");
}