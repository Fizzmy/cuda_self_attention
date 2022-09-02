class MultiHeadLayer{
    public:
        MultiHeadLayer();
        ~MultiHeadLayer();
        void SetConfig(int bs,int tl,int hn,int hs);
        void Forward(float* input,float* qkv,float* o,float* output);
        void Backward(float* grad,float* input_grad,float* qkv_grad,float* o_grad);
    private:
        int batch_size;
        int tgt_len;
        int head_num;
        int hidden_size;
        int output_size;
        float *QKV,*QKVT,*softmax_input,*KT,*softmax_output,*softmax_T;
};