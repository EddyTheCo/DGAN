#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>
#ifdef USE_YAML
#include<yaml-cpp/yaml.h>
#endif


namespace custom_models{
class DGANImpl : public torch::nn::Module {
    public:
        DGANImpl(const std::vector<size_t> &layers,double dropout_m=0.3,double leaky_relu_m=0.2):input_size(layers.front()),output_size(layers.back()),dropout_(dropout_m),leaky_relu_(leaky_relu_m)
        {
            for(auto i=0;i<layers.size()-1;i++)
            {
                module_cont.push_back(register_module(("fc"+std::to_string(i)).c_str(),
                                                      torch::nn::Linear(layers[i],layers[i+1])));
            }
        }
#ifdef USE_YAML
        DGANImpl(YAML::Node config):DGANImpl(config["layers"].as<std::vector<size_t>>(),
				config["dropout"].as<double>(),
				config["leaky relu"].as<double>()){std::cout<<config<<std::endl;};
#endif
        void update(void)const{};
        torch::Tensor forward(torch::Tensor x) {
            if(x.sizes().size()>2)x=x.view({x.size(0),x.numel()/x.size(0)});

            for(auto i=0;i<module_cont.size();i++)
            {
                if(i<module_cont.size()-1)
                {
                    x = torch::leaky_relu(module_cont[i](x),leaky_relu_);
                    x = torch::dropout(x, dropout_,is_training());
                }
                else
                {
                    x = torch::sigmoid(module_cont[i](x));
                }
            }

           return x;
        }
	int64_t get_numel(void)const
        {
            int64_t sum=0;
            for(auto mod:module_cont)
            {
                for (auto param:mod->parameters())
                {
                    sum+=param.numel();
                }
            }
            return sum;
        }
        const int64_t input_size,output_size;
	const double dropout_,leaky_relu_;
    private:
        std::vector<torch::nn::Linear> module_cont;

};
TORCH_MODULE(DGAN);

};
