# Quantization

###  GPTQ quantization

GPTQ uses post-training quantization (PTQ) to compress models and achieve smaller sizes with a calibration dataset.

AutoGPTQ library is integrated with transformers to:
- Quantize transformers
- Load GPTQ quantized models
- Finetune GPTQ quantized models using Lora adapters (full finetuning not possible on GPTQ quantized models) :

```
model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config_loading, device_map="auto")
```

Checkout this [blog](https://huggingface.co/blog/gptq-integration)

###  Bitsandbytes quantization
bitsandbytes can do integer quantization but does not require an input mini-batch for quantization. Hence applicable on any model directly.

4 bit quantization [blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

8 bit quantization [blog](https://huggingface.co/blog/hf-bitsandbytes-integration)

#### Comparing bitsandbytes and auto-gptq

Checkout this [blog](https://huggingface.co/blog/overview-quantization-transformers)

Bitsandbytes:
- It does not require calibrating the quantized model with input data as long as it contains torch.nn.Linear modules. Quantization is performed on model load.
- merging adapteers on top of the quantized base model with 0 performance degradation. It is poqqible to merge the adapters on top of the dequantized model (This is not supported for GPTQ)

AutoGPTQ:
- In terms of inference speed, GPTQ generally outperforms bitsandbytes, but bitsandbytes can be faster for fine-tuning.
- It is possible to quantize models up to 2 bits.
- supports serialization for any number of bits. Bitsandbytes supports 8-bit serialization but does not support 4-bit serialization as of today

Bitsandbytes is better suited for fine-tuning while GPTQ is better for generation. From this observation, one way to get better merged models would be to:
- Quantize the base model using bitsandbytes (zero-shot quantization)
- add and fine-tune the adapters
- merge the trained adapters on top of the dequantized model
- quantize the merged model using GPTQ and use it for deployment

###  AWQ quantization

AWQ quantization, you can look at different existing tools in the ecosystem to quantize their models with AWQ algorithm, such as:
- [llm-awq](https://github.com/mit-han-lab/llm-awq) from MIT Han Lab
- [autoawq](https://github.com/casper-hansen/AutoAWQ) from casper-hansen


#### Benchmark

Check out this [link](https://huggingface.co/docs/transformers/v4.35.0/main_classes/quantization#quantize--transformers-models) for a detailed speed, throughput and latency benchmarks.

