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


In terms of inference speed, GPTQ generally outperforms bitsandbytes, but bitsandbytes can be faster for fine-tuning.
