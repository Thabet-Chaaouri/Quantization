# Quantization

###  GPTQ quantization

AutoGPTQ library is integrated with transformers to:
- Quantize transformers
- Load GPTQ quantized models
- Finetune GPTQ quantized models using Lora adapters (full finetuning not possible on GPTQ quantized models) :

```
model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config_loading, device_map="auto")
```

###  Bitsandbytes quantization
