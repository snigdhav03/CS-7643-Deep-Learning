from peft import LoraConfig, TaskType, get_peft_model, PeftConfig

def add_adapter(model, adapter_name='LORA', config=None):
    print("Adding Adapter")
    if config is None:
        #config = PeftConfig(peft_type = adapter_name, task_type = "CAUSAL_LM", inference_mode=False)
        config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"])
    
    model = get_peft_model(model, config)
    # print(model.print_trainable_parameters())
    # print(model)
    return model