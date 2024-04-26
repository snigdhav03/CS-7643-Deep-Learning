from peft import LoraConfig, TaskType, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit

# Adapter configuration mappings
adapter_configs = {
    'LORA': LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]),
    'PREFIX_TUNING': PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
    # Add more adapters here with their respective configurations
}

def add_adapter(model, adapter_name='LORA', config=None):
    print("Adding Adapter")
    if config is None:
        if adapter_name in adapter_configs:
            config = adapter_configs[adapter_name]
        else:
            raise ValueError(f"No configuration available for the adapter type: {adapter_name}")
    
    model = get_peft_model(model, config)
    # print(model.print_trainable_parameters())
    # print(model)
    return model