from peft import LoraConfig, TaskType, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, PromptEncoderConfig, IA3Config, AdaLoraConfig

def add_adapter(model, adapter_name='LORA', config=None):
    print("Adding Adapter")
    if config is None:
            config = get_adapter(adapter_name)
    
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    print(model)
    return model

def get_adapter(adapter_name='LORA'):
    # Adapter configuration mappings
    adapter_configs = {
        'LORA': LoraConfig(r=64, lora_alpha=128, lora_dropout=0.01, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]),
        'PREFIX_TUNING': PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30),
        'ADALORA': AdaLoraConfig(
                    peft_type="ADALORA", task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["lm_head", "q_proj", "v_proj"],
                    lora_dropout=0.01,
                    ),
        'IA3': IA3Config(
                            peft_type="IA3",
                            task_type=TaskType.CAUSAL_LM,
                            target_modules=["lm_head", "q_proj", "v_proj"],
                            feedforward_modules=["lm_head"],
                        )
    }
    return adapter_configs[adapter_name]