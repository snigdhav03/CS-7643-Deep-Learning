from peft import LoraConfig, TaskType, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, PromptEncoderConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig, MultitaskPromptTuningConfig, OFTConfig
def add_adapter(self):
    print("Adding Adapter")
    if self.adapter_config is None:
            config = get_adapter(self)
    else:
        config = self.adapter_config

    
    model = get_peft_model(self.model, config)
    #model = add_prompt_tuning(model, num_prompts=10, prompt_length=20, freeze_model=True)
    print(model.print_trainable_parameters())
    print(model)
    return model

def get_adapter(self):
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
                        ),
        'LOHA': LoHaConfig(
                            r=8,
                            target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                            rank_dropout=0.0,
                            module_dropout=0.0,
                            init_weights=True,
                        ),
        'LOKR': LoKrConfig(
                            r=8,
                            target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                            rank_dropout=0.1,
                            module_dropout=0.1,
                            init_weights=True,
                        ),
        'OFT': OFTConfig(
                            r=8,
                            target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                            module_dropout=0.1,
                            init_weights=True,
                        )
    }
    return adapter_configs[self.adapter_name]