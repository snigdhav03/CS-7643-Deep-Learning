# Exploring Efficient Finetuning of Language Models

![Type](https://img.shields.io/badge/Type-Course_Project-yellow)
![Concepts](https://img.shields.io/badge/Concepts-Deep_Learning,_Natural_Language_Processing-blue)
![Language](https://img.shields.io/badge/Language-Python-red)
![Libraries](https://img.shields.io/badge/Libraries-PyTorch,_Huggingface-green)

This work introduces a novel approach that utilizes parameter-efficient fine-tuning (PEFT) techniques to adapt language models (LMs) for downstream text classification tasks with significantly reduced resource requirements. We experiment with a variety of adapters, including LoRA, Prefix Tuning, LoHA, LoKR, IA3, OFT, and AdaLoRA, to demonstrate that substantial reductions in the number of trainable parameters can be achieved while maintain- ing competitive performance when finetuning LMs for a downstream task. We use Quora Question Pairs (QQP) dataset from the GLUE benchmark to evaluate the effec- tiveness of these adapters on the OPT model variants ranging from 125M to 2.7B parameters. Our findings indicate that adapters like LoRA not only preserve the capabilities of LMs but also enhance their adaptability to specific tasks, offering an optimal balance between performance and efficiency. They achieve accuracy within 0.75% while finetuning only 6.6% of parameters and using only using 7.37% extra memory. The results highlight the potential of scalable, adaptable LMs in practical applications which will be beneficial when finetuning in resource-constrained environments.

Link to full report containing all the results - [https://drive.google.com/file/d/1WTyz8BwX-6qywZ-pj_ZVxxJHA7lkqMY0/view?usp=share_link](https://drive.google.com/file/d/1WTyz8BwX-6qywZ-pj_ZVxxJHA7lkqMY0/view?usp=share_link)

## Usage

### Setup

1. Clone the Repository

   ```sh
   git clone https://github.com/snigdhav03/CS-7643-Deep-Learning.git
   cd CS-7643-Deep-Learning
   ```
2. Create Environment
   
```sh
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```

3. Start Executing

### Arguments 

To run the script, you need to use Python's `argparse` to specify the task and various options. Below are the arguments available.

- `--model_name`: Specifies the model to use for training and evaluation. Choices are:
    - `opt-125m`
    - `opt-350m`
    - `opt-1.3b`
    - `opt-2.7b`
    - `opt-6.7b`
    - `opt-13b`
    - `opt-30b`
    - `opt-66b`
- `--dataset`: Specifies the dataset to use. Choices are `qqp` and `mnli`, however, only qqp is supported presently.
- `--task_type`: Specifies the task type. Choices are `train` and `evaluate`. Train is used for fine-tuning.
- `--evaluation_mode`: Specifies the evaluation mode. Choices are `icl` for in-context learning and `sft` for only using the task instance.
- `--batch_size`: Specifies the batch size for training. Default is 8.
- `--device`: Specifies the device to use for training. Default is `cpu`.
- `--adapter_name`: Specifies the adapter with which to fine-tune. Default is `None`.
- `--checkpoint`: Specifies the path to the checkpoint to load. Default is `None`.


### Execution

Example command:

Fine-tuning a model (with/without adapter) -

```sh
python3 main.py --dataset qqp --model_name opt-125m --task_type train --device cuda --batch_size 8 --adapter_name LORA
```

If end-to-end finetuning is needed, don't provide a Language Model

Evaluating a model from a saved checkpoint in $cache$ folder created after fine-tuning -

```sh
python3 main.py --dataset qqp --model_name opt-125m --task_type train --device cuda --batch_size 8 --adapter_name LORA --checkpoint <PATH_TO_CHECKPOINT_IN_CACHE_DIR> --evaluation_mode icl
```

Current accepted adapters: LORA, PREFIX_TUNING, ADALORA, IA3, LOHA, LOKR, OFT
Current supported LMs: All versions of OPTs
Datasets: [Quora Question Pairs](https://huggingface.co/datasets/merve/qqp)

