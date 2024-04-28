# CS-7643: Deep-Learning
## Learning from Demonstrations using Parameter-efficient Finetuning of Language Models

The knowledge of Language Models (LMs) as next-word predictors have been proven to be very powerful recently.
However, there is still a challenge in aligning their responses to user objectives. The ideal way is to finetune them
for a downstream task, but that is inefficient and resource-consuming due to the scale of these models. Recently, in-
context learning, which is prompting LMs with in-context examples to get results for a task under zero-shot settings,
has also produced great results. It stands out for its efficiency; however, for long sequence tasks, it encounters
limitations due to restrictive context size of LMs, i.e., a maximum cap of sequence length under which they produce
good results. When providing many in-context examples, it is natural that the prompt size for long-sequence tasks
can exceed this limit. We aim to bridge this gap between finetuning and in-context learning by building a technique
that is efficient and improves outputs without expensive finetuning/training these large LMs on large datasets.


## Environment Setup

```conda create --name dl_project python=3.9```

```conda activate dl_project ```

```pip install -r requirements.txt```


## CLI

Dataset name options: qqp, mnli, rte

To run the model:

```python main.py --checkpoint bert-base-uncased --dataset qqp```

## Running with Adapters
Example command:
```python3 main.py --dataset qqp --model_name opt-125m --task_type icl --device cpu --batch_size 8 --adapter_name LORA```

Current accepted adapters: LORA, PREFIX_TUNING, ADALORA, IA3, LOHA, LOKR, OFT
