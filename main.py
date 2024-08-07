from src.trainer import TrainerLM
import argparse
from src.const import datasets, models, task_types, eval_types
from src.utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Trainer')
    parser.add_argument('--model_name', type=str, default='opt-125m', help='Choose the model from the given choices',
                        choices=models)
    parser.add_argument('--dataset', type=str, default='qqp', help='Choose the dataset from given choices',
                        choices=datasets)
    parser.add_argument('--task_type', type=str, default='train', help='Choose the task type from the given choices',
                        choices=task_types)
    parser.add_argument('--evaluation_mode', type=str, default='icl', help='Choose the evaluation mode from the given choices',
                        choices=eval_types)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--adapter_name', type=str, default=None, help='Choose the adapter with which to fine-tune')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to load, eg: trainer_checkpoints/opt125mLORA/checkpoint-45000/pytorch_model.bin')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.task_type == 'train':
        print(f'Training LM')
        custom_trainer = TrainerLM(args.model_name, args.dataset, args.adapter_name, batch_size=args.batch_size, device=args.device, checkpoint=args.checkpoint)
        custom_trainer.train()
    if args.task_type == 'evaluate':
        evaluate(args.model_name, args.dataset, mode=args.evaluation_mode, batch_size=args.batch_size, device=args.device,
                            adapter_name=args.adapter_name, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
