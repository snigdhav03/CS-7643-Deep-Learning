
from src.trainer import CustomTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Custom Trainer')
    parser.add_argument('--checkpoint', type=str, default='bert-base-uncased', help='checkpoint model')
    parser.add_argument('--dataset', type=str, default='qqp', help='dataset name')
    args = parser.parse_args()

    custom_trainer = CustomTrainer(checkpoint=args.checkpoint, dataset=args.dataset)
    custom_trainer.train()


if __name__ == "__main__":
    main()