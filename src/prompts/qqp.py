import random

import torch


class QQPPrompt:
    def __init__(self, mode='icl', example=16):
        self._mode = mode
        self._example = example
        self._context_prefix = "Question: Do both questions ask the same thing? "

    def generate_context(self, batch, exclude_index):
        indices = list(range(len(batch['question1'])))
        indices.remove(exclude_index)
        random.shuffle(indices)
        sample_size = min(self._example, len(indices))
        context = [
            # self._context_prefix + "\n" +
            "Question 1: " + batch['question1'][i] + "\n" +
            "Question 2: " + batch['question2'][i] + "\n" +
            self._context_prefix +
            "Answer: " + ('Yes' if batch['label'][i].item() == 1 else 'No')
            for i in indices[:sample_size]
        ]

        return context

    def get_prompt(self, batch):
        prompts = []
        labels = batch['label']
        for idx in range(len(batch['question1'])):
            if self._mode == 'icl':
                context = self.generate_context(batch, idx)
            current_prompt = (
                    # self._context_prefix + "\n" +
                    "Question 1: " + batch['question1'][idx] + "\n" +
                    "Question 2: " + batch['question2'][idx] + "\n" +
                    self._context_prefix +
                    "Answer: "
            )
            if self._mode == 'icl':
                full_prompt = '\n\n'.join(context) + "\n\n" + current_prompt
            else:
                full_prompt = current_prompt
            prompts.append(full_prompt)

        return prompts, labels

    def extract_predicted_answer(self, preds):
        answers = []
        for pred in preds:
            last_answer_start = pred.rfind("Answer: ") + len("Answer: ")
            if last_answer_start > len("Answer: "):
                answer = pred[last_answer_start:].strip().split('\n')[0]
            else:
                answer = "No answer found"
            answers.append(answer)
        return answers

    def label_to_answer(self, label):
        result = ['No' if x == 0 else 'Yes' for x in label]
        return result

    def __call__(self, batch):
        return self.get_prompt(batch)
