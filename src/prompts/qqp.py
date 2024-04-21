import random


class QQPPrompt:
    def __init__(self, mode='icl', example=16):
        self._mode = mode
        self._example = example
        self._context_prefix = "Given two questions, question1 and question2, evaluate if both the questions are same or different. Return 1 for same and return 0 for different. Examples:\n\n"

    def generate_context(self, batch, exclude_index):
        # Generate context based on the batch data excluding the current instance
        indices = list(range(len(batch['question1'])))
        indices.remove(exclude_index)  # Remove the current index from context generation
        random.shuffle(indices)

        # Sample fewer examples if not enough entries are available after exclusion
        sample_size = min(self._example, len(indices))
        context = [

            "question1: " + batch['question1'][i] + "\n" +
            "question2: " + batch['question2'][i] + "\n" +
            "output: " + str(batch['label'][i].item())
            for i in indices[:sample_size]  # Use shuffled indices to pick examples
        ]

        return context

    def get_prompt(self, batch):
        prompts = []
        labels = batch['label']
        for idx in range(len(batch['question1'])):
            # Generate context excluding the current index
            context = self.generate_context(batch, idx)

            # Create the current prompt using string concatenation
            current_prompt = (
                    "question1: " + batch['question1'][idx] + "\n" +
                    "question2: " + batch['question2'][idx] + "\n" +
                    "output: "
            )

            # Combine context and current prompt
            full_prompt = '\n'.join(context) + "\n\n" + current_prompt
            full_prompt = self._context_prefix + full_prompt
            prompts.append(full_prompt)

        return prompts, labels

    def __call__(self, batch):
        return self.get_prompt(batch)
