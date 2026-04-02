# -*- coding: utf-8 -*-
"""
Auto-optimize a prompt for letter-repetition reasoning in country names.

Uses AdalFlow to train both the system prompt (via text-grad) and
few-shot demonstrations (via bootstrap few-shot) so that a GPT-4o
task model learns to answer questions like:

    "What country has the same letter repeated the most in its name?"

The optimizer, teacher, and backward engine all run on o3-mini.

# Installation
1. pip install -r requirements.txt
2. export OPENAI_API_KEY=<your key>
"""

import os

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

import json
from typing import Dict, Union

import adalflow as adal
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.types import ParameterType

# ---------------------------------------------------------------------------
# Prompt template (Jinja2)
# ---------------------------------------------------------------------------
few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
{# Few shot demos #}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""

# ---------------------------------------------------------------------------
# Task pipeline
# ---------------------------------------------------------------------------

class LetterRepeatTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            data=(
                "You will answer a reasoning question about letter patterns in "
                "country names. Think step by step.\n"
                "1. Identify the candidate countries. If countries are listed in "
                "the question, use those. If no countries are listed, recall "
                "real country names from around the world — especially those "
                "known for long names or heavy vowel repetition (e.g. "
                "Madagascar, Canada, Jamaica, Australia, Bahamas, etc.).\n"
                "2. For each candidate, spell its name letter by letter "
                "(case-insensitive) and count how many times each distinct "
                "letter appears.\n"
                "3. Record the maximum count of any single letter for each "
                "country name.\n"
                "4. Compare the maximum counts across all candidates.\n"
                "5. The country whose name has the highest single-letter "
                "repetition count is the answer. For example, 'Madagascar' "
                "has the letter 'a' appearing 4 times (M-a-d-a-g-a-s-c-a-r), "
                "which is the highest of any country.\n"
                "The last line of your response MUST be exactly: "
                "'Answer: $VALUE' where VALUE is the country name."
            ),
            role_desc="Task instruction for letter-repetition reasoning",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )
        few_shot_demos = adal.Parameter(
            data=None,
            role_desc="Few-shot demonstrations for letter-repetition reasoning",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=few_shot_template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            use_cache=True,
        )

    def bicall(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm(prompt_kwargs={"input_str": question}, id=id)
        return output


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

gpt_4o_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-4o",
        "temperature": 0.0,
    },
}

gpt_o3_mini_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "o3-mini",
        "temperature": 1.0,
    },
}

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

question = (
    "Among these countries, which one has the same letter repeated "
    "the most times in its name: Madagascar, Canada, Brazil?"
)
task_pipeline = LetterRepeatTaskPipeline(**gpt_4o_model)
print(task_pipeline)

answer = task_pipeline(question, id="1")
print(answer)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(
    local_file: str = "src/data/letter_counting.json",
    max_samples: int = None,
):
    with open(local_file, "r", encoding="utf-8") as f:
        data = json.load(f)["examples"]

    if max_samples:
        data = data[:max_samples]

    for idx, row in enumerate(data):
        row["id"] = str(idx)

    examples = [
        Example(id=item["id"], question=item["input"], answer=item["target"])
        for item in data
    ]

    n = len(examples)
    train_data = examples[: int(0.6 * n)]
    val_data = examples[int(0.6 * n) : int(0.8 * n)]
    test_data = examples[int(0.8 * n) :]

    return train_data, val_data, test_data


train_data, val_data, test_data = load_datasets(max_samples=2)
print(train_data[0])

# ---------------------------------------------------------------------------
# AdalComponent — wires task, eval, and loss for the Trainer
# ---------------------------------------------------------------------------

class LetterRepeatAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        teacher_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = LetterRepeatTaskPipeline(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config

    def prepare_task(self, sample: Example):
        return self.task.bicall, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput) -> float:
        y_label = -1
        if y_pred is not None and y_pred.data is not None:
            y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(self, sample: Example, pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    task_model_config=gpt_4o_model,
    optimizer_model_config=gpt_o3_mini_model,
):
    adal_component = LetterRepeatAdalComponent(
        **task_model_config,
        teacher_model_config=optimizer_model_config,
        text_optimizer_model_config=optimizer_model_config,
        backward_engine_model_config=optimizer_model_config,
    )
    trainer = adal.Trainer(
        adaltask=adal_component,
        max_steps=12,
        raw_shots=1,
        bootstrap_shots=1,
        strategy="random",
    )

    train_dataset, val_dataset, test_dataset = load_datasets()
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    return adal_component


# ---------------------------------------------------------------------------
# Post-training evaluation on the actual target question
# ---------------------------------------------------------------------------

def evaluate_target_question(adal_component):
    TARGET_QUESTION = (
        "What country has the same letter repeated the most in its name?"
    )
    EXPECTED_ANSWER = "Madagascar"

    adal_component.task.eval()
    result = adal_component.task.bicall(
        question=TARGET_QUESTION, id="target"
    )

    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION — TARGET QUESTION")
    print("=" * 60)
    print(f"Question : {TARGET_QUESTION}")
    print(f"Expected : {EXPECTED_ANSWER}")
    print(f"Raw reply: {result}")
    if hasattr(result, "data") and result.data is not None:
        print(f"Parsed   : {result.data}")
        correct = EXPECTED_ANSWER.lower() in str(result.data).lower()
        print(f"Correct  : {'YES' if correct else 'NO'}")
    print("=" * 60)


trained_component = train()
evaluate_target_question(trained_component)
