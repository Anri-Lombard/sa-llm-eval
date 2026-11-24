import ast

from lm_eval.utils import weighted_f1_score  # noqa: F401


def afrimmlu_choices(doc):
    return ast.literal_eval(doc["choices"])


def afrimmlu_target(doc):
    return "ABCD".index(doc["answer"])


def afrimmlu_prompt(doc):
    choices = afrimmlu_choices(doc)
    return (
        f"Answer this multiple-choice question about {doc['subject']}.\n"
        f"Question: {doc['question']}\n"
        f"A: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\n"
        "Answer:"
    )


def xnli_target_tfn(doc):
    return {0: "True", 1: "Neither", 2: "False"}[doc["label"]]


def xnli_target_enc(doc):
    return {0: "entailment", 1: "neutral", 2: "contradiction"}[doc["label"]]


def xnli_target_tif(doc):
    return {0: "true", 1: "inconclusive", 2: "false"}[doc["label"]]
