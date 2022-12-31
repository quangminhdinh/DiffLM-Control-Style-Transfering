from .config import train_set
import numpy as np
import evaluate


def compute_metrics(eval_preds):
    metric = evaluate.load('rouge')
    logits, labels = eval_preds
    # print(len(logits), len(logits[0]), len(logits[0][0]), len(logits[0][0][0]))
    # print(len(labels), len(labels[0]), len(labels[0][0]))
    # return
    predictions = np.argmax(logits[0], axis=-1)
    preds = [train_set.tokenizer.decode(s) for s in predictions]
    refs = [ train_set.tokenizer.decode(s) for s in labels]
    # lab = np.argmax(labels, axis=-1)
    return metric.compute(predictions=preds, references=refs)
