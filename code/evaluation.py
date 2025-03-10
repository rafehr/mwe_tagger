from seqeval.metrics import f1_score, accuracy_score, classification_report

def evaluate(gold_labels, logits):
    return accuracy_score(gold_labels, logits)