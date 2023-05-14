def batch_accuracy(logits, labels):
    preds = logits.argmax(-1)
    return (preds == labels).mean()
