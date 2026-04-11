def compute_accuracy(pred, true):
    return (pred == true).mean()