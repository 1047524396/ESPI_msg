def calc_metrics(preds, gt):
    """
    Return acc, precision, recall, f1
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, label in zip(preds, gt):
        if pred == label:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp) if tp + fp != 0 else float('nan') 
    rec = tp / (tp + fn) if tp + fn != 0 else float('nan')
    f1 = 2 * pre * rec / (pre + rec)

    return acc, pre, rec, f1
