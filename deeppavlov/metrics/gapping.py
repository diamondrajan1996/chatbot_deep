from deeppavlov.core.common.metrics_registry import register_metric

@register_metric('gapping_sentence_f1')
def gapping_sentence_f1(y_true, y_pred):
    tp_sents, tn_sents, fn_sents, fp_sents = 0, 0, 0, 0
    for (true_verbs, true_gaps), (pred_verbs, pred_gaps) in zip(y_true, y_pred):
        if len(true_verbs) > 0:
            if len(pred_verbs) > 0:
                tp_sents += 1
            else:
                fn_sents += 1
        elif len(pred_verbs) > 0:
            fp_sents += 1
        else:
            tn_sents += 1
    sent_f1 = 2*tp_sents / (2*tp_sents + fp_sents + fn_sents) if tn_sents < len(y_true) else 1.0
    return sent_f1

@register_metric('gapping_position_f1')
def gapping_position_f1(y_true, y_pred):
    tp_words, fp_words, fn_words = 0, 0, 0
    has_positive_sents = False
    for elem in zip(y_true, y_pred):
        print(elem[0], elem[1])
        break
    for (true_verbs, true_gaps), (pred_verbs, pred_gaps) in zip(y_true, y_pred):
        has_positive_sents |= (len(true_gaps) > 0)
        for gap in true_gaps:
            if gap in pred_gaps:
                tp_words += 1
            else:
                fn_words += 1
        for gap in pred_gaps:
            if gap not in true_gaps:
                fp_words += 1
    gap_f1 = 2 * tp_words / (2 * tp_words + fp_words + fn_words) if has_positive_sents else 1.0
    return gap_f1