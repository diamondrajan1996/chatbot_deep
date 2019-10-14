from collections import Counter
import itertools
from sklearn.metrics import precision_recall_fscore_support

from deeppavlov.core.common.metrics_registry import register_metric

MIN_LABEL_COUNT = 10

@register_metric('sequence_per_class_f1')
def sequence_per_class_f1(y_true, y_predicted):
    y_true = list(itertools.chain(*y_true))
    y_predicted = list(itertools.chain(*y_predicted))
    label_counts = Counter(y_true)
    labels = [x for x, count in label_counts.items() if count >= MIN_LABEL_COUNT]
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_predicted, labels=labels)
    return "\n" + str({label: "{:.4f}".format(value) for label, value in zip(labels, f1)}) + "\n"