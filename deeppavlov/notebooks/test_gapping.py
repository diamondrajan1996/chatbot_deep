from deeppavlov.dataset_readers.gapping_dataset_reader import read_gapping_file, GappingDatasetReader
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer
from deeppavlov.models.preprocessors.bert_preprocessor import BertNerPreprocessor
from deeppavlov.models.preprocessors.mask import Mask
from deeppavlov.models.gapping.utils import CharToWordIndexer, VerbSelector, GappingSourcePreprocessor
from deeppavlov.models.gapping.network import *


def gapping_metric(y_true, y_pred):
    tp_sents, tn_sents, fn_sents, fp_sents = 0, 0, 0, 0
    tp_words, fp_words, fn_words = 0, 0, 0
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
        for gap in true_gaps:
            if gap in pred_gaps:
                tp_words += 1
            else:
                fn_words += 1
        for gap in pred_gaps:
            if gap not in true_gaps:
                fp_words += 1
    sent_f1 = 2*tp_sents / (2*tp_sents + fp_sents + fn_sents) if tn_sents < len(y_true) else 1.0
    gap_f1 = 2*tp_words / (2*tp_words + fp_words + fn_words) if tn_sents < len(y_true) else 1.0
    sent_accuracy = (tp_sents + tn_sents) / (tn_sents + tp_sents + fp_sents + fn_sents)
    return {"sent_f1": 100 * sent_f1, "gap_f1": 100 * gap_f1, "sent_acc": 100 * sent_accuracy}


BERT_VOCAB_PATH = "/home/alexeysorokin/data/SberChallenge/data/my_models/rubert_cased_v1/vocab.txt"
BERT_CONFIG_FILE = "/home/alexeysorokin/data/SberChallenge/data/my_models/rubert_cased_v1/bert_config.json"
BERT_MODEL_FILE = "/home/alexeysorokin/data/SberChallenge/data/my_models/rubert_cased_v1/bert_model.ckpt"

tokenizer, char_to_word_indexer = LazyTokenizer(), CharToWordIndexer()
reader = GappingDatasetReader()
data = reader.read(data_path=["/home/alexeysorokin/data/Other/AGRR-2019/dev.csv"], data_types=["train"])
iterator = DataLearningIterator(data, shuffle=False)
verb_selector = VerbSelector("configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json")
gapping_source_transformer = GappingSourcePreprocessor()
bert_preprocessor = BertNerPreprocessor(BERT_VOCAB_PATH, max_subword_length=15,
                                        subword_mask_mode="last")
masker = Mask()
recognizer = BertGappingRecognizer(keep_prob=0.1, bert_config_file=BERT_CONFIG_FILE,
                                   pretrained_bert=BERT_MODEL_FILE,
                                   learning_rate=1e-3,
                                   bert_learning_rate=2e-5,
                                   min_learning_rate=1e-7,
                                   use_birnn=True,
                                   optimizer="tf.train.AdamOptimizer",
                                   save_path="../experiments/bert_gapping",
                                   load_path="../experiments/bert_gapping")
for j, elem in enumerate(iterator.gen_batches(batch_size=8), 1):
    if j % 10 == 0:
        print("{} batches generated".format(j))
    if j == 500:
        break
    texts, labels, data = elem
    tokenized_texts = tokenizer(texts)
    word_indexes = char_to_word_indexer(texts, tokenized_texts, data)
    verb_indexes = verb_selector(tokenized_texts)
    verb_gap_matrixes, verb_gap_indexes = gapping_source_transformer(tokenized_texts, word_indexes, verb_indexes)
    bert_words, bert_subtokens, bert_indexes, bert_mask = bert_preprocessor(tokenized_texts)
    bert_subtokens_mask = masker(bert_subtokens)
    # for sent, label, curr_indexes, curr_verb_indexes, curr_matrix in\
    #         zip(tokenized_texts, labels, word_indexes, verb_indexes, verb_gap_indexes):
    #     print(sent)
    #     print(label, curr_indexes, curr_verb_indexes, *curr_matrix, sep="\t")
    #     # print(curr_matrix.shape, curr_matrix.nonzero())
    # break
    for i in range(1):
        batch_output = recognizer.train_on_batch(bert_indexes, bert_subtokens_mask, bert_mask,
                                                 verb_indexes, verb_gap_matrixes)
        all_verb_gap_matrixes = np.concatenate(verb_gap_matrixes, axis=0)
        print(j, np.sum(all_verb_gap_matrixes), "{:.2f}".format(batch_output["loss"]), end=" ")
        batch_predictions = recognizer(bert_indexes, bert_subtokens_mask,
                                       bert_mask, verb_indexes)
        for r, (verbs, gaps) in enumerate(batch_predictions):
            if len(gaps) > 0:
                print(r, *verbs, sep=":", end="-")
                print(*gaps, sep=",", end=" ")
        print("")
    for r, elem in enumerate(verb_gap_indexes):
        if len(elem[1]) > 0:
            print(r, *elem[0], sep=":", end="-")
            print(*elem[1], sep=",", end=" ")
    print("")
    print(gapping_metric(verb_gap_indexes, batch_predictions))