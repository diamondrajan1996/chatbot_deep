from typing import List


from deeppavlov import build_model, configs
from deeppavlov.models.augmentation.thesaurus_aug import ThesaurusAug
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import EnThesaurus
from deeppavlov.models.augmentation.utils.inflection import EnInflector
from deeppavlov.models.augmentation.utils.word_filter import EnWordFilter

class QueryExpander(ThesaurusAug):
    """
    class that is received tokenized sentence (example: [dog, is, big]
    and returns set of all synonyms to tokens from input sentence (example: [putty...)
    for english language
    """

    def __init__(self,
                 is_use_morpho_model: bool = True,
                 with_source_token: bool=True,
                 isalpha_only: bool=True,
                 not_replaced_tokens: List[str]=[],
                 replaced_pos_tags: List[str] = ['ADJ', 'ADV', 'NOUN', 'VERB'],
                 en_classical_pluralize: bool = True):
        """"""
        self.thesaurus = EnThesaurus(with_source_token=with_source_token)
        self.word_filter = EnWordFilter(replace_freq=1,
                                        isalpha_only=isalpha_only,
                                        not_replaced_tokens=not_replaced_tokens,
                                        replaced_pos_tags=replaced_pos_tags)
        self.is_use_morpho_model = is_use_morpho_model
        if is_use_morpho_model:
            self.inflector = EnInflector(classical_pluralize=en_classical_pluralize)
            self.morpho_tagger = build_model(configs.morpho_tagger.UD2_0.morpho_en, download=True)

    def _infer_instance_with_morpho_tagger(self,
                                           tokens: List[str],
                                           morpho_tags: str,
                                           return_lemmas: bool=True,
                                           return_inflected: bool=True):
        assert return_inflected or return_inflected, """at least one of params:
                                                        return_inflected or return_lemmas should be true"""
        res = set()
        morpho_tags = self._transform_morpho_tags_in_dict(morpho_tags)
        filter_res = self.word_filter.filter_words(tokens, morpho_tags)
        lemmas = self._get_lemmas(tokens, morpho_tags, filter_res)
        lemma_synonyms = self._get_synonyms(lemmas, morpho_tags, filter_res)
        if return_inflected:
            inflected_syns = self._inflect_synonyms(lemma_synonyms, morpho_tags, filter_res)
            inflected_syns = self._filter_none_value(inflected_syns)
            for syn_to_token in inflected_syns:
                if syn_to_token:
                    res |= set(syn_to_token)
        if return_lemmas:
            for syn_to_token in lemma_synonyms:
                if syn_to_token:
                    res |= set(syn_to_token)
        return res

    def _infer_minibatch_with_morpho_tagger(self,
                                            minibatch: List[List[str]],
                                            return_lemmas: bool=True,
                                            return_inflected: bool=True):
        batch_morpho_tags = self.morpho_tagger(minibatch)
        res = [self._infer_instance_with_morpho_tagger(tokens, morpho_tags, return_lemmas, return_inflected)
               for tokens, morpho_tags in zip(minibatch, batch_morpho_tags)]
        return res

    def _infer_instance(self, tokens: List[str]):
        res = set()
        filter_res = self.word_filter.filter_isalpha_only(tokens)
        synonyms = self._get_synonyms(tokens, [None]*len(tokens), filter_res)
        for syn_to_token in synonyms:
            if syn_to_token:
                res |= set(syn_to_token)
        return res

    def _infer_minibatch(self, batch: List[List[str]]):
        return [self._infer_instance(tokens) for tokens in batch]

    def __call__(self,
                 minibatch: List[List[str]],
                 return_lemmas: bool=True,
                 return_inflected: bool=True):
        if self.is_use_morpho_model:
            return self._infer_minibatch_with_morpho_tagger(minibatch, return_lemmas, return_inflected)
        else:
            return self._infer_minibatch(minibatch)











