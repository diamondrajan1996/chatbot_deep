from typing import List


from deeppavlov import build_model, configs
from deeppavlov.models.augmentation.thesaurus_aug import ThesaurusAug
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import EnThesaurus
from deeppavlov.models.augmentation.utils.inflection import EnInflector
from deeppavlov.models.augmentation.utils.word_filter import EnWordFilter

class QueryExpander(ThesaurusAug):
    """Query Expander for ODQA task

    it receives tokenized sentence (example: ["dog", "is", "big"])
    and returns set of all synonyms to tokens from input sentence (example: [putty...)
    for english language. Based on WordNet.
    Args:
        is_use_morpho_model: flag that says use morphological model or not.
            Morphological model allows to get lemma of source tokens, and inflect synonyms.
        with_source_token: flag that says result should include source tokens or not
        isalpha_only: flag that says try to search synonyms only for "isalpha" tokens or not
        not_replaced_tokens: list of tokens, no synonyms will be searched for token in this list
        not_replaced_pos_tag: list of pos tags, no synonyms will be searched for words with these pos tags,
            only with morphological model
    Attributes:
        thesaurus: wrap on the wordnet thesaurus
        word_filter: object that decides search synonyms for this token or not
        is_use_morpho_model: flag that says use morphological model or not.
            Morphological model allows to get lemma of source tokens, and inflect synonyms.
    """

    def __init__(self,
                 is_use_morpho_model: bool = True,
                 with_source_token: bool = True,
                 isalpha_only: bool = True,
                 not_replaced_tokens: List[str] = [],
                 replaced_pos_tags: List[str] = ['ADJ', 'ADV', 'NOUN', 'VERB']):
        self.thesaurus = EnThesaurus(with_source_token=with_source_token)
        self.word_filter = EnWordFilter(replace_freq=1,
                                        isalpha_only=isalpha_only,
                                        not_replaced_tokens=not_replaced_tokens,
                                        replaced_pos_tags=replaced_pos_tags)
        self.is_use_morpho_model = is_use_morpho_model
        if is_use_morpho_model:
            self.inflector = EnInflector(classical_pluralize=True)
            self.morpho_tagger = build_model(configs.morpho_tagger.UD2_0.morpho_en, download=True)

    def _infer_instance_with_morpho_tagger(self,
                                           tokens: List[str],
                                           morpho_tags: str,
                                           return_lemmas: bool=True,
                                           return_inflected: bool=True):
        assert return_inflected or return_inflected, """at least one of params:
                                                        return_inflected or return_lemmas should be true"""
        morpho_tags = self._transform_morpho_tags_in_dict(morpho_tags)
        filter_res = self.word_filter.filter_words(tokens, morpho_tags)
        #not_replaced_filter_res = self.word_filter.filter_not_replaced_token(tokens, morpho_tags)
        #isalpha_filter_res = self.word_filter.filter_isalpha_only(tokens)
        #filter_res = [all(i) for i in zip(isalpha_filter_res, not_replaced_filter_res)]
        lemmas = self._get_lemmas(tokens, morpho_tags, filter_res)
        lemma_synonyms = self._get_synonyms(lemmas, morpho_tags, filter_res)
        result = set()
        if return_inflected:
            inflected_syns = self._inflect_synonyms(lemma_synonyms, morpho_tags, filter_res)
            inflected_syns = self._filter_none_value(inflected_syns)
            for syn_to_token in inflected_syns:
                if syn_to_token:
                    result |= set(syn_to_token)
        if return_lemmas:
            for syn_to_token in lemma_synonyms:
                if syn_to_token:
                    result |= set(syn_to_token)
        return result

    def _infer_minibatch_with_morpho_tagger(self,
                                            minibatch: List[List[str]],
                                            return_lemmas: bool=True,
                                            return_inflected: bool=True):
        """
        It expands query using morphological model
        Args:
            minibatch: minibatch of tokenized text
            return_lemmas: flag that says return synonyms in lemma form
            return_inflected: flag that says return synonyms in inflected form (the same as form of source token)
        Returns:
            minibatch of sets with synonyms to all tokens in sample
        """
        batch_morpho_tags = self.morpho_tagger(minibatch)
        result = [self._infer_instance_with_morpho_tagger(tokens, morpho_tags, return_lemmas, return_inflected)
                  for tokens, morpho_tags in zip(minibatch, batch_morpho_tags)]
        return result

    def _infer_instance(self, tokens: List[str]):
        result = set()
        filter_res = self.word_filter.filter_isalpha_only(tokens)
        synonyms = self._get_synonyms(tokens, [None]*len(tokens), filter_res)
        for syn_to_token in synonyms:
            if syn_to_token:
                result |= set(syn_to_token)
        return result

    def _infer_minibatch(self, batch: List[List[str]]):
        """
        It expands query without using morphological model
        Args:
            minibatch: minibatch of tokenized text
            return_lemmas: flag that says return synonyms in lemma form
            return_inflected: flag that says return synonyms in inflected form (the same as form of source token)
        Returns:
            minibatch of sets with synonyms to all tokens in sample
        """
        return [self._infer_instance(tokens) for tokens in batch]

    def _disunit_tokens(self, tokens: List[str]):
        return [" ".join(i.split('_')) for i in tokens]

    def __call__(self,
                 minibatch: List[List[str]],
                 return_lemmas: bool=True,
                 return_inflected: bool=True):
        """
        It expands query. Use morphological model or not indicated in 'is_use_morpho_model' attribute.
        Args:
            minibatch: minibatch of tokenized text
            return_lemmas: flag that says return synonyms in lemma form
            return_inflected: flag that says return synonyms in inflected form (the same as form of source token)
        Returns:
            minibatch of sets with synonyms to all tokens in sample
        """
        if self.is_use_morpho_model:
            result = self._infer_minibatch_with_morpho_tagger(minibatch, return_lemmas, return_inflected)
        else:
            result = self._infer_minibatch(minibatch)
        return [self._disunit_tokens(tokens) for tokens in result]












