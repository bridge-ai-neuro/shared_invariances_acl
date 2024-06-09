# Adapted from textattack

"""
Robustness Capability ~ While being label-preserving
=================================================================
"""

import random

from textattack.attack import Attack
from textattack.attack_recipes import AttackRecipe

from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import WordSwap

from textattack.attack_recipes import AttackRecipe
from textattack.goal_functions.classification import ClassificationGoalFunction

from attacks.greedy_search import GreedySearch
from attacks.label_preserve_clf import LabelPreserveClassification

import numpy as np
import torch.nn as nn

class SwapMiddleChars(WordSwap):
    """ Transforms an input by swapping middle-characters of a word.
    """


    def _get_replacement_words(self, word):
        """ 
        Swaps middle-character of a word if possible.
        """
        idexes = [idx for idx in range(1, len(word)-1)]
        idx1, idx2 = random.sample(idexes, k=2) ## sample two indicies without rep.
        pert_count = 0

        while word[idx1] == word[idx2] and pert_count<10:
            idx1, idx2 = random.sample(idexes, k=2) ## without rep
            pert_count+=1
        
        word_list = list(word)
        word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]

        pert_word = ''.join(w for w in word_list)
        return [pert_word]
    

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for _ in range(15):
            pert_text = current_text
            for i in indices_to_modify:
                word_to_replace = words[i]
                replacement_word = self._get_replacement_words(word_to_replace)[0]

                if replacement_word == word_to_replace:
                    continue

                pert_text = pert_text.replace_word_at_index(i, replacement_word)
            transformed_texts.append(pert_text)
        return transformed_texts




class InvPerturbRobustness(AttackRecipe):
    """An implementation of the attack used in "Combating Adversarial
    Misspellings with Robust Word Recognition", Pruthi et al., 2019.

    This attack focuses on a small number of character-level changes that simulate common typos. It combines:
        - Swapping neighboring characters
        - Deleting characters
        - Inserting characters
        - Swapping characters for adjacent keys on a QWERTY keyboard.

    https://arxiv.org/abs/1905.11268

    :param model: Model to attack.
    :param max_num_word_swaps: Maximum number of modifications to allow.
    """


    @staticmethod
    def build(model_wrapper):
        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper

        transformation = SwapMiddleChars()

        # only edit words of length >= 4, edit max_num_word_swaps words.
        # note that we also are not editing the same word twice, so
        # max_num_word_swaps is really the max number of character
        # changes that can be made. The paper looks at 1 and 2 char attacks.
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            RepeatModification(),
        ]
        # untargeted attack
        goal_function = LabelPreserveClassification(model_wrapper)

        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)