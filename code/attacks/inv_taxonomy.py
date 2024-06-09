## Adapted from textattack

from textattack.attack import Attack
from textattack.attack_recipes import AttackRecipe

from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import WordSwapWordNet
from textattack.goal_functions.classification import ClassificationGoalFunction

import numpy as np
import torch.nn as nn

from attacks.greedy_wir import GreedyWordSwapWIR
from attacks.label_preserve_clf import LabelPreserveClassification


class InvPerturbTaxonomy(AttackRecipe):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Langauge Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Words are prioritized for a synonym-swap transformation based on
    a combination of their saliency score and maximum word-swap effectiveness.
    Note that this implementation does not include the Named
    Entity adversarial swap from the original paper, because it requires
    access to the full dataset and ground truth labels in advance.

    https://www.aclweb.org/anthology/P19-1103/
    """


    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = LabelPreserveClassification(model_wrapper)
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)