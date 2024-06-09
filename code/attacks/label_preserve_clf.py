## Adapted from textattack

import numpy as np
import torch.nn as nn

from textattack.goal_functions.classification import ClassificationGoalFunction

class LabelPreserveClassification(ClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(self, *args, target_max_score=None, **kwargs):
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, atk_text):
        return (model_output.argmax() == self.ground_truth_output and  atk_text.text not in self.prev_samples[self.initial_attacked_text.text])
        
    def _get_score(self, model_output, attacked_text):

        og_softmax = nn.functional.softmax(self.model(self.initial_attacked_text.text).view(1,-1), dim=1).detach().cpu().numpy()
        pert_softmax = nn.functional.softmax(model_output.view(1,-1), dim=1).detach().cpu().numpy()

        diff = np.sum(np.abs(og_softmax - pert_softmax), axis=1)

        if attacked_text.text == self.initial_attacked_text.text:
            score = 0
        else:
            score = 1/(float(diff)+1) ## inversing the difference as search_method is optimized to select transformations with highest score (i.e., lowest difference)
        
        return score