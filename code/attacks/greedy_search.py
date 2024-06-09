## Adapted from textattack

"""
Beam Search
===============
"""
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class BeamSearch(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8):
        self.beam_width = beam_width

    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED: ## While it doesn't succeed
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # print(f'No Potential Beam')
                # If we did not find any possible perturbations, give up.
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam) ## search-over: queries exhausted

            new_results = []
            for result in results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    new_results.append(result)

            if len(new_results) == 0:
                scores_ = np.array([r.score for r in results])
                failed_best_result = results[scores_.argmax()]
                return failed_best_result

            results = new_results

            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()] ## result corresponding to best_result
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]


"""
Greedy Search
=================
"""

class GreedySearch(BeamSearch):
    """A search method that greedily chooses from a list of possible
    perturbations.

    Implemented by calling ``BeamSearch`` with beam_width set to 1.
    """

    def __init__(self):
        super().__init__(beam_width=1)

    def extra_repr_keys(self):
        return []