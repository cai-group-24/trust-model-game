import numpy as np
import random


class TrustBelief:
    """
    Class to represent a trust belief in humans.
    """
    # We ignore competence and willingness for now
    competence: float
    willingness: float
    competence_alpha: float
    willingness_alpha: float

    def __init__(self, competence: float, willingness: float):
        self.competence = competence
        self.willingness = willingness
        self.competence_alpha = 1
        self.willingness_alpha = 1

    def increment_willingness(self, x: float):
        """
        Increment the willingness by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.willingness = np.clip([self.willingness + x * self.willingness_alpha], -1, 1)[0]
        self.willingness_alpha = self.willingness_alpha - self.willingness_alpha * 0.15

    def increment_competence(self, x: float):
        """
        Increment the competence by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.competence = np.clip([self.competence + x * self.competence_alpha],  -1, 1)[0]
        self.competence_alpha = self.competence_alpha - self.competence_alpha * 0.15

    def increment_trust(self, x: float):
        """
        Increment trust by increasing willingness and trust by x.
        """
        self.increment_competence(x)
        self.increment_willingness(x)

    def decrement_willingness(self, x: float):
        """
        Decrement the willingness by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.increment_willingness(-x)

    def decrement_competence(self, x: float):
        """
        Decrement the willingness by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.increment_competence(-x)

    def decrement_trust(self, x: float):
        """
        Decrement trust by increasing willingness and trust by x.
        """
        self.decrement_competence(x)
        self.decrement_willingness(x)

    def reset_competence_alpha(self):
        """
        Reset the competence alpha value to 1.
        """
        self.competence_alpha = 1

    def reset_willingness_alpha(self):
        """
        Reset the willingness alpha value to 1.
        """
        self.willingness_alpha = 1

    def should_trust(self, min_comp, min_will):
        """
        Given some competence and willingness thresholds,
        decide if self should be trusted or not.
        For activities which require no competence, just put -1 as max_comp.
        """
        if self.competence < min_comp or self.willingness < min_will:
            return False
        # Define randomness to return true or false
        return self.trust_formula()

    def trust_formula(self):
        """
        Decide whether to trust or not by combining competence and willingness using a formula.
        """
        # TODO find proper weights based on literature
        willingness_weight = 0.4
        competence_weight = 0.6
        competence_willingness_sum = self.competence*competence_weight + self.willingness*willingness_weight

        random_threshold = random.uniform(0, 1)

        return competence_willingness_sum > random_threshold
