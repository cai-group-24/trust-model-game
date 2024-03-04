import numpy as np
import random


class TrustBelief:
    """
    Class to represent a trust belief in humans.
    """
    # We ignore competence and willingness for now
    competence: float
    willingness: float

    def __init__(self, competence: float, willingness: float):
        self.competence = competence
        self.willingness = willingness

    def increment_willingness(self, x: float):
        """
        Increment the willingness by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.willingness = np.clip([self.willingness + x], -1, 1)[0]

    def increment_competence(self, x: float):
        """
        Increment the competence by a factor x, correct by alpha and clip to [-1, 1].
        """
        self.competence = np.clip([self.competence + x],  -1, 1)[0]

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
        Decide whether to trust or not by combining competence and willingness using a weighted sum and random threshold.
        """
        # TODO find proper weights based on literature
        willingness_weight = 0.4
        competence_weight = 0.6
        competence_willingness_sum = self.competence*competence_weight + self.willingness*willingness_weight

        random_threshold = random.uniform(0, 1)

        return competence_willingness_sum >= random_threshold
