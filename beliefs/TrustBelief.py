import numpy as np


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
