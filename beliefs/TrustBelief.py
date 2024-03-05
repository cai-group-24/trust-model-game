import numpy as np
import random
import enum

class TrustMechanism(enum.Enum):
    NEVER_TRUST = 1,
    ALWAYS_TRUST = 2,
    RANDOM_TRUST = 3,
    CUSTOM_TRUST = 4

class TrustBelief:
    """
    Class to represent a trust belief in humans.
    """
    # We ignore competence and willingness for now
    competence: float
    willingness: float

    def __init__(self, competence: float, willingness: float, trust_mechanism: TrustMechanism):
        self.competence = competence
        self.willingness = willingness
        self.trust_mechanism = trust_mechanism

    def increment_willingness(self, x: float):
        """
        Increment the willingness by a factor x, correct by alpha and clip to [-1, 1].
        """
        if self.trust_mechanism == TrustMechanism.CUSTOM_TRUST:
            self.willingness = np.clip([self.willingness + x], -1, 1)[0]

    def increment_competence(self, x: float):
        """
        Increment the competence by a factor x, correct by alpha and clip to [-1, 1].
        """
        if self.trust_mechanism == TrustMechanism.CUSTOM_TRUST:
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

    def should_trust(self, min_comp: float, min_will: float, comp_weight=0.3, will_weight=0.7):
        """
        Given some competence and willingness thresholds,
        decide if self should be trusted or not.
        For activities which require no competence, just put -1 as max_comp.
        """
        if self.competence < min_comp or self.willingness < min_will:
            return False
        # Define randomness to return true or false
        return self.trust_formula(comp_weight, will_weight)

    def trust_formula(self, comp_weight: float, will_weight: float):
        """
        Decide whether to trust or not by combining competence and willingness using a weighted sum and random threshold.
        """
        competence_willingness_sum = self.competence*comp_weight + self.willingness*will_weight

        random_threshold = random.uniform(-1, 1)

        return competence_willingness_sum >= random_threshold
