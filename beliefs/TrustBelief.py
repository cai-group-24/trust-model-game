import numpy as np
import random
import enum


class TrustMechanism(enum.Enum):
    NEVER_TRUST = 1,
    ALWAYS_TRUST = 2,
    RANDOM_TRUST = 3,
    CUSTOM_TRUST = 4


# Amount of decimals we use for competence and willingness rounding (cleaner memory files)
TRUST_DECIMALS = 5


class TrustBelief:
    """
    Class to represent a trust belief in humans.
    """
    # We ignore competence and willingness for now
    competence: float
    willingness: float
    trust_mechanism: TrustMechanism
    # Represents the amount of ticks that have been played with this character
    # Used in confidence calculations for trust updating
    ticks_played: int

    def __init__(self, competence: float, willingness: float, trust_mechanism: TrustMechanism, ticks_played: int):
        self.competence = competence
        self.willingness = willingness
        self.trust_mechanism = trust_mechanism
        self.ticks_played = ticks_played

    def clip(self, x: float) -> float:
        """
        Clip value between -1 and 1 and round decimals to certain amount
        """
        return np.round(max(-1.00, min(1.00, x)), decimals=TRUST_DECIMALS)

    def increment_willingness(self, x: float):
        """
        Increment the willingness by a factor x, correct by alpha and clip to [-1, 1].
        Also takes confidence into account.
        """
        if self.trust_mechanism == TrustMechanism.CUSTOM_TRUST:
            self.willingness = self.clip(self.willingness + self.trust_difference_with_confidence(x))

    def increment_competence(self, x: float):
        """
        Increment the competence by a factor x, correct by alpha and clip to [-1, 1].
        Also takes confidence into account.
        """
        if self.trust_mechanism == TrustMechanism.CUSTOM_TRUST:
            self.competence = self.clip(self.competence + self.trust_difference_with_confidence(x))

    def increment_trust(self, x: float):
        """
        Increment trust by increasing willingness and trust by x.
        Also takes confidence into account.
        """
        self.increment_competence(x)
        self.increment_willingness(x)

    def decrement_willingness(self, x: float):
        """
        Decrement the willingness by a factor x, correct by alpha and clip to [-1, 1].
        Also takes confidence into account.
        """
        self.increment_willingness(-x)

    def decrement_competence(self, x: float):
        """
        Decrement the willingness by a factor x, correct by alpha and clip to [-1, 1].
        Also takes confidence into account.
        """
        self.increment_competence(-x)

    def decrement_trust(self, x: float):
        """
        Decrement trust by increasing willingness and trust by x.
        Also takes confidence into account.
        """
        self.decrement_competence(x)
        self.decrement_willingness(x)

    def should_trust(self, min_comp: float, min_will: float, comp_weight=0.3, will_weight=0.7):
        """
        Given some competence and willingness thresholds,
        decide if self should be trusted or not.
        For activities which require no competence, just put -1 as max_comp.
        """
        competence_willingness_sum = self.competence * comp_weight + self.willingness * will_weight
        return competence_willingness_sum >= (min_comp * comp_weight) + (min_will * will_weight)

    def trust_difference_with_confidence(self, diff: float):
        """
        Calculate the increment/decrement in confidence/willingness while keeping confidence in mind.
        The higher the confidence, the slighter the increase/decrease.
        """
        confidence_score = self.calculate_confidence()
        multiplier = 1 - confidence_score
        return diff * multiplier

    def calculate_confidence(self) -> float:
        """
        Calculate the confidence (number between 0-1) based on the amount of ticks that have been played with this human.
        The longer a robot plays with a human, the more confident the robot is in the trust it has in that human.
        """
        seconds = self.ticks_played / 10.0
        # We assume 100% confidence after 40 minutes of playing
        MAX_CONFIDENCE_SECONDS = 40 * 60

        # Use function which starts around 0.05 and approaches 1 near the max confidence threshold
        # Function was found by experimenting in a graphing calculator (https://www.desmos.com/calculator)
        def confidence_func(x: float):
            return -np.exp([-3.0 / MAX_CONFIDENCE_SECONDS * x])[0] + 1.05

        # Clip formula output between 0 and 0.9 (max 90% confidence)
        # divide by 2 for a slower start
        confidence_score = confidence_func(seconds) / 2
        return min(0.9, max(0.05, confidence_score))
