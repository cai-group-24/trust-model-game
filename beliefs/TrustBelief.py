from dataclasses import dataclass


@dataclass
class TrustBelief:
    """
    Class to represent a trust belief in humans
    """
    # We ignore competence and willingness for now
    competence: float
    willingness: float
    # Probability distribution over labels
    very_untrusted: float
    untrusted: float
    neutral: float
    trusted: float
    very_trusted: float
