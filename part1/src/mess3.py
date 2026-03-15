"""Mess3 Process Hidden Markov Model generator and predictive vector tracker."""

import numpy as np


class Mess3Process:
    """A 3-state Hidden Markov Model process that emits tokens {0,1,2}.

    The transition dynamics and the emitted token probabilities are jointly
    determined by two parameters: alpha and x.

    Each token-labeled transition matrix T^z gives the joint probability:
        T^z[s, s'] = P(token=z, next_state=s' | current_state=s)
    """

    def __init__(self, alpha: float, x: float):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < x < 0.5):
            raise ValueError(f"x must be in (0, 0.5), got {x}")

        self.alpha = alpha
        self.x = x

        beta = (1.0 - alpha) / 2.0
        y = 1.0 - 2.0 * x

        # T0: matrix for observed token 0
        self.T0 = np.array([
            [alpha * y, beta * x,  beta * x],
            [alpha * x, beta * y,  beta * x],
            [alpha * x, beta * x,  beta * y],
        ], dtype=np.float64)

        # T1: matrix for observed token 1
        self.T1 = np.array([
            [beta * y,  alpha * x, beta * x],
            [beta * x,  alpha * y, beta * x],
            [beta * x,  alpha * x, beta * y],
        ], dtype=np.float64)

        # T2: matrix for observed token 2
        self.T2 = np.array([
            [beta * y,  beta * x,  alpha * x],
            [beta * x,  beta * y,  alpha * x],
            [beta * x,  beta * x,  alpha * y],
        ], dtype=np.float64)

        # Total transition matrix
        self.T = self.T0 + self.T1 + self.T2

        # Validate shapes and non-negativity
        for mat, name in [(self.T0, "T0"), (self.T1, "T1"), (self.T2, "T2")]:
            if mat.shape != (3, 3):
                raise ValueError(f"{name} has wrong shape {mat.shape}, expected (3, 3)")
            if (mat < 0).any():
                raise ValueError(f"{name} has negative entries")

        # Validate row sums of T equal 1
        row_sums = self.T.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(f"Rows of total transition matrix do not sum to 1: {row_sums}")

        # Uniform stationary/initial distribution
        self.initial_distribution = np.array([1/3, 1/3, 1/3], dtype=np.float64)

    def get_token_transition_matrix(self, token: int) -> np.ndarray:
        """Return the transition matrix T^(token)."""
        if token == 0:
            return self.T0
        elif token == 1:
            return self.T1
        elif token == 2:
            return self.T2
        else:
            raise ValueError(f"Token must be 0, 1, or 2. Got {token}")

    def sample_initial_hidden_state(self, rng: np.random.Generator) -> int:
        """Sample one hidden state from [0, 1, 2] uniformly."""
        return int(rng.choice(3, p=self.initial_distribution))

    def sample_step(self, hidden_state: int, rng: np.random.Generator) -> tuple[int, int]:
        """Sample exactly one pair of (emitted token, next hidden state)."""
        if hidden_state not in (0, 1, 2):
            raise ValueError(f"Invalid current hidden state: {hidden_state}")

        # Flatten the 9 probabilities P(z, s' | s)
        # Order: z=0(s'=0,1,2), z=1(s'=0,1,2), z=2(s'=0,1,2)
        probs = np.concatenate([
            self.T0[hidden_state],
            self.T1[hidden_state],
            self.T2[hidden_state]
        ])

        # Sample from the 9 outcomes
        outcome_idx = rng.choice(9, p=probs)

        # Decode outcome back to (token, next_state)
        token = int(outcome_idx // 3)
        next_state = int(outcome_idx % 3)

        return token, next_state

    def sample_sequence(self, length: int, rng: np.random.Generator) -> dict:
        """Generate one full sequence of `length` tokens."""
        if length <= 0:
            raise ValueError(f"Sequence length must be positive, got {length}")

        tokens = np.zeros(length, dtype=int)
        hidden_states = np.zeros(length + 1, dtype=int)

        current_state = self.sample_initial_hidden_state(rng)
        hidden_states[0] = current_state

        for t in range(length):
            token, next_state = self.sample_step(current_state, rng)
            tokens[t] = token
            hidden_states[t + 1] = next_state
            current_state = next_state

        return {
            "tokens": tokens,
            "hidden_states": hidden_states,
        }

    def update_predictive_vector(self, eta: np.ndarray, token: int) -> np.ndarray:
        """Compute the next predictive vector after observing `token`.

        eta' = (eta @ T^(token)) / (eta @ T^(token) @ 1)
        """
        if token not in (0, 1, 2):
            raise ValueError(f"Token must be in {{0,1,2}}, got {token}")
        if eta.shape != (3,):
            raise ValueError(f"eta must have shape (3,), got {eta.shape}")
        if not np.allclose(eta.sum(), 1.0, atol=1e-9):
            raise ValueError(f"eta must sum to 1, got {eta.sum()}")

        T_z = self.get_token_transition_matrix(token)
        A = eta @ T_z
        denom = A.sum()

        if denom <= 0:
            raise ValueError(f"Normalization denominator non-positive: {denom}")

        return A / denom

    def compute_predictive_vectors(self, tokens: np.ndarray) -> np.ndarray:
        """Compute the predictive vector sequence for an observed token sequence.

        Returns an array of shape (L+1, 3).
        Row 0 is the uniform initial predictive vector.
        Row t is the vector after observing the first t tokens.
        """
        L = len(tokens)
        etas = np.zeros((L + 1, 3), dtype=np.float64)

        # Initial predictive vector is always uniform
        eta_current = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        etas[0] = eta_current

        for t in range(L):
            eta_current = self.update_predictive_vector(eta_current, tokens[t])
            etas[t + 1] = eta_current

        return etas

def compute_component_likelihood_and_belief(tokens: list[int], alpha: float, x: float) -> tuple[float, np.ndarray]:
    """
    Computes the prefix likelihood under component C and the belief state P(S_t | C, prefix).
    Using exactly the part 5 user formulas.
    """
    process = Mess3Process(alpha, x)
    matrices = {0: process.T0, 1: process.T1, 2: process.T2}
    
    # uniform stationary distribution (row vector)
    row = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    
    for token in tokens:
        T_token = matrices[token]
        row = row @ T_token
        
    likelihood = row.sum()
    if likelihood > 0:
        belief = row / likelihood
    else:
        belief = np.array([1/3, 1/3, 1/3])  # fallback
        
    return likelihood, belief
