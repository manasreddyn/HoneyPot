"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEXUS-GUARDIAN SYSTEM v4.0 - PART 2                       ║
║                     Continuation of Core Neural Layers                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: BIDIRECTIONAL LSTM WITH ATTENTION (LAYER 3)
# ═══════════════════════════════════════════════════════════════════════════

class BidirectionalLSTMAnalyzer:
    """
    Bidirectional LSTM for sequence analysis with attention mechanism
    
    Mathematical Model:
    Forward LSTM:  h_t→ = LSTM(x_t, h_{t-1}→)
    Backward LSTM: h_t← = LSTM(x_t, h_{t+1}←)
    Combined:      h_t = [h_t→ || h_t←]
    Attention:     α_t = softmax(v^T tanh(W_h h_t + W_x x_t))
    Context:       c = Σ α_t h_t
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Forward LSTM parameters
        self.Wf_fwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wi_fwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wc_fwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wo_fwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        
        # Backward LSTM parameters
        self.Wf_bwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wi_bwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wc_bwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        self.Wo_bwd = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.02
        
        # Attention parameters
        self.W_attn = np.random.randn(hidden_dim * 2, hidden_dim * 2) * 0.02
        self.v_attn = np.random.randn(hidden_dim * 2, 1) * 0.02
        
        # Classification head
        self.W_clf = np.random.randn(hidden_dim * 2, 1) * 0.02
        
    def _lstm_cell(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                   Wf, Wi, Wc, Wo) -> Tuple[np.ndarray, np.ndarray]:
        """Single LSTM cell computation"""
        # Concatenate input and hidden state
        combined = np.concatenate([x, h_prev])
        
        # Gates
        f = self._sigmoid(Wf @ combined)  # Forget gate
        i = self._sigmoid(Wi @ combined)  # Input gate
        c_tilde = np.tanh(Wc @ combined)  # Candidate cell state
        o = self._sigmoid(Wo @ combined)  # Output gate
        
        # Update cell state
        c = f * c_prev + i * c_tilde
        
        # Compute hidden state
        h = o * np.tanh(c)
        
        return h, c
    
    def forward_pass(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward LSTM pass"""
        seq_len = len(sequence)
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        hidden_states = []
        
        for t in range(seq_len):
            h, c = self._lstm_cell(sequence[t], h, c,
                                   self.Wf_fwd, self.Wi_fwd, self.Wc_fwd, self.Wo_fwd)
            hidden_states.append(h)
        
        return np.array(hidden_states), h
    
    def backward_pass(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backward LSTM pass"""
        seq_len = len(sequence)
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        hidden_states = []
        
        for t in range(seq_len - 1, -1, -1):
            h, c = self._lstm_cell(sequence[t], h, c,
                                   self.Wf_bwd, self.Wi_bwd, self.Wc_bwd, self.Wo_bwd)
            hidden_states.append(h)
        
        # Reverse to maintain temporal order
        return np.array(hidden_states[::-1]), h
    
    def attention(self, hidden_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights and context vector
        
        α_t = softmax(v^T tanh(W h_t))
        c = Σ α_t h_t
        """
        # Compute attention scores
        scores = []
        for h_t in hidden_states:
            score = self.v_attn.T @ np.tanh(self.W_attn @ h_t)
            scores.append(float(score))
        
        # Softmax
        scores = np.array(scores)
        attention_weights = np.exp(scores - np.max(scores))
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Compute context vector
        context = np.zeros(hidden_states.shape[1])
        for weight, h_t in zip(attention_weights, hidden_states):
            context += weight * h_t
        
        return context, attention_weights
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text using bidirectional LSTM"""
        # Convert text to sequence (character-level for simplicity)
        sequence = self._text_to_sequence(text)
        
        if len(sequence) == 0:
            return {'lstm_fraud_score': 0.5}
        
        # Forward pass
        h_fwd, final_fwd = self.forward_pass(sequence)
        
        # Backward pass
        h_bwd, final_bwd = self.backward_pass(sequence)
        
        # Concatenate bidirectional hidden states
        h_combined = np.concatenate([h_fwd, h_bwd], axis=1)
        
        # Apply attention
        context, attention_weights = self.attention(h_combined)
        
        # Classify
        logit = self.W_clf.T @ context
        fraud_prob = self._sigmoid(logit)
        
        return {
            'lstm_fraud_score': float(fraud_prob),
            'lstm_attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-10))),
            'lstm_attention_max': float(np.max(attention_weights)),
            'lstm_context_norm': float(np.linalg.norm(context)),
            'lstm_forward_final_norm': float(np.linalg.norm(final_fwd)),
            'lstm_backward_final_norm': float(np.linalg.norm(final_bwd)),
        }
    
    def _text_to_sequence(self, text: str, max_len: int = 200) -> np.ndarray:
        """Convert text to numerical sequence"""
        # Character-level encoding
        char_codes = np.array([ord(c) / 1114111.0 for c in text[:max_len]])
        
        if len(char_codes) == 0:
            return np.zeros((1, self.input_dim))
        
        # Reshape and pad to input_dim
        sequence = []
        for code in char_codes:
            vec = np.zeros(self.input_dim)
            vec[0] = code
            sequence.append(vec)
        
        return np.array(sequence)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: HYBRID CNN-RNN ARCHITECTURE (LAYER 4)
# ═══════════════════════════════════════════════════════════════════════════

class HybridCNNRNN:
    """
    Hybrid CNN-RNN with Residual Connections
    
    Architecture:
    1. Conv layers for local pattern extraction
    2. Max pooling for dimensionality reduction
    3. RNN layers for sequential modeling
    4. Residual connections for gradient flow
    """
    
    def __init__(self, num_filters: int = 128, filter_sizes: List[int] = [2, 3, 4, 5],
                 rnn_hidden: int = 128):
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.rnn_hidden = rnn_hidden
        
        # CNN filters
        self.filters = {}
        for size in filter_sizes:
            self.filters[size] = np.random.randn(num_filters, size, 1) * 0.02
        
        # RNN parameters
        self.W_rnn = np.random.randn(rnn_hidden, num_filters * len(filter_sizes) + rnn_hidden) * 0.02
        self.U_rnn = np.random.randn(rnn_hidden, rnn_hidden) * 0.02
        
        # Residual projection
        self.W_residual = np.random.randn(rnn_hidden, num_filters * len(filter_sizes)) * 0.02
        
        # Classification
        self.W_clf = np.random.randn(rnn_hidden, 1) * 0.02
        
    def convolve(self, text_vector: np.ndarray, filter_size: int) -> np.ndarray:
        """Apply convolution operation"""
        filters = self.filters[filter_size]
        text_len = len(text_vector)
        
        if text_len < filter_size:
            return np.zeros(self.num_filters)
        
        feature_maps = []
        for f in filters:
            # Convolve
            conv_results = []
            for i in range(text_len - filter_size + 1):
                window = text_vector[i:i+filter_size].reshape(-1, 1)
                activation = np.sum(f.reshape(-1) * window.reshape(-1))
                conv_results.append(np.maximum(0, activation))  # ReLU
            
            # Max pooling
            if conv_results:
                feature_maps.append(max(conv_results))
            else:
                feature_maps.append(0.0)
        
        return np.array(feature_maps)
    
    def cnn_forward(self, text: str) -> np.ndarray:
        """CNN forward pass"""
        # Convert text to vector
        text_vector = np.array([ord(c) / 1114111.0 for c in text[:500]])
        
        if len(text_vector) == 0:
            return np.zeros(self.num_filters * len(self.filter_sizes))
        
        # Apply convolutions with different filter sizes
        all_features = []
        for filter_size in self.filter_sizes:
            features = self.convolve(text_vector, filter_size)
            all_features.append(features)
        
        # Concatenate all features
        return np.concatenate(all_features)
    
    def rnn_forward(self, cnn_features: np.ndarray) -> np.ndarray:
        """RNN forward pass with residual connection"""
        h = np.zeros(self.rnn_hidden)
        
        # Treat CNN features as a single timestep input
        combined = np.concatenate([cnn_features, h])
        h_new = np.tanh(self.W_rnn @ combined)
        
        # Residual connection
        residual = np.tanh(self.W_residual @ cnn_features)
        if residual.shape[0] != h_new.shape[0]:
            residual = np.pad(residual, (0, max(0, h_new.shape[0] - residual.shape[0])))[:h_new.shape[0]]
        
        h_final = h_new + residual
        
        return h_final
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text using hybrid CNN-RNN"""
        # CNN forward pass
        cnn_features = self.cnn_forward(text)
        
        # RNN forward pass
        rnn_output = self.rnn_forward(cnn_features)
        
        # Classification
        logit = self.W_clf.T @ rnn_output
        fraud_prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        
        return {
            'cnn_rnn_fraud_score': float(fraud_prob[0]),
            'cnn_features_norm': float(np.linalg.norm(cnn_features)),
            'cnn_features_mean': float(np.mean(cnn_features)),
            'cnn_features_std': float(np.std(cnn_features)),
            'rnn_output_norm': float(np.linalg.norm(rnn_output)),
        }

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: VARIATIONAL AUTOENCODER FOR ANOMALY DETECTION (LAYER 6)
# ═══════════════════════════════════════════════════════════════════════════

class VariationalAutoencoder:
    """
    Variational Autoencoder for anomaly detection
    
    Mathematical Model:
    - Encoder: q(z|x) = N(μ(x), σ²(x))
    - Decoder: p(x|z)
    - Loss: 𝓛 = -E[log p(x|z)] + KL[q(z|x) || p(z)]
    - KL Divergence: KL = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
    
    High reconstruction error indicates anomaly (fraud)
    """
    
    def __init__(self, input_dim: int = 100, latent_dim: int = 20):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.W_enc1 = np.random.randn(input_dim, 64) * 0.02
        self.W_enc2 = np.random.randn(64, 32) * 0.02
        self.W_mu = np.random.randn(32, latent_dim) * 0.02
        self.W_logvar = np.random.randn(32, latent_dim) * 0.02
        
        # Decoder
        self.W_dec1 = np.random.randn(latent_dim, 32) * 0.02
        self.W_dec2 = np.random.randn(32, 64) * 0.02
        self.W_dec3 = np.random.randn(64, input_dim) * 0.02
        
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent space
        Returns: (μ, log(σ²))
        """
        h1 = np.maximum(0, x @ self.W_enc1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W_enc2)
        
        mu = h2 @ self.W_mu
        logvar = h2 @ self.W_logvar
        
        return mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = μ + σ * ε
        where ε ~ N(0, 1)
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstruction"""
        h1 = np.maximum(0, z @ self.W_dec1)
        h2 = np.maximum(0, h1 @ self.W_dec2)
        reconstruction = self._sigmoid(h2 @ self.W_dec3)
        return reconstruction
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def reconstruction_error(self, x: np.ndarray) -> float:
        """
        Compute reconstruction error (anomaly score)
        Higher error = more anomalous = more likely fraud
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        reconstruction, mu, logvar = self.forward(x)
        
        # MSE reconstruction error
        mse = np.mean((x - reconstruction) ** 2)
        
        # KL divergence
        kl_div = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        # Total loss (ELBO)
        total_loss = mse + 0.01 * kl_div  # Weight KL term
        
        return float(total_loss)
    
    def compute_anomaly_score(self, x: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive anomaly metrics"""
        reconstruction, mu, logvar = self.forward(x)
        
        mse = np.mean((x - reconstruction) ** 2)
        mae = np.mean(np.abs(x - reconstruction))
        kl_div = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        # Normalize to [0, 1]
        anomaly_score = min(mse * 10, 1.0)
        
        return {
            'vae_anomaly_score': anomaly_score,
            'vae_reconstruction_mse': float(mse),
            'vae_reconstruction_mae': float(mae),
            'vae_kl_divergence': float(kl_div),
            'vae_latent_mean_norm': float(np.linalg.norm(mu)),
            'vae_latent_std': float(np.mean(np.exp(0.5 * logvar))),
        }
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: DEEP Q-NETWORK REINFORCEMENT LEARNING (LAYER 7)
# ═══════════════════════════════════════════════════════════════════════════

class DeepQNetworkAgent:
    """
    Deep Q-Network for optimal fraud response strategy
    
    Mathematical Model:
    - Q(s, a; θ) ≈ Q*(s, a)
    - Loss: 𝓛 = E[(r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ))²]
    - ε-greedy: a = argmax_a Q(s, a) with probability 1-ε
    """
    
    def __init__(self, state_dim: int = 100, action_dim: int = 5, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Q-Network
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.02
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.02
        
        # Target network (frozen copy)
        self.W1_target = self.W1.copy()
        self.W2_target = self.W2.copy()
        self.W3_target = self.W3.copy()
        
        # Exploration parameters
        self.epsilon = 0.1  # Low epsilon for production
        self.gamma = 0.99  # Discount factor
        
        # Actions
        self.actions = [
            'ALLOW',           # 0: Legitimate message
            'MONITOR',         # 1: Suspicious, continue engagement
            'WARN_USER',       # 2: Warn but don't block
            'SOFT_BLOCK',      # 3: Block with explanation
            'HARD_BLOCK'       # 4: Immediate block
        ]
    
    def q_network(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        """Compute Q-values for all actions"""
        W1 = self.W1_target if target else self.W1
        W2 = self.W2_target if target else self.W2
        W3 = self.W3_target if target else self.W3
        
        h1 = np.maximum(0, state @ W1)  # ReLU
        h2 = np.maximum(0, h1 @ W2)
        q_values = h2 @ W3
        
        return q_values
    
    def select_action(self, state: np.ndarray, fraud_score: float) -> Tuple[str, int]:
        """
        Select optimal action based on state
        
        If fraud_score is very high, bias toward blocking actions
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Compute Q-values
        q_values = self.q_network(state)
        
        # ε-greedy with fraud score bias
        if np.random.random() < self.epsilon:
            # Explore: random action weighted by fraud score
            if fraud_score > 0.8:
                # Bias toward blocking
                probs = np.array([0.05, 0.15, 0.15, 0.25, 0.40])
            else:
                # Uniform
                probs = np.ones(self.action_dim) / self.action_dim
            
            action_idx = np.random.choice(self.action_dim, p=probs)
        else:
            # Exploit: best action
            # Add fraud score bias to Q-values
            bias = np.array([0, 0.1, 0.2, 0.3, 0.4]) * fraud_score
            adjusted_q = q_values[0] + bias
            action_idx = int(np.argmax(adjusted_q))
        
        return self.actions[action_idx], action_idx
    
    def compute_state_value(self, state: np.ndarray) -> float:
        """Compute V(s) = max_a Q(s, a)"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        q_values = self.q_network(state)
        return float(np.max(q_values))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: GRADIENT BOOSTING WITH FEATURE INTERACTIONS (LAYER 8)
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedGradientBoosting:
    """
    Gradient Boosting with feature interaction modeling
    
    Mathematical Model:
    - F_m(x) = F_{m-1}(x) + η * h_m(x)
    - h_m(x) = argmin_h Σ L(y_i, F_{m-1}(x_i) + h(x_i))
    - Feature interactions: h(x) can split on x_i * x_j
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, interaction_depth: int = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.interaction_depth = interaction_depth
        self.trees = []
        self.base_score = 0.5
        
    def _create_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create pairwise interaction features
        [x1, x2, x3] -> [x1, x2, x3, x1*x2, x1*x3, x2*x3]
        """
        n_samples, n_features = X.shape
        interactions = []
        
        # Add original features
        interactions.append(X)
        
        # Add pairwise interactions
        if self.interaction_depth >= 2:
            for i in range(n_features):
                for j in range(i + 1, min(i + 10, n_features)):  # Limit interactions
                    interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        return np.hstack(interactions)
    
    def _build_tree(self, X: np.ndarray, residuals: np.ndarray, depth: int = 0) -> Dict:
        """Build decision tree on residuals"""
        if depth >= self.max_depth or len(X) < 10:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        best_split = self._find_best_split(X, residuals)
        
        if best_split is None:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], residuals[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], residuals[right_mask], depth + 1),
        }
    
    def _find_best_split(self, X: np.ndarray, residuals: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find best split using variance reduction"""
        best_gain = -float('inf')
        best_split = None
        
        n_features = X.shape[1]
        features_to_try = np.random.choice(n_features, min(int(np.sqrt(n_features)) + 5, n_features), replace=False)
        
        for feature_idx in features_to_try:
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                
                # Variance reduction
                total_var = np.var(residuals)
                left_var = np.var(residuals[left_mask])
                right_var = np.var(residuals[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(residuals)
                
                weighted_var = (n_left / n_total) * left_var + (n_right / n_total) * right_var
                gain = total_var - weighted_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, float(threshold))
        
        return best_split
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train gradient boosting model"""
        # Add interaction features
        X_interact = self._create_interaction_features(X)
        
        # Initialize predictions
        predictions = np.full(len(X), self.base_score)
        
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions
            
            # Build tree
            tree = self._build_tree(X_interact, residuals)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(X_interact, tree)
            predictions += self.learning_rate * tree_predictions
    
    def _predict_tree(self, X: np.ndarray, tree: Dict) -> np.ndarray:
        """Get predictions from single tree"""
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            predictions[i] = self._predict_single(X[i], tree)
        
        return predictions
    
    def _predict_single(self, x: np.ndarray, tree: Dict) -> float:
        """Predict single sample"""
        if tree['type'] == 'leaf':
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_interact = self._create_interaction_features(X)
        
        predictions = np.full(len(X), self.base_score)
        
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(X_interact, tree)
        
        # Sigmoid
        return 1 / (1 + np.exp(-predictions))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: BAYESIAN NEURAL NETWORK (LAYER 9)
# ═══════════════════════════════════════════════════════════════════════════

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network for uncertainty quantification
    
    Mathematical Model:
    - Weight posterior: p(W|D) ∝ p(D|W) * p(W)
    - Prediction: p(y|x,D) = ∫ p(y|x,W) p(W|D) dW
    - MC Dropout approximation: E[y] ≈ 1/T Σ f(x; W_t)
    - Uncertainty: Var[y] ≈ 1/T Σ f(x; W_t)² - E[y]²
    """
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.2, n_samples: int = 20):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Network weights
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim))
            self.biases.append(np.zeros((1, hidden_dim)))
            prev_dim = hidden_dim
        
        # Output layer
        self.weights.append(np.random.randn(prev_dim, 1) * np.sqrt(2.0 / prev_dim))
        self.biases.append(np.zeros((1, 1)))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout"""
        activation = x
        
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            # Linear
            z = activation @ W + b
            
            # ReLU
            activation = np.maximum(0, z)
            
            # Dropout
            if training:
                mask = np.random.binomial(1, 1 - self.dropout_rate, activation.shape)
                mask = mask / (1 - self.dropout_rate)
                activation = activation * mask
        
        # Output layer (sigmoid)
        z = activation @ self.weights[-1] + self.biases[-1]
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        return output
    
    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, float]:
        """
        Make prediction with uncertainty quantification
        
        Returns mean, variance, and confidence interval
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Monte Carlo sampling
        predictions = []
        for _ in range(self.n_samples):
            pred = self.forward(x, training=True)  # Always use dropout
            predictions.append(float(pred[0]))
        
        predictions = np.array(predictions)
        
        # Statistics
        mean = np.mean(predictions)
        variance = np.var(predictions)
        std = np.std(predictions)
        
        # Epistemic uncertainty (knowledge uncertainty)
        epistemic = variance
        
        # Aleatoric uncertainty (data uncertainty) - approximation
        aleatoric = mean * (1 - mean)  # Bernoulli variance
        
        # Total uncertainty
        total_uncertainty = epistemic + aleatoric
        
        # Confidence (inverse of uncertainty)
        confidence = 1 / (1 + total_uncertainty)
        
        return {
            'bayesian_mean': float(mean),
            'bayesian_std': float(std),
            'bayesian_epistemic_uncertainty': float(epistemic),
            'bayesian_aleatoric_uncertainty': float(aleatoric),
            'bayesian_total_uncertainty': float(total_uncertainty),
            'bayesian_confidence': float(confidence),
            'bayesian_ci_lower': float(np.percentile(predictions, 2.5)),
            'bayesian_ci_upper': float(np.percentile(predictions, 97.5)),
        }

# This file continues in the next part with Meta-Learning and Master System...
"""
═══════════════════════════════════════════════════════════════════════════
TO BE CONTINUED IN PART 3:
- Meta-Learning Aggregator (Layer 10)
- Dual-LLM Air-Gapped Cognitive Core (Layer 11)  
- Master NEXUS-GUARDIAN Integration System
- API Handler and Callback Logic
═══════════════════════════════════════════════════════════════════════════
"""
