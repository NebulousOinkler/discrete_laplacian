Title: Martingale Construction for Multi-State Battery Health Monitoring with Recovery and Multiple Failure Modes
Date: 2024-10-30
Category: blog
Tags: Statistical Theory, BHM
Status: published

## Multi-State Battery Health Model

### State Space Definition

Define the battery state space $\mathcal{S} = \{H, D_1, D_2, \ldots, D_k, F_1, F_2, \ldots, F_m, R\}$ where:
- $H$: Healthy/Normal operating state
- $D_i$: Degradation state $i$ (recoverable)
- $F_j$: Failure state $j$ (non-recoverable)
- $R$: Recovery/Rejuvenation state (during maintenance)

The transition intensity matrix $Q$ for continuous-time monitoring:

$$ Q = \begin{pmatrix}
-\lambda_H & \lambda_{HD_1} & \lambda_{HD_2} & \cdots & \lambda_{HF_1} & \cdots & \lambda_{HR} \\
\mu_{D_1H} & -\lambda_{D_1} & \lambda_{D_1D_2} & \cdots & \lambda_{D_1F_1} & \cdots & \lambda_{D_1R} \\
\vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & -\lambda_{F_j} & \cdots & 0 \\
\mu_{RH} & \mu_{RD_1} & 0 & \cdots & 0 & \cdots & -\lambda_R
\end{pmatrix} $$

where $\lambda_{ij}$ represents transition rate from state $i$ to $j$, and $\mu_{ij}$ represents recovery rates.

### Observable Signal Model

For state $s \in \mathcal{S}$, the observation process follows:

$$ Y_t | X_t = s \sim \mathcal{N}(\mu_s(t), \Sigma_s(t)) $$

where $Y_t = (V_t, I_t, T_t, Z_t, \text{SoC}_t)$ is the multivariate observation.

## Martingale Construction for State Transitions

### Likelihood Ratio Martingale

For hypothesis testing between state trajectories, construct the likelihood ratio process:

$$ M_t = \frac{dP^{(1)}}{dP^{(0)}}(Y_{0:t}) = \prod_{i=0}^{t} \frac{p^{(1)}(Y_i | Y_{0:i-1})}{p^{(0)}(Y_i | Y_{0:i-1})} $$

Under $P^{(0)}$, this forms a martingale with $\mathbb{E}_{P^{(0)}}[M_t | \mathcal{F}_{t-1}] = M_{t-1}$.

### State-Specific Martingales

For each state $s \in \mathcal{S}$, define the state likelihood martingale:

$$ M_t^{(s)} = \frac{P(Y_{0:t} | X_t = s)}{P(Y_{0:t} | X_0 = H)} \cdot \frac{P(X_t = s)}{P(X_0 = H)} $$

The posterior probability process:

$$ \pi_t^{(s)} = P(X_t = s | Y_{0:t}) = \frac{M_t^{(s)}}{\sum_{s' \in \mathcal{S}} M_t^{(s')}} $$

### Compensated Counting Process Martingale

For transitions between states, define counting processes $N_{ij}(t)$ for transitions from state $i$ to $j$:

$$ M_{ij}(t) = N_{ij}(t) - \int_0^t \lambda_{ij}(u) \cdot \mathbb{1}(X_u = i) \, du $$

This compensated process is a martingale under the true model.

## Recovery Detection Martingales

### Health Score Process

Define a health score function $h: \mathcal{S} \rightarrow [0, 1]$:

$$ h(s) = \begin{cases}
1 & s = H \\
0.7 + 0.3e^{-\kappa_i} & s = D_i \\
0.9 & s = R \\
\delta_j & s = F_j
\end{cases} $$

where $\kappa_i$ is degradation severity and $\delta_j \ll 1$ for failure states.

### Recovery Martingale

Construct a martingale that captures improvement in health:

$$ M_t^{\text{recovery}} = h(X_t) - h(X_0) - \int_0^t \mathcal{L}h(X_s) \, ds $$

where $\mathcal{L}$ is the infinitesimal generator:

$$ \mathcal{L}h(s) = \sum_{s' \neq s} q_{ss'}(h(s') - h(s)) $$

### Cumulative Recovery E-Process

Define the recovery E-process:

$$ E_t^{\text{rec}} = \exp\left(\int_0^t \lambda_{\text{rec}}(s) \, dM_s^{\text{recovery}} - \frac{1}{2}\int_0^t \lambda_{\text{rec}}^2(s) \, d\langle M^{\text{recovery}} \rangle_s\right) $$

This detects statistically significant health improvements.

## Multiple Failure Mode Detection

### Failure Mode Signatures

Each failure mode $F_j$ has characteristic signature:

1. **Capacity Fade** ($F_1$):
   $$ \mu_{F_1} = (\bar{V} - \alpha t, \bar{I}, \bar{T}, Z_0 e^{\beta t}, \text{SoC}_{\max} \cdot (1 - \gamma t)) $$

2. **Internal Short** ($F_2$):
   $$ \mu_{F_2} = (\bar{V} - \Delta V_{\text{short}}, I_{\text{leak}}, T_{\text{elevated}}, Z_{\text{low}}, \text{SoC}_{\text{unstable}}) $$

3. **Thermal Runaway Risk** ($F_3$):
   $$ \mu_{F_3} = (\bar{V}, \bar{I}, T_{\text{high}} + \eta(t), Z_{\text{varying}}, \text{SoC}_{\text{normal}}) $$

4. **Dendrite Formation** ($F_4$):
   $$ \mu_{F_4} = (V_{\text{spiky}}(t), I_{\text{irregular}}, \bar{T}, Z_{\text{oscillating}}(t), \text{SoC}_{\text{normal}}) $$

### Competing Risk Martingales

For simultaneous detection of multiple failure modes:

$$ M_t^{(F_j)} = \exp\left(\sum_{i=1}^{n} \log\frac{f_{F_j}(Y_{t_i} | \mathcal{F}_{t_{i-1}})}{f_H(Y_{t_i} | \mathcal{F}_{t_{i-1}})}\right) $$

The first failure detected:

$$ \hat{F} = \arg\max_{j} \{M_t^{(F_j)} : M_t^{(F_j)} > \tau_j\} $$

where $\tau_j$ are mode-specific thresholds.

## Semi-Markov Model Extensions

### State Sojourn Times

Incorporate state duration dependencies:

$$ P(X_{t+\Delta t} = j | X_t = i, S_i = s) = q_{ij}(s) \Delta t + o(\Delta t) $$

where $S_i$ is sojourn time in state $i$ and $q_{ij}(s)$ is duration-dependent transition rate.

### Sojourn-Adjusted Martingale

$$ M_t^{\text{semi}} = \prod_{k=1}^{N(t)} \frac{f_{X_k}(S_k)}{g_{X_k}(S_k)} \cdot \prod_{k=0}^{N(t)} \frac{p_{X_k X_{k+1}}}{q_{X_k X_{k+1}}} $$

where $f$ and $g$ are sojourn distributions under alternative and null hypotheses.

## Hidden Markov Model Filtering

### Forward-Backward Martingale

The forward filtering process:

$$ \alpha_t(s) = P(X_t = s | Y_{1:t}) = \frac{P(Y_t | X_t = s) \sum_{s'} P(X_t = s | X_{t-1} = s') \alpha_{t-1}(s')}{P(Y_t | Y_{1:t-1})} $$

Define the prediction error martingale:

$$ M_t^{\text{pred}} = Y_t - \mathbb{E}[Y_t | Y_{1:t-1}] = Y_t - \sum_{s \in \mathcal{S}} \mu_s \cdot P(X_t = s | Y_{1:t-1}) $$

### Viterbi-Based Martingale

For most likely path detection:

$$ V_t(s) = \max_{x_{0:t-1}} P(X_{0:t-1} = x_{0:t-1}, X_t = s, Y_{1:t}) $$

The path probability martingale:

$$ M_t^{\text{path}} = \frac{\max_s V_t(s)}{P(Y_{1:t})} $$

## Adaptive Learning and Online Updates

### Parameter Learning Martingale

For unknown transition rates, use the EM algorithm score process:

$$ S_t(\theta) = \sum_{i=1}^{t} \left(\nabla_\theta \log p_\theta(Y_i | Y_{1:i-1}) - \mathbb{E}_\theta[\nabla_\theta \log p_\theta(Y_i | Y_{1:i-1})]\right) $$

This score process is a martingale under true parameter $\theta_0$.

### Online Transition Rate Estimation

Update transition rates using martingale estimators:

$$ \hat{\lambda}_{ij}(t) = \frac{N_{ij}(t)}{\int_0^t \mathbb{1}(X_s = i) \, ds} $$

with confidence bounds from martingale CLT:

$$ \sqrt{t}(\hat{\lambda}_{ij}(t) - \lambda_{ij}) \xrightarrow{d} \mathcal{N}(0, \lambda_{ij}) $$

## Practical Implementation Algorithms

### Algorithm 1: Multi-State Health Monitoring

```
class MultiStateBatteryMonitor:
    def __init__(self, states, transition_matrix, observation_models):
        self.states = states  # [H, D1, D2, F1, F2, R]
        self.Q = transition_matrix
        self.obs_models = observation_models
        self.state_probs = np.ones(len(states)) / len(states)
        self.martingales = {s: 1.0 for s in states}
        
    def update(self, observation):
        # Update state-specific martingales
        for state in self.states:
            likelihood = self.obs_models[state].pdf(observation)
            baseline_likelihood = self.obs_models['H'].pdf(observation)
            
            # Martingale increment
            increment = likelihood / baseline_likelihood
            self.martingales[state] *= increment
            
        # Normalize to get posterior probabilities
        total = sum(self.martingales.values())
        self.state_probs = {s: m/total for s, m in self.martingales.items()}
        
        # Check for state transitions
        return self.detect_transitions()
    
    def detect_transitions(self):
        max_state = max(self.state_probs, key=self.state_probs.get)
        
        if max_state in ['D1', 'D2']:
            return f"DEGRADATION: {max_state}"
        elif max_state in ['F1', 'F2']:
            return f"FAILURE: {max_state}"
        elif max_state == 'R':
            return "RECOVERY DETECTED"
        elif max_state == 'H' and self.previous_state != 'H':
            return "RETURNED TO HEALTHY"
        return "NORMAL"
```

### Algorithm 2: Recovery Detection with E-Process

```
class RecoveryDetector:
    def __init__(self, health_score_func, significance_level=0.05):
        self.h = health_score_func
        self.alpha = significance_level
        self.E_recovery = 1.0
        self.health_history = []
        
    def update(self, state_estimate, observation):
        current_health = self.h(state_estimate)
        self.health_history.append(current_health)
        
        if len(self.health_history) > 1:
            # Compute health improvement
            delta_h = current_health - self.health_history[-2]
            
            # Update E-process for recovery
            if delta_h > 0:
                # Positive evidence for recovery
                lambda_opt = self.optimize_lambda(delta_h)
                increment = np.exp(lambda_opt * delta_h - self.psi(lambda_opt))
                self.E_recovery *= increment
                
            # Check for significant recovery
            if self.E_recovery > 1/self.alpha:
                return "SIGNIFICANT RECOVERY DETECTED"
                
        return None
    
    def optimize_lambda(self, delta_h):
        # Adaptive lambda based on improvement magnitude
        return min(1.0, delta_h / 0.1)  # Cap at 1.0
    
    def psi(self, lambda_val):
        # MGF under null (no recovery)
        return lambda_val**2 * self.variance_null / 2
```

### Algorithm 3: Competing Failure Modes

```
class CompetingFailureDetector:
    def __init__(self, failure_models):
        self.failure_models = failure_models
        self.failure_martingales = {f: 1.0 for f in failure_models.keys()}
        self.thresholds = self.set_bonferroni_thresholds()
        
    def update(self, observation):
        for failure_mode, model in self.failure_models.items():
            # Compute likelihood ratio
            l_failure = model.likelihood(observation)
            l_healthy = self.healthy_model.likelihood(observation)
            
            # Update martingale
            self.failure_martingales[failure_mode] *= (l_failure / l_healthy)
            
        # Check for failure detection
        detected = []
        for mode, M in self.failure_martingales.items():
            if M > self.thresholds[mode]:
                detected.append((mode, M))
                
        if detected:
            # Return most likely failure
            return max(detected, key=lambda x: x[1])[0]
            
        return None
    
    def set_bonferroni_thresholds(self, alpha=0.01):
        n_modes = len(self.failure_models)
        return {mode: n_modes/alpha for mode in self.failure_models}
```

## Optimal Stopping for Maintenance Decisions

### Value Function Martingale

Define the value function for maintenance decision:

$$ V_t = \sup_{\tau \geq t} \mathbb{E}\left[\int_t^\tau r(X_s) \, ds - c(\tau, X_\tau) \Big| \mathcal{F}_t\right] $$

where $r(s)$ is reward rate in state $s$ and $c(\tau, s)$ is maintenance cost.

The martingale:

$$ M_t^V = V_t - V_0 - \int_0^t \mathcal{L}V(X_s) \, ds $$

### Optimal Maintenance Trigger

Maintenance is triggered when:

$$ M_t^{\text{maint}} = \frac{P(X_t \in \{D_i\} \cup \{F_j\})}{P(X_t = H)} > \tau^* $$

where $\tau^*$ solves:

$$ \tau^* = \arg\min_\tau \left\{c_{\text{prev}} \cdot P(\tau < T_{\text{failure}}) + c_{\text{fail}} \cdot P(\tau \geq T_{\text{failure}})\right\} $$

## Continuous Learning Framework

### Bayesian Martingale Updates

Incorporate prior knowledge via conjugate priors:

$$ \pi_t(\theta) \propto \pi_0(\theta) \cdot \exp\left(\int_0^t \log p_\theta(dY_s | \mathcal{F}_{s-})\right) $$

The posterior mean process:

$$ \hat{\theta}_t = \mathbb{E}_{\pi_t}[\theta] $$

forms a martingale under the prior predictive distribution.

### Reinforcement Learning Integration

Use martingale differences for policy gradient:

$$ \nabla J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} M_t^{\text{reward}} \cdot \nabla \log \pi_\theta(a_t | s_t)\right] $$

where:

$$ M_t^{\text{reward}} = R_t - \mathbb{E}[R_t | \mathcal{F}_{t-1}] $$

## Validation and Performance Metrics

### Martingale Residual Analysis

Test martingale property via residuals:

$$ R_t = \frac{M_t - M_{t-1}}{\sqrt{\langle M \rangle_t - \langle M \rangle_{t-1}}} $$

Under correct model, $R_t \sim \mathcal{N}(0, 1)$ independently.

### Cross-Entropy for State Estimation

Evaluate state estimation accuracy:

$$ H(P || \hat{P}) = -\sum_{s \in \mathcal{S}} P(X_t = s) \log \hat{P}(X_t = s | Y_{1:t}) $$

### Recovery Detection Performance

- **Sensitivity**: $P(\text{detect recovery} | \text{true recovery})$
- **Specificity**: $P(\text{no detection} | \text{no recovery})$
- **Time to detection**: $\mathbb{E}[\tau_{\text{detect}} - \tau_{\text{recovery}}]$

## Case Study: EV Battery Pack with Cell Balancing

### State Space for Cell-Level Monitoring

For $n$ cells in series:

$$ \mathcal{S} = \{H, D_{\text{imbalance}}, D_{\text{weak}}, F_{\text{open}}, F_{\text{short}}, R_{\text{balanced}}\} $$

### Cell Balancing as Recovery Mechanism

Balancing triggers transition $D_{\text{imbalance}} \rightarrow R_{\text{balanced}} \rightarrow H$:

$$ P(R_{\text{balanced}} \rightarrow H) = 1 - \exp(-\mu_{\text{balance}} \cdot t_{\text{balance}}) $$

### Multi-Level Martingales

Hierarchical monitoring:

1. **Cell Level**: $M_t^{(i)}$ for cell $i$
2. **Module Level**: $M_t^{\text{module}} = \prod_{i \in \text{module}} M_t^{(i)}$
3. **Pack Level**: $M_t^{\text{pack}} = \min_{\text{module}} M_t^{\text{module}}$

## Advanced Topics

### Non-Linear State Space Models

For non-linear dynamics:

$$ X_{t+1} = f(X_t, W_t), \quad Y_t = g(X_t, V_t) $$

Use particle filter martingales:

$$ M_t^{\text{PF}} = \frac{1}{N} \sum_{i=1}^{N} w_t^{(i)} $$

where $w_t^{(i)}$ are importance weights.

### Deep Learning Integration

Neural network state classifier with martingale regularization:

$$ \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathbb{E}\left[(M_{t+1} - M_t)^2 | \mathcal{F}_t\right] $$

This ensures learned representations maintain martingale property.

### Quantum Battery States

For quantum batteries with superposition states:

$$ |\psi\rangle = \sum_{s \in \mathcal{S}} \alpha_s |s\rangle $$

The measurement process creates quantum martingales:

$$ M_t^Q = \langle \psi_t | \hat{M} | \psi_t \rangle $$

## Conclusion

The martingale framework for multi-state battery health monitoring enables:

1. **Unified Detection**: Simultaneous monitoring of degradation, failure, and recovery events
2. **Optimal Decision Making**: Maintenance scheduling via optimal stopping theory
3. **Uncertainty Quantification**: Rigorous confidence bounds on state estimates
4. **Adaptive Learning**: Online parameter updates maintaining statistical guarantees
5. **Hierarchical Monitoring**: From cell to pack level with consistent framework

Key innovations include:
- Recovery martingales that detect health improvements
- Competing risk framework for multiple failure modes
- Semi-Markov extensions for state-duration dependencies
- Integration with reinforcement learning for adaptive maintenance

This framework provides the theoretical foundation for next-generation battery management systems capable of detecting not just failures but also recovery events, enabling proactive maintenance and extended battery life.