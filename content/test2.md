Title: E-values, Sequential Testing, and Concentrated Differential Privacy: Theory and Practice
Date: 2024-10-27
Category: Blog
Tags: Statistical Theory
Status: published

## Introduction to E-values and E-processes

An **E-value** (evidence value) is a non-negative random variable $E$ with expected value at most 1 under the null hypothesis:

\[ \mathbb{E}_{H_0}[E] \leq 1 \]

E-values generalize likelihood ratios and provide a universal framework for hypothesis testing without p-values. The fundamental property enabling sequential analysis is **Ville's inequality**: for any E-value $E$ and $\alpha > 0$:

$$ P_{H_0}(E \geq 1/\alpha) \leq \alpha $$

An **E-process** $(E_t)_{t=1}^{\infty}$ is a non-negative adapted process where each $E_t$ is an E-value with respect to the filtration $\mathcal{F}_t$. Formally:

\[ \mathbb{E}_{H_0}[E_t | \mathcal{F}_{t-1}] \leq E_{t-1} \]

This makes $(E_t)$ a supermartingale under $H_0$, enabling anytime-valid inference.

## Mathematical Framework of Sequential Testing

### Test Martingales and Supermartingales

Let $X_1, X_2, \ldots$ be observations. A **test martingale** $M_t$ for testing $H_0: \theta = \theta_0$ versus $H_1: \theta = \theta_1$ is:

$$ M_t = \prod_{i=1}^{t} \frac{f_{\theta_1}(X_i | X_1, \ldots, X_{i-1})}{f_{\theta_0}(X_i | X_1, \ldots, X_{i-1})} $$

Under $H_0$, $(M_t)$ is a martingale with $\mathbb{E}_{H_0}[M_t] = 1$.

For composite hypotheses $H_1: \theta \in \Theta_1$, we construct test supermartingales via **mixture methods**:

$$ S_t = \int_{\Theta_1} M_t(\theta) \, d\pi(\theta) $$

where $\pi$ is a prior over $\Theta_1$, or via **maximum likelihood**:

$$ S_t = \sup_{\theta \in \Theta_1} M_t(\theta) $$

### Anytime-Valid Confidence Sequences

A $(1-\alpha)$-confidence sequence $(C_t)_{t=1}^{\infty}$ satisfies:

$$ P_{\theta}(\forall t \geq 1: \theta \in C_t) \geq 1 - \alpha $$

Confidence sequences are dual to E-processes via the fundamental relationship:

$$ C_t = \{\theta : E_t(\theta) < 1/\alpha\} $$

where $E_t(\theta)$ is an E-process for testing $H_0: \theta$.

## Exponential Tilting and Optimal E-processes

### Basic Exponential Tilting

For bounded observations $X_i \in [a, b]$, exponential tilting creates E-values via:

$$ E_{\lambda} = \exp\left(\lambda \sum_{i=1}^{n} (X_i - \mu_0) - n\psi(\lambda)\right) $$

where $\psi(\lambda) = \log \mathbb{E}_{H_0}[e^{\lambda(X - \mu_0)}]$ is the cumulant generating function.

### Mixture E-values and Optimization

The **mixture E-value** over tilting parameters:

$$ E_{\text{mix}} = \int_{\Lambda} E_{\lambda} \, d\pi(\lambda) $$

achieves minimax optimality for certain priors $\pi$. The optimal prior for sub-Gaussian observations is:

$$ \pi^*(\lambda) \propto \exp\left(-\frac{\lambda^2 \sigma^2}{2}\right) $$

yielding the E-value:

$$ E_t = \exp\left(\frac{S_t^2}{2t\sigma^2} - \frac{S_t^2}{2t\sigma^2 + 2\sigma^2}\right) $$

where $S_t = \sum_{i=1}^{t} (X_i - \mu_0)$.

### Predictable Mixture Martingales

For adaptive testing, we use **predictable mixtures**:

$$ M_t = \prod_{i=1}^{t} \int \frac{dP_{\theta}(X_i | \mathcal{F}_{i-1})}{dP_0(X_i | \mathcal{F}_{i-1})} d\pi_i(\theta) $$

where $\pi_i$ can depend on past observations through $\mathcal{F}_{i-1}$.

## Connection to Concentrated Differential Privacy

### Concentrated Differential Privacy (CDP)

A randomized mechanism $\mathcal{M}$ satisfies $\rho$-concentrated differential privacy if for neighboring datasets $D, D'$:

$$ D_{\alpha}(\mathcal{M}(D) || \mathcal{M}(D')) \leq \rho $$

for all $\alpha > 1$, where $D_{\alpha}$ is the Rényi divergence of order $\alpha$.

### Privacy Loss as a Martingale

The privacy loss random variable for a single query:

$$ Z = \log\frac{P(\mathcal{M}(D) = y)}{P(\mathcal{M}(D') = y)} $$

For sequential mechanisms, the cumulative privacy loss:

$$ L_t = \sum_{i=1}^{t} Z_i $$

forms a martingale under specific conditions, enabling sequential analysis.

### E-processes for Privacy Accounting

Define the privacy E-process:

$$ E_t^{\text{priv}} = \exp(\lambda L_t - t\psi(\lambda)) $$

where $\psi(\lambda) = \log \mathbb{E}[e^{\lambda Z}]$ is the privacy loss CGF. Under $\rho$-CDP:

$$ \psi(\lambda) \leq \frac{\lambda(\lambda + 1)\rho}{2} $$

This E-process enables anytime-valid privacy guarantees:

$$ P(L_t \geq \log(1/\delta) + t\psi(\lambda)/\lambda) \leq \delta \cdot e^{-\lambda \log(1/\delta) + t\psi(\lambda)} $$

## Composition Theorems in Sequential Testing

### Basic Composition for E-processes

**Theorem (Product Composition):** If $E^{(1)}, \ldots, E^{(k)}$ are independent E-values for testing $H_0^{(1)}, \ldots, H_0^{(k)}$, then:

$$ E = \prod_{i=1}^{k} E^{(i)} $$

is an E-value for testing $H_0 = \cap_{i=1}^{k} H_0^{(i)}$.

### Advanced Composition via Rényi Divergence

**Theorem (Rényi Composition):** For sequential tests with Rényi divergences bounded by $\rho_i$:

$$ D_{\alpha}\left(\prod_{i=1}^{t} P_i || \prod_{i=1}^{t} Q_i\right) \leq \sum_{i=1}^{t} \rho_i $$

This enables tight privacy composition in sequential settings.

### Adaptive Composition

**Theorem (Adaptive E-process Composition):** Let $(E_t^{(i)})$ be E-processes for hypotheses $H_0^{(i)}$. The adaptively weighted combination:

$$ E_t = \sum_{i=1}^{k} w_i(t) E_t^{(i)} $$

remains an E-process if $\sum_{i=1}^{k} w_i(t) \leq 1$ and weights are predictable.

### Optimal Composition Rates

For $k$ independent sub-Gaussian tests with variance proxy $\sigma^2$:

$$ E_{\text{composed}} = \exp\left(\frac{\|S\|_2^2}{2k\sigma^2} - \frac{k\log(k)}{2}\right) $$

where $S = (S_1, \ldots, S_k)$ are the individual test statistics.

## Practical Implementation of Composition Theorems

### Algorithm 1: Sequential Privacy Accounting

```
Initialize: E_0 = 1, privacy_budget = ε_total
For each query t = 1, 2, ...:
    1. Compute privacy loss Z_t for query t
    2. Update E-process:
       E_t = E_{t-1} · exp(λZ_t - ψ(λ))
    3. Check stopping condition:
       if E_t ≥ 1/α:
           STOP (privacy budget exceeded)
    4. Compute remaining budget:
       ε_remaining = (log(1/α) - log(E_t))/λ
```

### Algorithm 2: Adaptive Sequential Testing

```
Initialize: E_0^{(i)} = 1 for all hypotheses i
For each observation t = 1, 2, ...:
    1. Compute likelihood ratios L_t^{(i)}
    2. Update E-processes:
       E_t^{(i)} = E_{t-1}^{(i)} · L_t^{(i)}
    3. Compute composite E-value:
       E_t = min_i {k · E_t^{(i)}} (Bonferroni)
       or
       E_t = (∑_i (E_t^{(i)})^{1/r})^r (r-composition)
    4. Make decision:
       if E_t ≥ 1/α:
           REJECT H_0
```

### Algorithm 3: Confidence Sequence Construction

```
For each time t:
    1. Define E-process for each θ:
       E_t(θ) = exp(λ(S_t - tθ) - tψ_θ(λ))
    2. Optimize λ = λ_t(θ):
       λ_t(θ) = argmax_λ {λ(S_t - tθ) - tψ_θ(λ)}
    3. Construct confidence set:
       C_t = {θ : E_t(θ) < 1/α}
    4. Simplify to interval [L_t, U_t] via monotonicity
```

## Practical Considerations and Best Practices

### Choosing the Right E-process

1. **For bounded observations**: Use exponential tilting with mixture priors
2. **For unbounded observations**: Use truncation or robust E-values
3. **For composite nulls**: Use reverse information projection (RIPr)

### Optimizing Composition

**Strategy 1: Hierarchical Testing**
Organize hypotheses hierarchically and use:

$$ E_{\text{tree}} = \min_{v \in \text{tree}} \{|A(v)| \cdot E_v\} $$

where $|A(v)|$ is the number of ancestors of node $v$.

**Strategy 2: Weighted Composition**
For hypotheses with different importance $w_i$:

$$ E_{\text{weighted}} = \left(\sum_{i=1}^{k} w_i E_i^{1/r}\right)^r $$

Choose $r$ based on correlation structure.

### Computational Efficiency

1. **Use logarithmic representation**: Work with $\log E_t$ to avoid numerical overflow
2. **Batch updates**: Update E-processes in blocks for vectorization
3. **Pruning**: Remove hypotheses with $E_t^{(i)} < \epsilon$ from active set

### Handling Dependencies

For dependent observations, modify the E-process:

$$ E_t^{\text{dep}} = \exp\left(\lambda S_t - t\psi(\lambda) - \frac{\lambda^2 \hat{V}_t}{2}\right) $$

where $\hat{V}_t$ estimates the variance accounting for dependence.

## Advanced Topics

### Asymptotic Optimality

The **growth rate optimal** E-process satisfies:

$$ \lim_{t \to \infty} \frac{\log E_t}{t} = D_{KL}(P_1 || P_0) $$

achieving the Kullback-Leibler divergence asymptotically.

### Connection to Information Theory

The **e-power** of a test:

$$ \text{POW}_e = \inf_{P \in H_1} \mathbb{E}_P[\log^+ E] $$

relates to the information projection of $H_1$ onto $H_0$.

### Future Directions

1. **Nonparametric E-processes**: Kernel-based methods for complex hypotheses
2. **Causal E-values**: Sequential testing in causal inference
3. **Distributed testing**: E-processes in federated settings
4. **Quantum E-values**: Extension to quantum hypothesis testing

## Conclusion

Sequential testing via E-processes provides a powerful framework for anytime-valid inference with strong connections to differential privacy. The composition theorems enable practical implementation in complex settings while maintaining theoretical guarantees. Key advantages include:

- **Anytime validity**: Can stop at any time without alpha inflation
- **Optimal composition**: Tight bounds via Rényi divergence
- **Privacy accounting**: Natural connection to CDP mechanisms
- **Practical algorithms**: Efficient implementation strategies

The framework continues to evolve with applications in online experimentation, privacy-preserving analytics, and adaptive clinical trials.
