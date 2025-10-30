Title: Martingale Processes & Moment Generating Functions: Mathematical Summary
Date: 2024-10-28
Category: blog
Tags: Statistical Theory
Status: published

## Core Martingale Framework

### Definition

A stochastic process $\{M_t\}$ is a martingale with respect to filtration $\{\mathcal{F}_t\}$ if:

1. **Adapted**: $M_t$ is $\mathcal{F}_t$-measurable for all $t$
2. **Integrable**: $\mathbb{E}[|M_t|] < \infty$ for all $t$
3. **Martingale Property**: $\mathbb{E}[M_t | \mathcal{F}_{t-1}] = M_{t-1}$

### Wealth Process (Test Martingale)

For sequential testing, we construct a nonnegative martingale:

$$K_t = \prod_{i=1}^t [1 + \lambda_i(X_i - \mu_0)]$$

Where:

- $X_i$: Observations (sensor readings)
- $\mu_0$: Expected value under null hypothesis $H_0$
- $\lambda_i$: Predictable betting strategy ($|\lambda_i| \leq 1$ for boundedness)

### Ville's Inequality

For any nonnegative martingale with $K_0 = 1$:

$$P\left(\sup_{t \geq 1} K_t \geq C\right) \leq \frac{1}{C}$$

This provides anytime-valid Type I error control at level $\alpha = 1/C$.

## Moment Generating Functions (MGFs) in Sequential Testing

### MGF Definition

For random variable $X$, the MGF is:

$$M_X(\lambda) = \mathbb{E}[e^{\lambda X}]$$

### Key Properties for Testing

1. **Chernoff Bound**:
   $$P(X \geq a) \leq \min_{\lambda \geq 0} e^{-\lambda a} M_X(\lambda)$$

2. **Exponential Martingale Construction**:
   If $X_t$ has MGF $M_t(\lambda)$ under $H_0$, then:
   $$Z_t(\lambda) = \frac{e^{\lambda X_t}}{M_t(\lambda)}$$
   is a martingale under $H_0$.

### Connection to Betting Framework

The wealth process can be derived from MGF considerations:

$$K_t = \prod_{i=1}^t \frac{e^{\lambda_i X_i}}{\mathbb{E}_{H_0}[e^{\lambda_i X_i}]}$$

This is the likelihood ratio martingale for exponential families.

## Multi-State Extension via Mixture Martingales

### Composite Null Hypothesis

For multiple unhealthy states $U_1, U_2, \ldots, U_k$:

$$N_t^{\text{mix}} = \sum_{k=1}^K w_k N_t^{(k)}$$

Where:

- $N_t^{(k)} = \prod_{i=1}^t \frac{p_{\text{healthy}}(X_i)}{p_{U_k}(X_i)}$ (likelihood ratio martingale)
- $w_k > 0$, $\sum w_k = 1$ (mixture weights)

### Properties

- $N_t^{\text{mix}}$ is a martingale under composite null $H_0$: System is in some $U_k$
- Ville's inequality applies: $P(\sup_t N_t^{\text{mix}} \geq C) \leq 1/C$

## Stopping Times & Decision Rules

### Health Declaration Stopping Time

$$\tau_{\text{healthy}} = \inf \{ t \geq 0 : N_t \geq C \}$$

Where $C = 1/\alpha$ controls false healthy declaration rate.

### Expected Stopping Times

For a martingale $M_t$ and stopping time $\tau$:

- **Optional Stopping Theorem**: If conditions satisfied, $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$
- **Wald's Equation**: For random walk-like processes

## Practical Implementation Formulas

### Betting Strategy Selection

**Kelly Criterion** (for known alternative):

$$\lambda^* = \frac{\mu_1 - \mu_0}{\sigma^2}$$

**Conservative Hedging** (for uncertainty):

$$\lambda_t = \min\left(\frac{c}{\sqrt{t}}, \lambda_{\max}\right)$$

### Mixture Weights Optimization

Using prior knowledge or empirical Bayes:

$$w_k \propto \pi_k \cdot \text{BayesFactor}_{k}$$

## Key Theoretical Results

1. **Martingale Convergence**

   If $\{M_t\}$ is a nonnegative martingale, then $M_t \to M_\infty$ almost surely.

2. **Uniform Integrability**

   Implies $L^1$ convergence and validity of optional stopping.

3. **Test Martingale Construction**

   Any nonnegative martingale with $M_0 = 1$ can serve as a valid sequential test.

## Application to Sensor Monitoring

### Feature Transformation

For sensor data $Y_t$, use transformed features:

$$X_t = \phi(Y_t) - \mathbb{E}_{H_0}[\phi(Y_t)]$$

Where $\phi$ is a bounded function maintaining martingale property.

### Multi-Sensor Extension

For correlated sensors $Y_t^{(1)}, \ldots, Y_t^{(d)}$:

$$K_t = \prod_{i=1}^t \left[1 + \sum_{j=1}^d \lambda_{i,j}(Y_i^{(j)} - \mu_0^{(j)})\right]$$

## Summary of Mathematical Guarantees

1. **Error Control**: Ville's inequality provides uniform Type I error control
2. **Anytime-Validity**: Results valid at all stopping times
3. **Flexibility**: Works with various betting strategies and mixture constructions
4. **Efficiency**: Early stopping when evidence is strong
5. **Robustness**: Mixture methods handle model uncertainty

This mathematical framework provides the rigorous foundation for the sequential health monitoring system, ensuring statistical validity while enabling continuous, adaptive monitoring.
