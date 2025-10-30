# Composition Rules for E-value Based Sequential Testing

E-values provide elegant composition properties that make them particularly useful for combining multiple tests. Here are the key composition rules with their derivations:

## 1. Product Rule (Intersection of Nulls)

For testing the intersection of null hypotheses $H_0 = H_0^{(1)} \cap H_0^{(2)} \cap \cdots \cap H_0^{(k)}$:

**Rule:** If $E_1, E_2, \ldots, E_k$ are e-values for $H_0^{(1)}, H_0^{(2)}, \ldots, H_0^{(k)}$ respectively, then:

$$E = \prod_{i=1}^k E_i$$

is a valid e-value for $H_0$.

**Derivation:** Under $H_0$, all individual nulls are true, so $\mathbb{E}[E_i] \leq 1$ for all $i$. By independence or conditional independence:

$$\mathbb{E}[E] = \mathbb{E}\left[\prod_i E_i\right] = \prod_i \mathbb{E}[E_i] \leq 1$$

## 2. Weighted Average Rule (Same Hypothesis)

For combining multiple e-values testing the same hypothesis:

**Rule:** If $E_1, E_2, \ldots, E_k$ are e-values for the same $H_0$ with weights $w_i \geq 0$, $\sum w_i = 1$:

$$E = \sum_{i=1}^k w_i E_i$$

is a valid e-value for $H_0$.

**Derivation:** By linearity of expectation under $H_0$:

$$\mathbb{E}[E] = \sum_i w_i \mathbb{E}[E_i] \leq \sum_i w_i = 1$$

## 3. Sequential Composition (Continuation Testing)

For sequential testing where we decide whether to continue based on previous results:

**Rule:** If $E_1$ is an e-value and $E_2$ is computed on independent or future data:

$$E = E_1 \cdot \mathbf{1}\{E_1 > c\} + E_1 \cdot E_2 \cdot \mathbf{1}\{E_1 \leq c\}$$

for any threshold $c > 0$, is a valid e-value.

**Derivation:** The key is that the decision to continue $(E_1 \leq c)$ is a stopping time. By optional stopping theorem for test martingales:

$$\mathbb{E}[E|H_0] \leq \mathbb{E}[E_1] \leq 1$$

## 4. Calibration Rule (Power Enhancement)

To recalibrate an e-value for better power while maintaining validity:

**Rule:** For any non-decreasing function $f$ with $f(0) = 0$ and $f(x)/x$ non-increasing:

$$E' = f(E)$$

is a valid e-value if $E$ is.

**Derivation:** By Markov's inequality, $P(E \geq t) \leq 1/t$ under $H_0$. The transformed e-value satisfies:

$$\mathbb{E}[f(E)] \leq f(1) \leq 1$$

## 5. Maximum Rule (Union of Nulls)

For testing at least one hypothesis:

**Rule:** If $E_1, \ldots, E_k$ are e-values for $H_0^{(1)}, \ldots, H_0^{(k)}$:

$$E = \min(k, \max_i E_i)$$

provides family-wise error control.

**Derivation:** By union bound under global null:

$$P(\max_i E_i \geq \alpha k) \leq \sum_i P(E_i \geq \alpha k) \leq \frac{k}{\alpha k} = \frac{1}{\alpha}$$

## 6. e-BH Rule (Multiple Testing with FDR)

For false discovery rate control:

**Rule:** Given e-values $E_1, \ldots, E_k$, reject hypotheses with:

$$E_i \geq \frac{k}{\alpha \cdot |\{j: E_j \geq E_i\}|}$$

This controls FDR at level $\alpha$.

## 7. Mixture Rule (Composite Nulls)

For composite null hypotheses:

**Rule:** If $E_\theta$ is an e-value for each simple null $H_\theta$ and $\pi$ is a prior over $\theta$:

$$E = \int E_\theta \, d\pi(\theta)$$

is valid for the composite null.

**Derivation:** For any $\theta$ in the composite null:

$$\mathbb{E}_\theta[E] = \mathbb{E}_\theta\left[\int E_{\theta'} \, d\pi(\theta')\right] = \int \mathbb{E}_\theta[E_{\theta'}] \, d\pi(\theta') \leq 1$$

## 8. Data Splitting Rule

For using the same data for selection and testing:

**Rule:** Split data into $D_1$ for selection and $D_2$ for testing. If $E(D_2; S)$ is an e-value for each selected set $S$:

$$E = E(D_2; S(D_1))$$

is valid where $S(D_1)$ is the selection based on $D_1$.

## Key Properties

These composition rules leverage:
- The **martingale property** of e-processes
- The fact that **expectations of e-values are bounded by 1** under the null
- The **optional stopping theorem** for sequential procedures
- The **closure under mixtures** property

These properties make e-values particularly powerful for complex testing scenarios involving:
- Sequential testing with early stopping
- Multiple testing with various error rate controls
- Adaptive testing procedures
- Meta-analysis and evidence combination

## References and Further Reading

The mathematical foundation for these rules comes from:
- Ville's inequality and test martingales
- The connection between e-values and betting scores
- The duality between e-values and p-values through the Markov inequality
- Cumulant generating functions and exponential families

For rigorous proofs and extensions, see work on game-theoretic probability and safe, anytime-valid inference.
