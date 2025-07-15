# **C.A.S.T.L.E.**

#### Component Assembly Structure Tracking for Learning Emergence

***
## Mathematical Model of Assembly Tracking Neural Network with Atomic Codes

## 1. Weight Tensor Partitioning

Let $W \in \mathbb{R}^{m \times n}$ be the weight matrix of a neural network layer, where:
- $m$: number of input features,
- $n$: number of output features.

Partition $W$ into non-overlapping blocks (molecules) of size $k \times k$:

$$W = \bigcup\_{i=1}^{M} \bigcup\_{j=1}^{N} W\_{i,j}$$

where:
- $W\_{i,j}$: block (molecule) at position $(i, j)$,
- $M = \lfloor m/k \rfloor$: number of blocks along rows,
- $N = \lfloor n/k \rfloor$: number of blocks along columns,
- $k$: molecule (block) size.

## 2. Atomic Code Assignment

For each molecule $W\_{i,j}$, compute its average value:

$$\mu\_{i,j} = \frac{1}{k^2} \sum\_{a=1}^{k} \sum\_{b=1}^{k} W\_{i,j}[a, b]$$

where:
- $\mu\_{i,j}$: average value of molecule $W\_{i,j}$,
- $a, b$: indices within the block.

Assign an atomic code $S\_{i,j}$ using a discretization function $f$:

$$S\_{i,j} = f(\mu\_{i,j})$$

where:
- $S\_{i,j}$: atomic code assigned to molecule $W\_{i,j}$,
- $f$: function mapping average value to a symbol (e.g., 'Ze', 'Sm', 'Md').

## 3. Lattice Construction

The set of atomic codes forms a molecular lattice $L$:

$$L = \{ S\_{i,j} \mid 1 \leq i \leq M, 1 \leq j \leq N \}$$

where:
- $L$: molecular lattice,
- $S\_{i,j}$: atomic code at position $(i, j)$.

## 4. Assembly Index Calculation

Define the assembly index $A(L)$ as the minimal number of unique atomic codes or assembly steps required to construct $L$:

$$A(L) = \min \left( \text{number of steps to assemble } L \text{ from atomic codes and reused sub-lattices} \right)$$

where:
- $A(L)$: assembly index of lattice $L$,
- "steps": count of unique codes and reused patterns needed for construction.

## 5. Complexity Reward in Loss Function

During training, modify the loss function to reward or penalize assembly complexity:

$$\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{base}} - \lambda \cdot R(A(L))$$

where:
- $\mathcal{L}\_{\text{total}}$: total loss,
- $\mathcal{L}\_{\text{base}}$: standard loss (e.g., binary cross-entropy),
- $R(A(L))$: reward function based on assembly index,
- $\lambda$: hyperparameter controlling reward strength.

## 6. Gradient Modification

Scale gradients for each molecule according to reuse and complexity:

$$\nabla W\_{i,j} \leftarrow \nabla W\_{i,j} \cdot g(S\_{i,j}, A(L))$$

where:
- $\nabla W\_{i,j}$: gradient for molecule $W\_{i,j}$,
- $g$: modifier function (e.g., reduces updates for highly reused molecules),
- $S\_{i,j}$: atomic code,
- $A(L)$: assembly index.

## 7. Architecture Evolution

If assembly index exceeds a threshold, suggest architectural changes:

$$\text{If } A(L) > T, \text{ then modify layer sizes or add/remove layers}$$

where:
- $T$: complexity threshold for architectural evolution,
- $A(L)$: assembly index.

## 8. System Complexity

### 8.1 Assembly Theory Metric

$$A\_{\text{sys}}^{(t)} = \frac{1}{|\mathcal{P}|} \sum\_{i=1}^{|\mathcal{P}|} e^{a\_i} \cdot \frac{n\_i - 1}{|\mathcal{P}|}$$

Where:  
- $A\_{\text{sys}}^{(t)}$: System assembly complexity at time $t$  
- $|\mathcal{P}|$: Population size  
- $a\_i$: Assembly complexity of module $i$  
- $n\_i$: Copy number of module type $i$  

## 9. Refined Assembly Theory Metric for Neural Networks

To better capture the modularity and reuse in neural networks, we refine the assembly theory metric by explicitly accounting for the diversity and recurrence of molecular patterns (atomic codes) in the lattice:

Let $\mathcal{S} = \{ S\_{i,j} \}$ be the set of atomic codes in lattice $L$, and let $u\_s$ be the number of unique codes, $n\_s$ the copy number of code $s$, and $a\_s$ the assembly complexity of code $s$.

Define the refined system assembly complexity as:

$$A\_{\text{sys}}^{(t)} = \frac{1}{u\_s} \sum\_{s \in \mathcal{S}} e^{a\_s} \cdot \frac{n\_s - 1}{|\mathcal{S}|}$$

where:
- $u\_s$: number of unique atomic codes in $L$,
- $n\_s$: number of occurrences of code $s$ in $L$,
- $a\_s$: assembly complexity of code $s$ (e.g., minimal steps to construct $s$ from primitives),
- $|\mathcal{S}|$: total number of molecules in $L$.

**Interpretation:**  
This metric rewards reuse (high $n\_s$) and penalizes diversity (high $u\_s$), while weighting by the intrinsic complexity $a\_s$ of each code. The exponential term amplifies the impact of complex codes, and the normalization by $u\_s$ and $|\mathcal{S}|$ ensures comparability across networks.

**Application in Training:**  
- Use $A\_{\text{sys}}^{(t)}$ as a regularizer in the loss function to promote modularity and efficient reuse.
- Track $A\_{\text{sys}}^{(t)}$ over epochs to monitor the evolution of network complexity and modularity.
- Suggest architectural changes if $A\_{\text{sys}}^{(t)}$ exceeds a threshold, indicating excessive diversity or insufficient reuse.

## 10. Complexity-Accuracy Relationship

Let $\mathcal{A}(t)$ denote the model accuracy at time $t$, and $A\_{\text{sys}}^{(t)}$ represent the system assembly complexity. The relationship between accuracy and complexity can be modeled as:

$$\mathcal{A}(t) = \mathcal{A}\_{\text{base}} + \gamma \cdot \left(1 - e^{-\beta \cdot A\_{\text{sys}}^{(t)}}\right)$$

where:
- $\mathcal{A}\_{\text{base}}$: base accuracy achievable with minimal complexity,
- $\gamma$: maximum potential accuracy improvement from complexity,
- $\beta$: rate parameter controlling how quickly complexity benefits manifest.

This formulation captures three key properties:
1. As $A\_{\text{sys}}^{(t)} \to 0$, $\mathcal{A}(t) \to \mathcal{A}\_{\text{base}}$,
2. As $A\_{\text{sys}}^{(t)} \to \infty$, $\mathcal{A}(t) \to \mathcal{A}\_{\text{base}} + \gamma$,
3. The marginal benefit of additional complexity diminishes as complexity increases, reflecting the principle of diminishing returns.

### Alternative Formulation for Small Complexity Ranges

For practical application with small ranges of complexity, a linear approximation may be suitable:

$$\mathcal{A}(t) \approx \mathcal{A}\_{\text{min}} + \delta \cdot A\_{\text{sys}}^{(t)}$$

where:
- $\mathcal{A}\_{\text{min}}$: minimum accuracy with zero complexity,
- $\delta$: linear rate of accuracy improvement with complexity.

## 11. Empirical Validation

This relationship can be validated by:
1. Measuring $A\_{\text{sys}}^{(t)}$ and $\mathcal{A}(t)$ at multiple training epochs,
2. Fitting the model parameters ($\mathcal{A}\_{\text{base}}$, $\gamma$, $\beta$) or ($\mathcal{A}\_{\text{min}}$, $\delta$),
3. Testing predictive power on held-out model configurations.

Based on experimental validation, two distinct relationship patterns emerge:

### 11.1 Empirical Complexity-Accuracy Relationships (Iris)
Pattern 1: Quadratic Relationship with Optimal Complexity 

$$
\mathcal{A}(t) = a + b \cdot A_{\text{sys}}^{(t)} - c \cdot \left(A_{\text{sys}}^{(t)}\right)^2
$$ 

where:

Optimal complexity occurs at $A_{\text{opt}} = \frac{b}{2c}$
Observed for Iris dataset: $A_{\text{opt}} = 1.87$, $R^2 = 0.9989$

### 11.2 Empirical Complexity-Accuracy Relationships (Breast Cancer)
Pattern 2: Complexity-Independent Accuracy 

$$
\mathcal{A}(t) = \mathcal{A}{\text{const}} + \epsilon \cdot A{\text{sys}}^{(t)}
$$ 

where:

$\epsilon \approx 0$ (coefficient near zero)
Observed for Breast Cancer dataset: $\epsilon = -0.0000$, $R^2 = 1.0000$

### 11.3 Dataset-Dependent Complexity Effects
The relationship between assembly complexity and model performance is dataset-dependent:

- Iris Dataset: Exhibits clear optimal complexity where performance is maximized
- Breast Cancer Dataset: Shows complexity-independent performance
- General Pattern: Complex relationships require empirical validation per dataset

There is meaningful relationships that can be describe depending on hyper-parameters as well as model training standards. The baseline relationship shows a consistent trend and is worth investigating due to the significant performance increase compared to baseline neural nets in terms of f1-score and accuracy across classes. 


---

This mathematical formalism describes the assembly tracking neural network, which partitions weight tensors into molecular blocks and assigns atomic codes based on block statistics. The resulting molecular lattice encodes the network's structural motifs, enabling calculation of an assembly index that quantifies modularity and reuse. During training, the loss function incorporates a complexity reward or penalty derived from the assembly index, and gradients are modulated according to molecular reuse. If the assembly index exceeds a threshold, the architecture is evolved to favor more efficient or interpretable structures. This approach bridges neural network optimization with principles from assembly theory, promoting modularity, interpretability, and adaptive architectural evolution.
