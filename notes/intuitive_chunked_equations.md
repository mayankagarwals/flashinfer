# Chunked Gated Delta Rule: Intuitive Derivation

**Prerequisite:** [intuitive_recurrence_relation.md](intuitive_recurrence_relation.md)
— understand every line of the recurrent loop before reading this.

## Reference: The Recurrent Loop (ground truth)

This is what the chunked algorithm must reproduce, but in parallel:

```python
for t in range(C):                          # C = chunk size (128)
    h = exp(g_t) * h                        # 1. decay old state
    prediction = k_t @ h                     # 2. (d_k,) @ (d_k, d_v) = (d_v,) soft lookup
    delta = v_t - prediction                 # 3. what's new?
    h = h + k_t^T @ (beta_t * delta)        # 4. write correction into state
    o_t = (scale * q_t) @ h                  # 5. query reads from state
```

## The Chunked Equations (all together)

The chunked algorithm replaces the sequential loop with these parallel operations,
executed once per chunk of C=128 tokens:

```
# Precompute decays
G     = exp(cumsum(g))                                    (C, 1)
Gamma = exp(cumsum(g) - cumsum(g).T)                      (C, C)

# Solve the coupling (the expensive part — matrix inversion)
T = inv(I + strictLower(B * Gamma * (K @ K.T)))           (C, C)

# Split the corrected deltas into value and prediction parts
U  = T @ (B * V)                                          (C, d)
W  = T @ (B * G * K)                                      (C, d)
V' = U - W @ S                                            (C, d)

# Output: inter-chunk (from state) + intra-chunk (from this chunk's deltas)
M' = (1/G) * M                                            (C, C)
O  = G * (Q @ S  +  (Q @ K.T * M') @ V')                 (C, d)

# State update: carry forward to next chunk
S  = G[-1] * (S + K.T @ ((1/G) * V'))                     (d_k, d_v)
```

Each line is explained in detail below.

---

## Shapes

All tensors are within one chunk. Single head, single batch for clarity.

```
Q, K        → (C, d)      queries, keys
V           → (C, d)      values
S           → (d_k, d_v)  recurrent state (the associative memory)
B (beta)    → (C, 1)      per-token update gate (how aggressively to correct)
G           → (C, 1)      cumulative gate = exp(cumsum(g)), per-token decay
Gamma       → (C, C)      pairwise decay: Gamma[i,j] = G[i]/G[j]
M           → (C, C)      causal mask (lower triangular 1s)
M'          → (C, C)      gate-scaled causal mask = M / G
```

---

## Equation 1: Precompute Decays

```
G     = exp(cumsum(g))                     (C, 1)
Gamma = exp(cumsum(g) - cumsum(g).T)       (C, C)
```

In the recurrent loop, every token decays the state before using it. By the time
token t acts, the original state S has been decayed by all gates g_0 through g_t.
That cumulative decay is `G[t]`.

When token i looks at token j's write (j < i), that write has decayed by the
gates between j and i. That pairwise decay is `Gamma[i,j] = G[i]/G[j]`.


---

## Equation 2: The Coupling Matrix and its Inverse

```
T = inv(I + strictLower(B * Gamma * (K @ K.T)))       (C, C)
```

This is the heart of the chunked algorithm. Here's why it's needed.

Look at the recurrent loop carefully — specifically lines 2, 3, and 4:

```python
    prediction = k_t @ h                     # 2. (d_k,) @ (d_k, d_v) = (d_v,) soft lookup
    delta = v_t - prediction                 # 3. what's new?
    h = h + k_t^T @ (beta_t * delta)        # 4. write correction into state
```

Line 4 **changes h**. Then on the next iteration, line 2 reads from this **changed h**.
That means token 1's prediction depends on token 0's write. Token 2's prediction
depends on token 0's AND token 1's writes. And so on.

Let's trace this for 3 tokens. Start with state S (before this chunk).

**Token 0:**
```python
    h = exp(g_0) * S                                      # decay
    prediction_0 = k_0 @ h                                  # line 2: (d_k,) @ (d_k,d_v) = (d_v,)
    delta_0 = v_0 - prediction_0                           # line 3: compute delta
    h = h + k_0^T @ (beta_0 * delta_0)                    # line 4: WRITE TO STATE
```

**Token 1:**
```python
    h = exp(g_1) * h                                       # decay (h now includes token 0's write!)
    prediction_1 = k_1 @ h                                  # line 2: reads token 0's write
    delta_1 = v_1 - prediction_1                           # line 3: delta DEPENDS on token 0's write
    h = h + k_1^T @ (beta_1 * delta_1)                    # line 4: WRITE TO STATE
```

**Token 2:**
```python
    h = exp(g_2) * h                                       # decay (h includes token 0 AND 1's writes!)
    prediction_2 = k_2 @ h                                  # line 2: reads both writes
    delta_2 = v_2 - prediction_2                           # line 3: depends on BOTH previous writes
    h = h + k_2^T @ (beta_2 * delta_2)                    # line 4: WRITE TO STATE
```

The deltas are **coupled** — each depends on all previous ones through the
state modifications on line 4.

**The naive parallel attempt (ignoring coupling):**

If we try to compute all deltas in parallel, we'd use the original state for
every prediction:


Refresher:
- `G[i] = exp(cumsum(g[:i]))` is the cumulative product of all gates up to i
- `k_i @ S` is the prediction for token i

```
naive_prediction_0 = G[0] * k_0 @ S     ← correct (no prior writes)
naive_prediction_1 = G[1] * k_1 @ S     ← WRONG (misses token 0's write to h on line 4)
naive_prediction_2 = G[2] * k_2 @ S     ← WRONG (misses token 0 and 1's writes on line 4)

naive_deltas = V - G * (K @ S)          ← only token 0's delta is correct
```

**What's missing?** Token 1's prediction should include the effect of token 0's
write (line 4). Let's work out what that correction looks like:

Token 0's write (line 4) added this outer product into h:

```
write_0 = k_0^T ⊗ (beta_0 * delta_0)       (d_k, 1) @ (1, d_v) = (d_k, d_v)
```

When token 1 reads the state with `k_1 @ h` (line 2), it picks up an extra
term from that write. Let's trace the shapes carefully:

```
extra = k_1 @ write_0
      = k_1 @ (k_0^T ⊗ (beta_0 * delta_0))
        ────   ────────────────────────────
        (d_k,) @       (d_k, d_v)           = (d_v,)    
```

This is just a soft lookup (exactly like `k @ S`!) — token 1's key reads from
the matrix that token 0 wrote. We can expand it:

```
k_1 @ (k_0^T ⊗ (beta_0 * delta_0))

= (k_1 · k_0) * beta_0 * delta_0           (d_v,)
  ──────────   ──────   ────────
  key similarity  write    what was
  (scalar dot     strength  written
   product)                 (d_v,)
```

This is very important to understand.
Why does the dot product appear? It's the **same soft lookup mechanism** from
Line 2, just applied to a smaller state.

In Line 2, the full soft lookup reads from the entire accumulated state S:

```
k_t @ S_full = k_t @ (k_a^T ⊗ v_a + k_b^T ⊗ v_b + ...)
             = (k_t · k_a) * v_a  +  (k_t · k_b) * v_b  + ...
               ──────────────────    ──────────────────
               dot product weights each stored value
```

Here we're doing the exact same operation, but S only contains **one write** —
token 0's write. It's a rank-1 state with a single association stored:

```
S_single_write = k_0^T ⊗ (beta_0 * delta_0)       (d_k, d_v)
```

When token 1 reads from this single-write state:

```
k_1 @ S_single_write = (k_1 · k_0) * beta_0 * delta_0
```

Same mechanism — the dot product `k_1 · k_0` determines the weight, measuring
how much k_1 overlaps with k_0's direction. With many writes you get a sum of
dot-product-weighted values; with one write you get a single weighted value.

This is the part that requires a little thinking. The way we see S is an accumulated state over multiple outer products. Each outer product `(d_k, 1) @ (1, d_v)` stores the `(d_v,)` value vector scaled by each key dimension. So a key could be 0 in x axis but 1 in y axis, it will store the second value row scaled by 1 while the first row would be scaled by 0. So when another key that comes along in a similar ratio of dimensions (which we call direction), it will query the value rows in the same proportion resulting in the same stored value. 
We think of this prediction as a 1D vector multiplying a state matrix that contains already value rows stored according to ratio of k dimensions 
But if you get the k out of the equation like above, it can be seen as dot product of the keys multiplying simply the value row and giving the same prediction
This dot product scaled (scalar) tells you how much j token's delta is going to bother token i's delta

If they're orthogonal, the extra term is zero (token 0's write doesn't affect
token 1's prediction at all). If they point the same way, the full effect comes
through.

With gates factored in (token 0's write decays by Gamma[1,0] before token 1 sees
it), the correction becomes:

```
correction_to_1_from_0 = (k_1 · k_0) * Gamma[1,0] * beta_0 * delta_0
```

For token 2, there are TWO corrections — one from token 0 and one from token 1:

```
correction_to_2_from_0 = (k_2 · k_0) * Gamma[2,0] * beta_0 * delta_0
correction_to_2_from_1 = (k_2 · k_1) * Gamma[2,1] * beta_1 * delta_1
```

But crucially, `delta_1` itself was affected by `delta_0` (because token 1's
prediction was shifted by token 0's write). So the corrections form a chain.

Writing this as a matrix system — the true deltas must satisfy:

```
delta_0 = naive_delta_0
delta_1 = naive_delta_1 - (k_1·k_0)*Gamma[1,0]*B[0] * delta_0
delta_2 = naive_delta_2 - (k_2·k_0)*Gamma[2,0]*B[0] * delta_0
                        - (k_2·k_1)*Gamma[2,1]*B[1] * delta_1
```

Rearranging (moving delta terms to the left side), we get a matrix system:

```
                    coupling matrix                     true deltas    naive deltas

┌                                                  ┐   ┌        ┐     ┌              ┐
│  1                       0                    0  │   │delta_0 │     │naive_delta_0 │
│                                                  │   │        │     │              │
│  (k₁·k₀)*Γ[1,0]*B[0]     1                    0  │ × │delta_1 │  =  │naive_delta_1 │
│                                                  │   │        │     │              │
│  (k₂·k₀)*Γ[2,0]*B[0]     (k₂·k₁)*Γ[2,1]*B[1]   1 │   │delta_2 │     │naive_delta_2 │
└                                                  ┘   └        ┘     └              ┘
```

Each entry `[i, j]` in the coupling matrix has a clear meaning:
- `(kᵢ · kⱼ)` — key similarity: how much does token j's write affect token i's lookup? If you understood the soft lookup example, this should feel intuitive.
- `Γ[i,j]` — decay: how much has token j's write faded by the time token i reads?
- `B[j]` — beta: how strong was token j's write?

The matrix is `I + strictLower(B * Gamma * K @ K.T)`. Each off-diagonal entry [i, j]
represents: "how much does token j's write (line 4) affect token i's prediction
(line 2), accounting for key similarity, gating, and write strength?"

It's lower-triangular because token i's prediction (line 2) can only be affected
by writes (line 4) from tokens j < i — causality.

```
(I + strictLower(B * Gamma * K @ K.T)) @ true_deltas = naive_deltas
```

**T is the inverse.** It converts naive deltas → true deltas in one matrix multiply:

```
true_deltas = T @ naive_deltas
```

At risk  of repetition
naive_deltas = G[i] * k_i @ S which is the prediction for token i but scaled by the gate
true_deltas include the effect of all previous writes on the prediction for token i. This effect is governed by the dot product of current key against the previous key, the decay of the previous write, and the beta of the previous write.
It is very crucial to understand why the dot product helps. If at this point it is not intuitive, I invite the reader to revisit the blog till here before moving forward.

## Equation 3: U, W, and V' — Splitting the Delta

The naive delta for each token is `naive_delta_t = v_t - G[t] * k_t @ S`. The
true delta resolves the coupling: `true_deltas = T @ naive_deltas`.

We can split this by linearity of T. The naive delta has two parts — the value
and the prediction — and T acts on each independently:

```
true_deltas = T @ (B * V   -   B * G * K @ S)
            = T @ (B * V)  -  T @ (B * G * K) @ S
              ───────────     ────────────────────
              U                W @ S
```

This gives us:

```
U  = T @ (B * V)            (C, C) @ (C, d) = (C, d)
W  = T @ (B * G * K)        (C, C) @ (C, d) = (C, d)
V' = U - W @ S              the true corrected deltas
```

**Important: `v_t` itself doesn't change.** The values are known inputs. But U is
not just `beta * V` — it's `T @ (beta * V)`. Why does T need to act on the values
at all, if they're fixed?

Because each token's value contributes to its delta, and that delta propagates
through the coupling chain to affect later tokens. Concretely, for 2 tokens:

```
true_delta_0 = beta_0 * (v_0 - G[0]*k_0 @ S)           no coupling
true_delta_1 = beta_1 * (v_1 - G[1]*k_1 @ S)
             - (k₁·k₀)*Γ*B[0] * true_delta_0           coupling from token 0
```

Expand the coupling term:

```
coupling = -(k₁·k₀)*Γ*B[0] * beta_0 * (v_0 - G[0]*k_0 @ S)
                                         ───
                                         v_0 appears here!
```

Token 0's value `v_0` shows up in token 1's true delta — not because `v_0` changed,
but because `v_0` contributed to `delta_0`, which modified the state, which shifted
`prediction_1`. U = `T @ (B * V)` traces this cascade: how each token's value
propagates through the chain of deltas.

**W = the prediction side of the same split.** Each token predicts from the state
using `beta * G * k` (key direction, scaled by gate and write strength). T resolves
the coupling on this side: earlier predictions shift later predictions. W is the
effective prediction direction per token.

**Both use the same T** — the coupling structure is identical for the value and
prediction sides of the delta. This is the shared-inverse optimization: one
inversion, two GEMMs, instead of two inversions.

---

## Equation 4: The Net Delta

```
V' = U - W @ S             (C, d) - (C, d) @ (d_k, d_v) = (C, d)
```

This is the chunk-level equivalent of `delta = v - prediction` from the recurrent
loop, but for all 128 tokens at once, with all interdependencies resolved.

- **U** — what the tokens want to write (corrected for coupling)
- **W @ S** — what the state already provides for the corrected directions

**V' is the net new information this chunk contributes to the state.**

`W @ S` works because S is an associative memory: each row of W is a (d_k,)
direction, and `W @ S` does a soft lookup — "for each corrected write direction,
what value does the state already predict?" The lookup is weighted by similarity:
each key dimension contributes proportionally to how much the direction points
along it.

---

## Equation 5: Output

```
O = G * (Q @ S  +  (Q @ K.T * M') @ V')       (C, d)
         ─────     ────────────────────
         inter         intra
```

In the recurrent loop, `o_t = q_t @ h` reads from the state. At token t, the state
has two components:

**Inter-chunk: `Q @ S`** — the original accumulated state, answering "what do all
previous chunks tell each query?" This is a (C, d) @ (d_k, d_v) = (C, d_v) GEMM.
Each query does a soft lookup of the state, same as any other read.

**Intra-chunk: `(Q @ K.T * M') @ V'`** — the new writes from this chunk, answering
"what do the new deltas in this chunk tell each query?"
- `Q @ K.T` → (C, C): attention score between each query and each key in the chunk.
  "How much does query i care about key j's write?"
- `* M'` → causal mask + gate scaling. Query i can only see writes from j <= i,
  decayed by the gate ratio between i and j.
- `@ V'` → (C, d): weighted sum of the corrected deltas. Each query collects new
  information from the tokens it can see, proportional to attention score.

**`G * (...)`** — scale everything by cumulative decay. Each query's output reflects
that the state has been decaying throughout the chunk.

---

## Equation 6: State Update

```
S_new = G[-1] * (S + K.T @ ((1/G) * V'))       (d_k, d_v)
```

After processing the chunk, the state must carry forward. In the recurrent loop,
after all C tokens, the state has been decayed by every gate and updated by every
write.

- **`S`** — the old state before this chunk
- **`K.T @ ((1/G) * V')`** — accumulate all 128 tokens' corrected deltas into the
  state. `K.T @ V'` is a (d, C) @ (C, d) = (d, d) GEMM — it's the sum of outer
  products `k_t^T @ v'_t` over all tokens, which is exactly what the recurrent
  loop does one by one.
- **`(1/G) * V'`** — un-gate before summing. Each token's delta V'[t] was computed
  under the cumulative gate G[t], but the state update needs deltas relative to
  the chunk's total gate. Dividing by G[t] removes the per-token gate.
- **`G[-1] * (...)`** — apply the total chunk decay. G[-1] = exp(sum of all gates
  in this chunk). The old state and all the un-gated writes get multiplied by the
  total decay.

**Why this roundabout un-gate then re-gate?** In the loop, token t's write gets
decayed by subsequent gates: `G[C-1]/G[t]`. Rather than applying 128 different
decay factors (not a GEMM), we:
1. Remove per-token gate: `V' / G[t]`
2. Sum via one GEMM: `K.T @ (V'/G)`
3. Apply total decay once: `G[-1] * (...)`

Algebraically identical, but expressible as a single GEMM.

---

## Summary: Recurrent → Chunked Mapping

```
Recurrent                    Chunked
───────────────────────────  ──────────────────────────────────────
cumulative gates             G, Gamma = precompute decays
                             T = inv(coupling matrix)

h = exp(g)*h                 (folded into G, Gamma throughout)

prediction = k @ h           }
delta = v - prediction       }  U, W, V' = T resolves all 128
h += k^T @ (beta * delta)   }  coupled deltas simultaneously

o = q @ h                   O = G * (Q@S + (Q@K.T * M') @ V')

(state carries forward)      S = G[-1] * (S + K.T @ (V'/G))
```

The sequential loop of C=128 steps collapsed into ~11 parallel GEMMs + one matrix
inversion (~4 more GEMMs). Every operation maps to tensor core MMAs.
