# Gated Delta Rule: Intuitive Recurrence Relation

## The Recurrent Loop

```python
for t in range(T):
    h = exp(g_t) * h                        # 1. decay old state
    prediction = k_t @ h                     # 2. (d_k,) @ (d_k, d_v) = (d_v,) soft lookup
    delta = v_t - prediction                 # 3. what's new?
    h = h + k_t^T @ (beta_t * delta)        # 4. write correction into state
    o_t = (scale * q_t) @ h                  # 5. query reads from state
```

**State `h`** is a (d_k x d_v) associative memory. Each row corresponds to a key
dimension. Reading from it (`k @ h`) is a soft lookup by similarity — you get a
weighted blend of all stored values, weighted by how much your key overlaps with
each stored key direction. Writing to it (`h += k^T @ delta`) stores new
information along your key direction.

We need to deeply understand every line of this loop. Let's go through each one
with a concrete example.

We'll use d_k = 2, d_v = 2 throughout. Start with a state that already has some
stored associations:

```
S = [[3, 5],      ← key dimension 0 is associated with value [3, 5]
     [7, 2]]      ← key dimension 1 is associated with value [7, 2]
```

And a new token arrives: `k_t = [0.8, 0.6]`, `v_t = [9, 1]`, `g_t = -0.3`, `beta_t = 0.7`.

---

## Line 1: `h = exp(g_t) * h` — Decay old state

```
h = exp(-0.3) * [[3, 5], [7, 2]]
  = 0.741 * [[3, 5], [7, 2]]
  = [[2.22, 3.70],
     [5.19, 1.48]]
```

Every entry in the state shrinks. Old associations fade. `g_t` is always negative
(log-sigmoid), so `exp(g_t)` is always between 0 and 1.

**Why decay?** Without it, the state accumulates forever. Old, irrelevant
associations from thousands of tokens ago pollute every prediction. The gate lets
the model decide: "how much of the past should I keep right now?"

---

## Line 2: `prediction = k_t @ h` — Soft lookup

**h is a (d_k x d_v) lookup table.** Each row maps a key dimension to a value:

```
h (after decay) = (d_k × d_v)

          v_dim0  v_dim1
k_dim0  [ 2.22    3.70  ]    ← "key dimension 0 is associated with value [2.22, 3.70]"
k_dim1  [ 5.19    1.48  ]    ← "key dimension 1 is associated with value [5.19, 1.48]"
```

These associations were built by previous writes. For example, the original state
before decay was built from earlier outer products:
```
k_a = [1, 0], v_a = [3, 5]   →  S += k_a^T ⊗ v_a = [[3,5],[0,0]]
k_b = [0, 1], v_b = [7, 2]   →  S += k_b^T ⊗ v_b = [[0,0],[7,2]]

S = [[3, 5],     (then decayed to [[2.22, 3.70],
     [7, 2]]                       [5.19, 1.48]])
```

**Reading from h is a soft lookup by key similarity.** When we compute
`prediction = k_t @ h`, we're asking: "given my key direction, what value does
the state predict?"

```
k_t = [0.8, 0.6]     (d_k,)

prediction = k_t @ h
           = [0.8, 0.6] @ [[2.22, 3.70],
                            [5.19, 1.48]]

           = 0.8 * [2.22, 3.70]  +  0.6 * [5.19, 1.48]
             ──────────────────     ──────────────────
             "k_t has 0.8 in dim 0   "k_t has 0.6 in dim 1
              → 80% of row 0"         → 60% of row 1"

           = [1.78, 2.96]  +  [3.11, 0.89]
           = [4.89, 3.85]     (d_v,)
```

The weights [0.8, 0.6] come from the key vector itself. Each weight says "how
much does k_t point in this key dimension?" The result is a weighted blend of all
stored values — more weight on dimensions that k_t aligns with. This is a **soft
lookup by key similarity**.

**For `K @ S` (all C tokens at once):** each row of the output is one token's
prediction — what the state associates with that token's key direction.

```
K @ S  →  (C, d_k) @ (d_k, d_v) = (C, d_v)

Row 0: "what does S predict for k_0's direction?"    → (d_v,)
Row 1: "what does S predict for k_1's direction?"    → (d_v,)
...
Row 127: "what does S predict for k_127's direction?" → (d_v,)
```

---

## Line 3: `delta = v_t - prediction` — What's new?

```
delta = [9, 1] - [4.89, 3.85]
      = [4.11, -2.85]     (d_v,)
```

**What's happening:** The state predicted [4.89, 3.85] for this key direction.
The actual value is [9, 1]. The delta is the gap — what the state doesn't already
know.

If the state already had the right value stored (prediction ≈ v_t), delta ≈ 0
and almost nothing gets written. This is the **idempotent correction** property:
writing the same (k, v) pair twice doesn't double the entry, because the second
write's prediction catches the first write's value.

**Example — same key, different value (with beta=1 for simplicity):**

```
S = [[0, 0],
     [0, 0]]

Write 1: k = [1, 0], v = [3, 5]  →  prediction = [1,0] @ [[0,0],[0,0]] = [0, 0]
                                     delta = [3,5] - [0,0] = [3, 5]
                                     S = S + [1,0]^T ⊗ [3,5] = [[3, 5], [0, 0]]

Write 2: k = [1, 0], v = [6, 1]  →  prediction = [1,0] @ [[3,5],[0,0]] = [3, 5]
                                     delta = [6,1] - [3,5] = [3, -4]
                                     S = [[3,5],[0,0]] + [[3,-4],[0,0]] = [[6, 1], [0, 0]]
```

The state now stores [6, 1] for key direction [1, 0] — the old value [3, 5] was
fully replaced. The prediction found the existing association, so the delta only
wrote the *difference*. The soft lookup is what enables correction instead of
blind accumulation.

**Contrast with vanilla linear attention (no delta rule):**

```
S = [[0,0],[0,0]]
Write 1: S += [1,0]^T ⊗ [3,5] = [[3, 5], [0, 0]]
Write 2: S += [1,0]^T ⊗ [6,1] = [[9, 6], [0, 0]]    ← accumulated, not replaced!
```

The state has [9, 6] = v_1 + v_2, which is neither value. The delta rule avoids
this by subtracting what's already known before writing.

---

## Line 4: `h = h + k_t^T @ (beta_t * delta)` — Write correction

```
beta_t * delta = 0.7 * [4.11, -2.85] = [2.88, -2.00]     (d_v,)

outer product:
k_t^T @ (beta_t * delta) = [[0.8],  @  [[2.88, -2.00]]
                             [0.6]]

                          = [[0.8*2.88, 0.8*(-2.00)],     = [[2.30, -1.60],
                             [0.6*2.88, 0.6*(-2.00)]]        [1.73, -1.20]]

h = [[2.22, 3.70],  +  [[2.30, -1.60],  =  [[4.52, 2.10],
     [5.19, 1.48]]      [1.73, -1.20]]      [6.92, 0.28]]
```

**What's happening, step by step:**

1. **`beta_t * delta`**: Scale the correction by beta (write strength). beta = 0.7
   means "correct 70% of the way toward the true value." beta = 1 would fully
   correct; beta = 0 would write nothing.

2. **`k_t^T @ (...)`**: The outer product creates a (d_k x d_v) update matrix.
   The key `k_t = [0.8, 0.6]` determines WHERE in the state to write:
   - Row 0 gets 0.8x of the correction (k_t has 0.8 in dim 0)
   - Row 1 gets 0.6x of the correction (k_t has 0.6 in dim 1)

   The correction is "spread" across key dimensions proportionally to k_t.

3. **`h = h + ...`**: Add the correction to the state.

**The net effect:** The state's prediction for `k_t`'s direction has been nudged
toward `v_t`. Not all the way (because beta = 0.7 and because k_t isn't a unit
basis vector), but closer. If the same (k_t, v_t) arrived again, the prediction
would be closer, the delta smaller, the write weaker — converging.

---

## Line 5: `o_t = (scale * q_t) @ h` — Query reads from state

```
q_t = [1, 0],  scale = 1.0

o_t = [1, 0] @ [[4.52, 2.10],
                 [6.92, 0.28]]

    = 1.0 * [4.52, 2.10]  +  0.0 * [6.92, 0.28]
    = [4.52, 2.10]     (d_v,)
```

**What's happening:** Same soft lookup as prediction, but with the query vector.
The query asks "what does the state know about my direction?" and gets a blended
answer.

Here `q_t = [1, 0]` is purely in dimension 0, so it retrieves row 0 exactly. A
query like `[0.5, 0.5]` would get a 50/50 blend of both rows.

**Key distinction between q and k:** Both do soft lookups from the same state h.
But `k` is used for prediction (to compute the delta for writing), while `q` is
used for output (to produce the attention result). They can point in completely
different directions — the model learns them separately.

---

## Aside: Why L2-normalized keys help

Not important to understand right away, but the kernel has a
`use_qk_l2norm_in_kernel` flag for this.

If stored keys have different magnitudes, the outer product write strength scales
with magnitude. A key with magnitude 10 writes 10x harder into the state than a
unit key, and any subsequent lookup along that dimension gets flooded by the large
write — regardless of actual similarity. L2 normalization makes all keys unit
length, so write strength is uniform and the blending weights during prediction
purely reflect directional similarity.

---

## Putting it together

One iteration of the loop transforms:
```
S_before = [[3.00, 5.00],     S_after = [[4.52, 2.10],
            [7.00, 2.00]]                [6.92, 0.28]]
```

The state decayed (old info faded), predicted (what do I know about this key?),
computed a delta (what's new?), and wrote the correction (update my knowledge).
Then a query read the updated state.

The sequential dependency is clear: line 4 modifies h, and the next iteration's
line 2 reads from this modified h. **This is why we can't naively parallelize —
and why the chunked algorithm needs the coupling matrix T.**

See [intuitive_chunked_equations.md](intuitive_chunked_equations.md) for how the
chunked algorithm resolves this.
