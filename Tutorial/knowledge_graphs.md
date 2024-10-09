### Knowledge graphs

These are Heterogenous graphs

applications of knowledge graphs
serving information
question answering and conversational agesnts

Publicly available KGs
Freebase, wikidata, dbpedia, YAGO, NELL etc

#### Knowledge Graph Completions

Given a enormous KG complete it
For a given (head, relation) we find the tails (missing).

How to do this?
We can do this using shallow embedding (previously covered)

KG representation
edges in KG are represented as triples (h, r, t)
[ h -> head, r -> realtion, t -> tail ]
Given a true triplet (h, r, t) the goal is that the embedding of (h, r) should be close to the embedding of t.
a. How to embed (h, r)?
b. How to define closeness?


1. TransE
    For a triple (h, r, t), $ h, r, t \in  \mathbb{R}^d $
    scoring function: $f_r(h,t) = -||h + r - t||$
    Will TransE capture all relation patterns?
    TransE can capture  antisymmetric relations, inverse relations, composition relations
    TransE cannot capture symmetric relations and 1-to-N relation.
    TransE models translation of any relation in the same embedding space.

2. TransR
    It models entities as a vector in entity space $\mathbb{R}^d$ and models each relation as a vector in relation space $r \in \mathbb{R}^k$ with $M_r \in \mathbb{R}^{k \times d}$ as the projection matrix.(Best to read on standford, remember $M_r$ is a matrix)
    Using $M_r$ to project from entity space $\mathbb{R}^d$ to relation space $\mathbb{R}^k$!
    $h_{\perp} = M_rh, t_{\perp} = M_rt$
    scoring function: $f_r(h,t) = -||h_{\perp} + r - t_{\perp}||$
    Will TransR capture all relation patterns?
    TrasR can capture antisymmetric, symmetric, inverse and 1-to-N relations
    TrasR cannot capture composition relations

3. DistMul
    Bilinear modelling using entities and relations using vectors in $\mathbb{R}^k$
    scoring function: $f_r(h,t) = <h, r, t> = \sum_i h_i . r_i . t_i$
    where h, r, t $\in \mathbb{R}^k$
    The scoring function can be viewed as cosine similarity between h, r and t
    Will DistMult capture all relation patterns?
    DistMult can capture symmetric  and 1-to-N relations
    DistMult cannot capture anti-symmetric, inverse and composition relations

4. ComplEx
    model entities and relations using vectors in $C^k$
    In addition to DistMult we use the scoring function of dismult and extract only the real part
    scoring function: $f_r(h,t) = <h, r, t> = Re(\sum_i h_i . r_i . \overline{t_i})$
    Will ComplEx capture all relation patterns?
    ComplEx can capture anti-symmetric, symmetric, inverse, 1-to-N relations
    ComplEx cannot capture composition relations

#### Relation patters

1. Symmetric (Antisymmetric) patterns
    $r(h, t) \rightarrow r(t, h) \ (r(h, t) \rightarrow \neg r(t,h)) \ \forall h, t$

    a. symmetric: Family, roommate
    b. Antisymmetric: Hypernym

2. Inverse relations
    $r_2(h, t) \rightarrow r_2(h, t)$
    eg: (Advisor, Advisee)

3. Composition relations
    $r_1(x, y) \wedge r_2(y, z) \rightarrow r_3(x, z) \ \forall x, y, z$
    eg: mother's husband is my father

4. 1-to-N relations
    $r(h, t_1), r(h, t_2), ..., r(h, t_n)$ are all true
    eg: r is a "student of"

#### KG in practice
start with TransE and then depending on the relations use the requied model.

---

### Reasoning in knowledge graphs


