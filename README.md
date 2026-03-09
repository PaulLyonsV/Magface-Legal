MagFace Loss for Legal Contract Retrieval

Deep learning models tend to generalize when the embedding space matches the geometry of
the input space. Legal contracts have a hierarchical, tree-like, structure where parent clauses
contain sub clauses made up of paragraphs and sentences. Euclidean space struggles to model
entailment relationships because the space’s volume grows polynomially with respect to length,
while the width of a hierarchical tree can grow exponentially with depth. Hyperbolic spaces are
capable of embedding hierarchies by placing roots near the origin and leaves at the boundary.
Ganea et. al. showed that hyperbolic spaces where volume grows exponentially with respect to length
are superior to Euclidean spaces for entailment graphs. Feng et. al. achieved state of the art results 
modeling knowledge graphs in a mixed Euclidean and hyperbolic space. Knowledge
graphs are typically made up of nodes represented as triples (parent entity, relation, tail entity),
and model information similar to the entailment structure of legal contracts.

Despite these promising results, hyperbolic metrics are computationally expensive and introduce 
exponential functions, which can be unstable if not handled carefully. We propose a method to
emulate the conical structure of hyperbolic embeddings in Euclidean space using the 
norm-adaptive margin angular loss introduced as MagFace by Meng et. al. Their model is applied 
to facial classification and map feature norms to image quality. We map norms to clause specificity. 
We adapt the geometric principles in their loss function to a contrastive learning framework used 
in natural language processing tasks.

[Read the full paper here.](./MagFace-Legal.pdf)
