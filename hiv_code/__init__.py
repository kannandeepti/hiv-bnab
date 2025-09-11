r"""
Model of the persistent humoral immune response
===============================================

The module `hiv_code` contains a `Simulation` class which performs the overall
simulation algorithm, a `Bcells` class to define different B cell populations,
and a `Concentrations` class to track the dynamics of the concentration and affinity
of antibodies targeting distinct epitopes.

Each simulation consists of `n_gc` Germinal Center (GC) reactions and subsequent
expansion of memory B cells in compartments outside of the GC called EGCs. Our
model includes activation of naive B cells for GC entry, mutation-selection
dynamics of B cells inside GCs, export and differentiation of GC B cells into
memory B cells and antibody-secreting plasma cells, and affinity-dependent
expansion of memory B cells inside the EGC. EGC memory B cells rapidly differentiate
into plasma cells which secrete the majority of antibodies in serum. The
simulation thus predicts the dynamics of the concentration and affinity of
antibodies targeting distinct epitopes.
"""
