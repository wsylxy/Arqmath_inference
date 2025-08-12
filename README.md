# This is a repo to apply the inference of math dense retrieve based on Arqmath3

## Data preparation
see `data` branch

## Dense embeddings generation:
- emb.py: generate the dense embeddings of MSE (math stack exchange) posts or formulas.

## Inference:
- inference.py: Arqmath3 task 1 inference based on pure math formulas, providing "start token", "mean pooling" and "max pooling" modes.
- inference_maxsim_post_new.py: Arqmath3 task 1 inference based on whole MSE posts
- inference_maxsim_post_new direct_emb.py: Arqmath3 task 1 inference based on whole MSE posts, using indexing provided by `https://github.com/approach0/pya0/tree/arqmath3?tab=readme-ov-file`
- inference_maxsim_formula_new.py: Arqmath3 task 1 inference based on pure math formulas
