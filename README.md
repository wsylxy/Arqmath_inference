## Data preparation for math dense retriever training and inference

# Get data from official xml file
- get_formulas.py: only extract math formulas.
- get_posts.py: extract whole posts.
Extracted data will be saved as json file.

# Data preprocess and filtering
- filtering_formula_pya0.py: preprocess and filter formulas.
- filtering_post_pya0.py: preprocess and filter posts.
- filtering_fusion_pya0.py: preprocess and filter fused formulas and posts.
The output file will in form of blocks of question posts/formulas and corresponding answer posts/formulas.

# Create trainset
- creating_trainset_formula_pya0.py: create trainset for pure formula based math dense retrievers finetuning.
- creating_trainset_post_pya0.py: create trainset for post based math dense retrievers finetuning.
- creating_trainset_fusion_pya0.py: create trainset for dual encoder based math dense retrievers finetuning.

# Create test set used for inference
- create_topic_formula_pya0.py: create pure math formula based test set.
- create_topic_post_pya0.py: create post based test set
  
  
