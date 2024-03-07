# binary_word_embedding

The codebase converts dense embeddings into binary using BRECS framework.

## Evaluation of vanilla embeddings.
First load and save the embeddings by running ```python3 load_vectors.py``` file.
Thereafter run evaluation on the vanilla embeddings by running ```python3 eval_vanilla.py```

The results will be stored at ```results/vanilla_embeddings.tsv```

## BRECS traning and evaluation
Train the BRECS framework by running bash ```scripts/train_scripts.sh``` . Evaluate the trained model using bash ```scripts/eval_scripts.sh```