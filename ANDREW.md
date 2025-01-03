Goal: I want to test how data mixture affects the performance of the model on all the downstream tasks in the evals.

I need a mixture of data and evals.

e.g. for Beauty, CDs, Clothing, Electronics and Movies dataset. 
For the current setup of running on sequential data.

1. I want to run eval on all 5 data mixtures.
2. I want to pre-make the data mixtures for all 5 datasets. The mixture is a constant amount of data that have different weights for each dataset. The weights are drawn from a dirichlet distribution. e.g. if i want to run 100 experiments, i want to make 100 mixtures of data for all 5 datasets.

Let's break this down into manageable steps:

### Phase 1: Multi-Dataset Evaluation
1. **Modify DistributedRunner and SingleRunner and Main.py to support multi-dataset evaluation**
   - Add a method to load test loaders for multiple datasets
   - Create a method to run evaluation on all datasets
   - Modify logging to track per-dataset metrics

2. **Update argument parsing**
   - Add arguments for eval datasets
   - Add arguments for eval paths
   - Add arguments for logging multi-dataset results

3. **Implement logging structure**
   - Create a consistent format for multi-dataset results
   - Set up proper file paths for each dataset's results


### Phase 2: Data Mixture Training
1. **Create Mixture Generator**
   - Implement Dirichlet sampling
   - Create data sampling based on weights
   - Save mixture configurations for reproducibility

2. **Data Loading Pipeline**
   - Create mixture dataset class
   - Implement weighted sampling
   - Handle different dataset sizes

3. **Training Loop**
   - Modify training to use mixture data
   - Track per-dataset performance during training
   - Save mixture-specific checkpoints

TODOS:
1. Mixture Generator.
   - Implement Dirichlet sampling
   - Create data sampling based on weights
   - Save mixture configurations for reproducibility
   

DONE!

2. Index Offsetter

Issues: the eval dataset index is already made, so if we do any reindexing, we'll have to reindex the eval dataset as well. 
unless we have a "mixture_reindex_function" that takes in an old index and gives a new index. 
for a given mixture, we reindex from old_index to new_index, then whenever we use any index, we just plug in this middle layer.

We can add an offset, e.g. for Datasets CDs, Beauty, Clothing, Electronics, Movies, we can add an offset of 100000, 200000, 300000, 400000, 500000 respectively. 

So we just maintain an offset json file, e.g. 
{
   "CDs": 100000,
   "Beauty": 200000,
   "Clothing": 300000,
   "Electronics": 400000,
   "Movies": 500000
}

Then when we load in dataset in MultiTaskDataset, we can just offset based on the dataset name.

