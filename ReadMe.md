### Execution Instructions
1. Change current directory to: 'Code/' (where code files are located)
2. Prepare data in the given format and place it in : 'Data/' directory
3. Execution
	3.1 python prepare_data.py
   It prepares and dumps the dataframes in a format which can be used by models and dataloaders to train and evaluate.

	3.2 python train_tgif.py
   Trains the proposed tgif model(27) on the prepared data and dumps the best models in 'models/27/'

	3.3 python eval_tgif.py
   Loads the best model and calculates weighted f1, recall and precision scores for both validation and test set. And saves the grouth truth and prediction csv in 'pred_csv/'
