
### Final Model
- The final model is implemented in **`FINAL_MODEL_deberta_wrs_categorical.py`**.

### Testing the Model
- The **`Final_Model_on_DevSet.ipynb`** notebook is used to:
  - Load the saved trained model.
  - Test it on the **development dataset (`dev_semeval_parids-labels.csv`)**.
  - Generated labels for devset is in **`dev.txt`** and for the hidden test set is in **`test.txt`**.

### Training Logs
- The logs from training our final model (DeBERTa with weighted random sampling and categorical columns) are stored in **`FINAL_MODEL.log`**.

### Usage
To use the final model, run the `FINAL_MODEL_deberta_wrs_categorical.py` script for label generation.
For testing and label generation on development and test datasets, execute the **`Final_Model_on_DevSet.ipynb`** notebook.
