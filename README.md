# jobLLMatch
jobLLMatch creates a fine-tuned BART model that provides feedback to candidates based on their resume and a job description

# Setup
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Synthetic Data Generation | using PALM-2
To generate a score and feedback report per (job description, resume) pair use the `palm_generate_reports.ipynb` notebook. It was run in Google Colab since in my experience the inference time was significantly shorter as compared to local or Google Cloud implementation.

## Model Fine-tuning
Refer to `finetune_BART.py`


## Inference | Generate Scores and Reports | using fine-tuned bart-large
Now that we have a fine-tuned model we can go to `inference_BART.py` and generate scores and reports for the test set (15% of the total data).
You can run inference by:

```bash
python inference_BART.py --checkpoint <path to model checkpoint>
```

there are also some optional arguments suchs as:

- `--test_mode`: if present, then it uses `--test_mode_size <int>` to slice the test dataset so that it runs an experimental run on a smaller part of the dataset
- `--csv`: if present the it exports the test dataset together with the generated data as csv, default is `.pkl`
- `--calculate_loss`: if present, it calculates the model loss of the test dataset (not present during training)


## Calculate Metrics and Scores
Refer to `metrics_notebook.ipynb`

# Disclaimers
- Occasionally ChatGPT was used to troubleshoot code bugs.