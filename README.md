# LLM Authorship Classification

CMPSC 448 midterm project

Team leader: Sanyam Agrawal  
Team member: Shaan Patel

This project uses a CNN and an LSTM to identify whether a response came from GPT, Claude, or Gemini. It covers RQ1, RQ2, RQ3, and RQ4.

## Dataset

The included `data.json` has 36 responses: 12 from each model family. Each model answered the same 12 prompts, covering explanations, everyday advice, and math or logic.

The text was extracted and cleaned from three response PDFs. One Gemini answer was clipped, so only its visible text was included. The model versions are team-reported labels and were not independently verified.

## Files

- [Final_Report.pdf](Final_Report.pdf): Final report covering RQ1 through RQ4.
- [Final_Report.md](Final_Report.md): Markdown version of the same final report.
- `main.py`: CNN and LSTM models, data checks, and RQ1/RQ2 experiments.
- `rq3_rq4.py`: RQ3/RQ4 experiments. It imports the models and helper functions from `main.py`.
- `data.json`: Prompts, responses, model labels, and task categories.
- `data_notes.json`: Data sources and extraction notes.
- `requirements.txt`: Python packages.

## Run

Use Python 3.10 or newer. Download or clone the repository, open its folder in a terminal, and run:

```bash
python -m pip install -r requirements.txt
python main.py
python rq3_rq4.py
```

Both scripts print results in the terminal. They do not save result files or graphs.

The defaults are 30 training epochs and 20 tokens for the shortened-answer experiment. Both scripts accept `--data PATH` and `--epochs NUMBER`. The RQ3/RQ4 script also accepts `--prefix-tokens NUMBER`. Changing these settings can change the results.

## Experiments

- **RQ1:** Identify the model using its response only.
- **RQ2:** Compare prompt only, response only, and prompt + response.
- **RQ3:** Train on two task categories and test on the third, rotating all three categories.
- **RQ4:** Compare token counts and lexical diversity, test a length-only classifier, and compare full responses with their first 20 tokens.

RQ1, RQ2, and RQ4 use four test groups, with 27 training responses and 9 test responses per group. RQ3 uses 24 training responses and 12 test responses per category. Training and test prompts stay separate, and the vocabulary is built from training text only.

Both neural models use 32-dimensional token embeddings, dropout of 0.2, Adam with a learning rate of 0.003, and a batch size of 9. The CNN uses 32 filters with a three-token window. The LSTM has 32 hidden units. Training runs on the CPU with fixed random seeds.

## Results

These results were reproduced using the included dataset and default settings.

| Classifier | Input used | Accuracy | Macro F1 |
|---|---|---:|---:|
| CNN | Prompt only | 33.3% | 0.272 |
| CNN | Response only | 47.2% | 0.440 |
| CNN | Prompt + response | 50.0% | 0.463 |
| LSTM | Prompt only | 33.3% | 0.233 |
| LSTM | Response only | 44.4% | 0.416 |
| LSTM | Prompt + response | 44.4% | 0.433 |

For RQ3, both classifiers scored 33.3% accuracy on every held-out category. For RQ4, the length-only classifier scored 52.8%. Using only the first 20 tokens gave 30.6% accuracy for both neural models.

Adding the prompt helped the CNN by one correct prediction and did not change LSTM accuracy. The dataset is small, so these results do not show reliable model identification. Shortening responses also removes content, so the accuracy drop cannot be explained by length alone.
