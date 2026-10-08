# Who Wrote It? Identifying LLMs from Their Responses

## RQ1 and RQ2 experiment report

### Problem and dataset

The goal is to classify whether a text response came from GPT, Claude, or Gemini. The dataset contains 36 responses: 12 from each model family. All three models answered the same 12 prompts, covering four explanations, four everyday-advice questions, and four math/logic questions.

The model labels supplied by the student are GPT 6.1 Sol Medium, Claude Sonnet 5.5 Medium, and Gemini 3.6 Flash. These labels are recorded as reported, not independently verified. The data came from the three supplied response PDFs. No additional responses or synthetic training examples were added.

PDF headers, footers, chat URLs, and interface buttons were removed. Claude's screenshot text was transcribed using OCR and checked visually. One Gemini response about computer text storage is clipped in the original PDF; the readable portion was retained and flagged. Tables and equations were converted to text. The models' factual errors were left unchanged because the task is to identify authorship, not grade answer correctness.

### Models and training

Two small neural networks were implemented in Python with PyTorch:

- **CNN:** a 32-dimensional word embedding, one convolution with 32 channels and a three-token window, ReLU, masked max pooling, dropout, and a three-class output layer.
- **LSTM:** a 32-dimensional word embedding, one LSTM layer with 32 hidden units, the last real-token hidden state, dropout, and a three-class output layer. An LSTM satisfies the assignment's RNN requirement.

Each model was trained from scratch for 30 epochs with Adam, a learning rate of 0.003, batch size 9, dropout 0.2, cross-entropy loss, and gradient clipping at 1.0. The architecture and training settings were fixed before examining test results. There was no validation-based model selection or early stopping.

Text was lowercased and split into tokens containing English letters and numbers. Punctuation and visual layout were removed consistently across sources. The vocabulary was learned from each training fold only, and unseen words were mapped to an unknown token. A separator token divided prompt and response in the combined condition. The maximum sequence length was 768 tokens; the longest combined record had 634, so no record was shortened by this limit.

### Evaluation design

Four-fold cross-validation was performed by prompt group. Each fold held out three prompts, one from each task category, including all three model responses to each prompt. Each training run therefore used 27 responses for training and 9 for testing. Every response received one held-out prediction per experiment.

| Fold | Held-out original prompt IDs | Training responses | Test responses |
|---|---|---:|---:|
| 1 | 3, 8, 14 | 27 | 9 |
| 2 | 2, 7, 16 | 27 | 9 |
| 3 | 4, 5, 15 | 27 | 9 |
| 4 | 1, 6, 13 | 27 | 9 |

Grouping prevents a prompt from appearing in both training and testing in the same fold. Both classifiers and all feature conditions used these same folds. Source filenames, model versions, and other metadata were excluded from the features. The base random seed was 42, with training seeds 42–45 across folds.

Accuracy and macro F1 were calculated over the 36 held-out predictions for each experiment. Macro F1 gives equal weight to the three model families. Uniform random guessing has an expected accuracy of 33.3% on this balanced dataset. The evaluation tests new prompts from the same three task categories; it is not an RQ3 experiment on an entirely unseen domain.

### Results

| Classifier | Features | Correct / 36 | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| CNN | Prompt only | 12 | 33.3% | 0.272 |
| CNN | Response only | 15 | 41.7% | 0.413 |
| CNN | Prompt + response | 12 | 33.3% | 0.297 |
| LSTM | Prompt only | 12 | 33.3% | 0.233 |
| LSTM | Response only | 16 | 44.4% | 0.439 |
| LSTM | Prompt + response | 12 | 33.3% | 0.325 |

### RQ1: Can we identify which LLM generated a response?

Using the response alone, the CNN correctly classified 15 of 36 responses (41.7%), and the LSTM correctly classified 16 of 36 (44.4%). Both observed accuracies exceed the 33.3% chance benchmark, but only by three and four correct predictions, respectively, relative to the 12-correct benchmark. The LSTM exceeded the CNN by one prediction. These results provide limited exploratory evidence of identifiable differences in this collection; they do not establish reliable general identification or a meaningful advantage for one architecture.

The response-only confusion matrices show where mistakes occurred. Rows represent actual sources; columns represent predicted sources.

**CNN, response only**

| Actual / Predicted | GPT | Claude | Gemini |
|---|---:|---:|---:|
| GPT | 3 | 6 | 3 |
| Claude | 0 | 5 | 7 |
| Gemini | 0 | 5 | 7 |

**LSTM, response only**

| Actual / Predicted | GPT | Claude | Gemini |
|---|---:|---:|---:|
| GPT | 5 | 3 | 4 |
| Claude | 4 | 4 | 4 |
| Gemini | 2 | 3 | 7 |

The CNN confused Claude and Gemini frequently and recognized only 3 of 12 GPT responses. The LSTM recognized 7 of 12 Gemini responses, 5 of 12 GPT responses, and 4 of 12 Claude responses. Neither classifier identified all families consistently.

### RQ2: Does the user's prompt help identify the LLM?

Prompt-only accuracy was exactly 33.3% for both classifiers. This is expected from the collection design: every question appears once with each of the three model labels. A deterministic classifier gives identical questions the same prediction and therefore gets exactly one of the three right. The prompt by itself contains no information that distinguishes the source label in this dataset.

Adding the prompt did not improve observed accuracy. CNN accuracy fell from 41.7% to 33.3%, a decrease of 8.3 percentage points. LSTM accuracy fell from 44.4% to 33.3%, a decrease of 11.1 percentage points. Shared question content may distract these small classifiers or make useful response patterns harder to learn from so few examples. This is a possible explanation, not a demonstrated cause. Training randomness and the small sample also matter.

In another dataset, unusually strong prompt-only performance could indicate that different models received different topics, templates, or collection sources. A classifier might then learn collection choices rather than response authorship. Using matched prompts across all three families removes that particular source of label information here.

### Limitations and lessons

Both models achieved 100% training accuracy in every response-only and combined-input fold, while their held-out accuracies were much lower. This large gap is evidence of overfitting: the networks learned the tiny training sets much better than they generalized to new prompts. High training accuracy alone is not evidence of a successful authorship classifier.

The dataset contains 36 rows but only 12 distinct prompt groups. A single additional correct prediction changes pooled accuracy by about 2.8 percentage points. Four-fold testing uses all available examples for evaluation, but it does not create new independent data. The training sets also overlap across folds. No statistical significance claim is made, and only one seed schedule was evaluated.

The source PDFs show shared conversation histories for GPT and Gemini. Only the immediate question is stored as the input feature; full prior context and personalization settings are unknown. Model-version labels are student-reported. Differences in response length, source interfaces, OCR, table extraction, and the clipped Gemini answer may affect identification. Normalizing punctuation and layout reduces some extraction differences but cannot eliminate all of them.

The main lessons are to keep model labels out of the features, split by prompt instead of by individual response, compare training and test performance, and report weak results honestly. For these 36 responses and the fixed training setup, response-only text performed best, and adding the prompt did not help. The evidence remains specific to this small collection.

### Reproducibility and sources

Run `python main.py` after installing `requirements.txt`. The dataset is in `data.json`; `data_notes.json` records extraction decisions and SHA-256 hashes of the source PDFs. `results/results.json` contains the configuration, all 216 held-out predictions across the six experiments, per-fold training losses, and per-class metrics.

Source documents: `GPT_Responses.pdf`, `Claude_Responses.pdf`, and `Gemini_Responses.pdf`, supplied by the student. Model labels were supplied in the conversation. The assignment requirements came from `Midterm project-1-1.pdf`. No external response dataset was used.

This is the RQ1/RQ2 report draft. Add the required author/team information and review the wording before preparing the final submission PDF.
