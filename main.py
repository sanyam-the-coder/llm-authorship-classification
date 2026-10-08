import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
LABELS = ['GPT', 'Claude', 'Gemini']
MODES = ['input_only', 'output_only', 'input_output']
SEED = 42
MAX_LENGTH = 768
BATCH_SIZE = 9
LEARNING_RATE = 0.003


def accuracy_score(actual, predicted):
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def evaluate(actual, predicted):
    matrix = [[0] * 3 for _ in range(3)]
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
    return {'accuracy': accuracy_score(actual, predicted),
            'macro_f1': sum(c['f1'] for c in per_class.values()) / 3,
            'confusion_matrix': matrix, 'per_class': per_class}


def tokenize(text):
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '-', text)
    text = text.replace('’', "'").replace('ʼ', "'")
    return re.findall(r'[a-z0-9]+', text.lower())


def get_tokens(row, mode):
    if mode == 'input_only':
        return tokenize(row['LLM_Input'])
    if mode == 'output_only':
        return tokenize(row['LLM_output'])
    return tokenize(row['LLM_Input']) + ['<sep>'] + tokenize(row['LLM_output'])


def validate_data(rows):
    assert len(rows) == 36, 'This experiment uses exactly 36 supplied responses.'
    assert Counter(r['LLM_name'] for r in rows) == Counter(dict.fromkeys(LABELS, 12))
    assert len({r['record_id'] for r in rows}) == 36
    prompt_ids = sorted({r['prompt_id'] for r in rows})
    assert len(prompt_ids) == 12
    for pid in prompt_ids:
        group = [r for r in rows if r['prompt_id'] == pid]
        assert len(group) == 3 and {r['LLM_name'] for r in group} == set(LABELS)
        assert len({r['LLM_Input'] for r in group}) == 1
        assert all(r['LLM_output'].strip() for r in group)
    assert not any('https://chatgpt.com/' in r['LLM_output'] or
                   'https://gemini.google.com/' in r['LLM_output'] for r in rows)


def make_folds(rows):
    rng = random.Random(SEED)
    folds = [[] for _ in range(4)]
    for task in ['explanation', 'advice', 'reasoning']:
        ids = sorted({r['prompt_id'] for r in rows if r['task'] == task})
        assert len(ids) == 4
        rng.shuffle(ids)
        for fold, pid in zip(folds, ids):
            fold.append(pid)
    return folds


def make_vocab(training_tokens):
    counts = Counter(token for tokens in training_tokens for token in tokens)
    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2}
    for token, _ in counts.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(token_lists, vocab):
    x = torch.zeros((len(token_lists), MAX_LENGTH), dtype=torch.long)
    lengths = []
    for i, tokens in enumerate(token_lists):
        ids = [vocab.get(token, 1) for token in tokens[:MAX_LENGTH]]
        assert ids
        x[i, :len(ids)] = torch.tensor(ids)
        lengths.append(len(ids))
    return x, torch.tensor(lengths)


class CNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.conv = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, 3)

    def forward(self, x, lengths):
        features = torch.relu(self.conv(self.embedding(x).transpose(1, 2)))
        features = features.masked_fill((x == 0).unsqueeze(1), float('-inf'))
        pooled = features.max(dim=2).values
        return self.output(self.dropout(pooled))


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, 3)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.embedding(x), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1]))


def train_fold(model_type, train_tokens, train_labels, test_tokens, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocab = make_vocab(train_tokens)
    x, lengths = encode(train_tokens, vocab)
    y = torch.tensor(train_labels)
    loader = DataLoader(TensorDataset(x, lengths, y), batch_size=BATCH_SIZE, shuffle=True)
    model = model_type(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch_x, batch_lengths, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x, batch_lengths), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_y)
        losses.append(total / len(y))
    model.eval()
    with torch.no_grad():
        test_x, test_lengths = encode(test_tokens, vocab)
        predictions = model(test_x, test_lengths).argmax(dim=1).tolist()
        train_accuracy = (model(x, lengths).argmax(dim=1) == y).float().mean().item()
    return predictions, losses, train_accuracy, len(vocab)


def save_chart(experiments, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, name, color in [(-0.18, 'CNN', '#245b83'), (0.18, 'LSTM', '#d77b40')]:
        scores = [100 * next(e['accuracy'] for e in experiments
                            if e['model'] == name and e['features'] == mode) for mode in MODES]
        bars = ax.bar(x + offset, scores, width=0.35, label=name, color=color)
        ax.bar_label(bars, fmt='%.1f%%', padding=4)
    ax.axhline(100/3, color='#777777', linestyle='--', label='Chance: 33.3%')
    ax.set(xticks=x, xticklabels=['Prompt only', 'Response only', 'Prompt + response'],
           ylim=(0, 110), ylabel='Accuracy (%)',
           title='LLM identification: 36 responses, four grouped test folds')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = json.loads((ROOT / 'data.json').read_text())
    validate_data(rows)
    folds = make_folds(rows)
    experiments = []
    for name, model_type in [('CNN', CNN), ('LSTM', LSTM)]:
        for mode in MODES:
            true_labels, predicted_labels, predictions, fold_results = [], [], [], []
            for fold_index, test_ids in enumerate(folds):
                train = [r for r in rows if r['prompt_id'] not in test_ids]
                test = [r for r in rows if r['prompt_id'] in test_ids]
                assert len(train) == 27 and len(test) == 9
                assert not {r['prompt_id'] for r in train} & {r['prompt_id'] for r in test}
                train_y = [LABELS.index(r['LLM_name']) for r in train]
                test_y = [LABELS.index(r['LLM_name']) for r in test]
                pred, loss, train_acc, vocab_size = train_fold(
                    model_type, [get_tokens(r, mode) for r in train], train_y,
                    [get_tokens(r, mode) for r in test], args.epochs, SEED + fold_index)
                acc = accuracy_score(test_y, pred)
                if mode == 'input_only':
                    assert abs(acc - 1/3) < 1e-9
                true_labels.extend(test_y)
                predicted_labels.extend(pred)
                for row, guess in zip(test, pred):
                    predictions.append({'record_id': row['record_id'], 'prompt_id': row['prompt_id'],
                                        'fold': fold_index + 1, 'actual': row['LLM_name'],
                                        'predicted': LABELS[guess]})
                fold_results.append({'fold': fold_index + 1, 'test_prompt_ids': test_ids,
                                     'accuracy': float(acc), 'training_accuracy': train_acc,
                                     'vocabulary_size': vocab_size, 'epoch_losses': loss})
                print(f'{name:4} {mode:12} fold {fold_index+1}: {acc:.1%}', flush=True)
            assert len({p['record_id'] for p in predictions}) == 36
            experiments.append({
                'model': name, 'features': mode,
                **evaluate(true_labels, predicted_labels),
                'folds': fold_results, 'predictions': predictions,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'config': {'seed': SEED, 'epochs': args.epochs, 'batch_size': BATCH_SIZE,
                   'learning_rate': LEARNING_RATE, 'max_length': MAX_LENGTH,
                   'embedding_size': 32, 'hidden_size': 32, 'dropout': 0.2,
                   'folds': 4, 'device': 'cpu', 'labels': LABELS,
                   'test_prompt_ids': folds, 'validation': 'None; fixed settings, no test-based selection.',
                   'python': sys.version.split()[0], 'torch': torch.__version__,
                   'numpy': np.__version__,
                   'data_sha256': hashlib.sha256((ROOT / 'data.json').read_bytes()).hexdigest()},
        'truncated_records': {m: [r['record_id'] for r in rows if len(get_tokens(r, m)) > MAX_LENGTH]
                              for m in MODES},
        'experiments': experiments,
    }
    (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# RQ1 and RQ2 results', '',
             'Four grouped folds; 27 training and 9 test responses per fold. Each of the 36 responses is tested once.', '',
             '| Classifier | Features | Correct / 36 | Accuracy | Macro F1 |',
             '|---|---|---:|---:|---:|']
    for e in experiments:
        correct = sum(p['actual'] == p['predicted'] for p in e['predictions'])
        lines.append(f"| {e['model']} | {e['features']} | {correct} | {e['accuracy']:.1%} | {e['macro_f1']:.3f} |")
    lines += ['', 'RQ1 uses the output_only rows. RQ2 compares the three feature choices within each classifier.',
              '', 'Chance accuracy is 33.3%. The prompt-only control must score exactly 33.3% because the same prompt has all three labels.',
              '', 'Small dataset: these are exploratory results. The 36 rows contain only 12 distinct prompt groups. No model or hyperparameter was selected using test results.']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    save_chart(experiments, args.output_dir)
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
