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
TASKS = ['explanation', 'advice', 'reasoning']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(path):
    path = Path(path)
    try:
        rows = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise ValueError(f'Dataset not found: {path}. Place data.json beside main.py or use --data PATH.') from error
    except (OSError, UnicodeError) as error:
        raise ValueError(f'Cannot read dataset {path}: {error}') from error
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid JSON in {path}: line {error.lineno}, column {error.colno}.') from error
    validate_data(rows)
    return rows


def accuracy_score(actual, predicted):
    require(len(actual) > 0 and len(actual) == len(predicted),
            'Evaluation needs nonempty actual and predicted lists of equal length.')
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def evaluate(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    require(all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(LABELS)
                for v in [*actual, *predicted]), 'Evaluation labels must be integer class IDs 0, 1, or 2.')
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
    return {'accuracy': accuracy,
            'macro_f1': sum(c['f1'] for c in per_class.values()) / 3,
            'confusion_matrix': matrix, 'per_class': per_class}


def tokenize(text):
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '-', text)
    text = text.replace('’', "'").replace('ʼ', "'")
    return re.findall(r'[a-z0-9]+', text.lower())


def get_tokens(row, mode):
    require(mode in MODES, f'Unknown feature mode: {mode}. Expected one of {MODES}.')
    if mode == 'input_only':
        return tokenize(row['LLM_Input'])
    if mode == 'output_only':
        return tokenize(row['LLM_output'])
    return tokenize(row['LLM_Input']) + ['<sep>'] + tokenize(row['LLM_output'])


def validate_data(rows):
    require(isinstance(rows, list), 'Dataset must be a JSON list of response records.')
    require(len(rows) == 36, 'This fixed experiment expects 36 responses: 12 per family.')
    required = ['record_id', 'prompt_id', 'task', 'LLM_name', 'LLM_Input', 'LLM_output']
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict), f'Record {index} must be a JSON object.')
        missing = [key for key in required if key not in row]
        require(not missing, f'Record {index} is missing fields: {", ".join(missing)}.')
        for key in ['record_id', 'task', 'LLM_name', 'LLM_Input', 'LLM_output']:
            require(isinstance(row[key], str) and bool(row[key].strip()),
                    f'Record {index}: {key} must be nonempty text.')
        require(isinstance(row['prompt_id'], int) and not isinstance(row['prompt_id'], bool),
                f'Record {index}: prompt_id must be an integer.')
        require(row['LLM_name'] in LABELS, f'Record {index}: unknown family {row["LLM_name"]}.')
        require(row['task'] in TASKS, f'Record {index}: task must be one of {TASKS}.')
        require(bool(tokenize(row['LLM_Input'])) and bool(tokenize(row['LLM_output'])),
                f'Record {index}: prompt and response must contain letters or digits.')
        require(not any(url in row['LLM_output'] for url in
                        ['https://chatgpt.com/', 'https://gemini.google.com/']),
                f'Record {index}: remove source interface URLs from the response.')
    require(Counter(r['LLM_name'] for r in rows) == Counter(dict.fromkeys(LABELS, 12)),
            'Expected exactly 12 responses each from GPT, Claude, and Gemini.')
    require(len({r['record_id'] for r in rows}) == 36, 'record_id values must be unique.')
    prompt_ids = sorted({r['prompt_id'] for r in rows})
    require(len(prompt_ids) == 12, 'Expected exactly 12 distinct prompt groups.')
    for pid in prompt_ids:
        group = [r for r in rows if r['prompt_id'] == pid]
        require(len(group) == 3 and {r['LLM_name'] for r in group} == set(LABELS),
                f'Prompt {pid} needs one response from each family.')
        require(len({r['LLM_Input'] for r in group}) == 1,
                f'Prompt {pid} must have identical question text for all families.')
        require(len({r['task'] for r in group}) == 1,
                f'Prompt {pid} must have one consistent task category.')
    for task in TASKS:
        require(len({r['prompt_id'] for r in rows if r['task'] == task}) == 4,
                f'Expected four distinct questions in category {task}.')


def validate_folds(rows, folds):
    require(len(folds) == 4 and all(len(fold) == 3 for fold in folds),
            'Expected four test folds with three question IDs each.')
    held_out = [pid for fold in folds for pid in fold]
    require(Counter(held_out) == Counter({r['prompt_id'] for r in rows}),
            'Every question must appear in exactly one test fold.')
    for index, ids in enumerate(folds, 1):
        test = [r for r in rows if r['prompt_id'] in ids]
        require(Counter(r['task'] for r in test) == Counter(dict.fromkeys(TASKS, 3)),
                f'Fold {index} must hold out one question from each task category.')
        require(Counter(r['LLM_name'] for r in test) == Counter(dict.fromkeys(LABELS, 3)),
                f'Fold {index} must contain three test responses per family.')


def make_folds(rows):
    rng = random.Random(SEED)
    folds = [[] for _ in range(4)]
    for task in TASKS:
        ids = sorted({r['prompt_id'] for r in rows if r['task'] == task})
        require(len(ids) == 4, f'Expected four questions in category {task}.')
        rng.shuffle(ids)
        for fold, pid in zip(folds, ids):
            fold.append(pid)
    validate_folds(rows, folds)
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
        require(bool(ids), f'Cannot encode empty token sequence at index {i}.')
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        rows = load_data(args.data)
    except ValueError as error:
        parser.error(str(error))
    folds = make_folds(rows)
    experiments = []
    for name, model_type in [('CNN', CNN), ('LSTM', LSTM)]:
        for mode in MODES:
            true_labels, predicted_labels, predictions, fold_results = [], [], [], []
            for fold_index, test_ids in enumerate(folds):
                train = [r for r in rows if r['prompt_id'] not in test_ids]
                test = [r for r in rows if r['prompt_id'] in test_ids]
                require(len(train) == 27 and len(test) == 9, 'Expected 27 training and 9 test responses.')
                require(not {r['prompt_id'] for r in train} & {r['prompt_id'] for r in test},
                        'Training and test questions overlap.')
                train_y = [LABELS.index(r['LLM_name']) for r in train]
                test_y = [LABELS.index(r['LLM_name']) for r in test]
                pred, loss, train_acc, vocab_size = train_fold(
                    model_type, [get_tokens(r, mode) for r in train], train_y,
                    [get_tokens(r, mode) for r in test], args.epochs, SEED + fold_index)
                acc = accuracy_score(test_y, pred)
                if mode == 'input_only':
                    require(abs(acc - 1/3) < 1e-9, 'Matched prompt-only accuracy must equal one-third.')
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
            require(len(predictions) == 36 and len({p['record_id'] for p in predictions}) == 36,
                    'Each response must receive exactly one held-out prediction.')
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
                   'data_path': str(args.data.resolve()),
                   'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()},
        'truncated_records': {m: [r['record_id'] for r in rows if len(get_tokens(r, m)) > MAX_LENGTH]
                              for m in MODES},
        'experiments': experiments,
    }
    (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('\nRQ1 and RQ2 results')
    for result in experiments:
        print(f"{result['model']} {result['features']}: "
              f"accuracy {result['accuracy']:.1%}, macro F1 {result['macro_f1']:.3f}")
    print(f"Detailed results saved to {args.output_dir / 'results.json'}")



if __name__ == '__main__':
    main()
