import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
import main as project

ROOT = Path(__file__).resolve().parent
TASKS = project.TASKS
MODELS = [('CNN', project.CNN), ('LSTM', project.LSTM)]


def fit_and_test(name, model_type, train, test, epochs, seed, prefix=None):
    project.require(not {r['prompt_id'] for r in train} & {r['prompt_id'] for r in test},
                    'Training and test questions overlap.')
    train_tokens = [project.get_tokens(r, 'output_only') for r in train]
    test_tokens = [project.get_tokens(r, 'output_only') for r in test]
    if prefix is not None:
        train_tokens = [tokens[:prefix] for tokens in train_tokens]
        test_tokens = [tokens[:prefix] for tokens in test_tokens]
    train_y = [project.LABELS.index(r['LLM_name']) for r in train]
    test_y = [project.LABELS.index(r['LLM_name']) for r in test]
    guesses, losses, train_accuracy, vocab_size = project.train_fold(
        model_type, train_tokens, train_y, test_tokens, epochs, seed)
    return {'model': name, **project.evaluate(test_y, guesses),
            'training_accuracy': train_accuracy, 'vocabulary_size': vocab_size,
            'epoch_losses': losses, 'seed': seed,
            'train_prompt_ids': sorted({r['prompt_id'] for r in train}),
            'test_prompt_ids': sorted({r['prompt_id'] for r in test}),
            'training_responses': len(train), 'test_responses': len(test),
            'predictions': [{'record_id': row['record_id'], 'prompt_id': row['prompt_id'],
                             'actual': row['LLM_name'], 'predicted': project.LABELS[guess]}
                            for row, guess in zip(test, guesses)]}


def run_rq3(rows, epochs):
    experiments = []
    for name, model_type in MODELS:
        for index, task in enumerate(TASKS):
            train = [r for r in rows if r['task'] != task]
            test = [r for r in rows if r['task'] == task]
            project.require(len(train) == 24 and len(test) == 12, 'Expected 24 training and 12 test responses for RQ3.')
            project.require(Counter(r['LLM_name'] for r in test) == Counter(dict.fromkeys(project.LABELS, 4)),
                            'RQ3 test data must contain four responses per family.')
            result = fit_and_test(name, model_type, train, test, epochs, project.SEED + index)
            result.update({'features': 'output_only', 'held_out_task': task,
                           'training_tasks': [t for t in TASKS if t != task]})
            experiments.append(result)
            print(f"RQ3 {name} held out {task}: {result['accuracy']:.1%}", flush=True)
    return experiments


def run_grouped(rows, folds, epochs, prefix=None):
    experiments = []
    for name, model_type in MODELS:
        results = []
        predictions = []
        for index, ids in enumerate(folds):
            train = [r for r in rows if r['prompt_id'] not in ids]
            test = [r for r in rows if r['prompt_id'] in ids]
            result = fit_and_test(name, model_type, train, test, epochs,
                                  project.SEED + index, prefix)
            result['fold'] = index + 1
            for prediction in result['predictions']:
                prediction['fold'] = index + 1
            predictions.extend(result['predictions'])
            results.append(result)
            print(f"RQ4 {name} prefix={prefix} fold {index + 1}: {result['accuracy']:.1%}", flush=True)
        actual = [project.LABELS.index(p['actual']) for p in predictions]
        guessed = [project.LABELS.index(p['predicted']) for p in predictions]
        experiments.append({'model': name, 'features': 'output_only' if prefix is None else 'response_prefix',
                            'prefix_tokens': prefix, **project.evaluate(actual, guessed),
                            'folds': results, 'predictions': predictions})
    return experiments


def run_length_only(rows, folds, lengths):
    predictions = []
    for index, ids in enumerate(folds):
        train = [r for r in rows if r['prompt_id'] not in ids]
        test = [r for r in rows if r['prompt_id'] in ids]
        means = [statistics.mean(math.log(lengths[r['record_id']])
                                 for r in train if r['LLM_name'] == label)
                 for label in project.LABELS]
        for row in test:
            value = math.log(lengths[row['record_id']])
            guess = min(range(3), key=lambda i: abs(value - means[i]))
            predictions.append({'record_id': row['record_id'], 'prompt_id': row['prompt_id'],
                                'fold': index + 1, 'actual': row['LLM_name'],
                                'predicted': project.LABELS[guess]})
    actual = [project.LABELS.index(p['actual']) for p in predictions]
    guessed = [project.LABELS.index(p['predicted']) for p in predictions]
    return {**project.evaluate(actual, guessed), 'predictions': predictions,
            'method': 'Nearest training-class mean log response token count; no tuned thresholds.'}


def write_sections(results, output_dir):
    rq3 = results['rq3']
    rq4 = results['rq4']
    prefix = results['config']['prefix_tokens']
    lines = ['### RQ3: Do LLM fingerprints generalize across tasks or domains?', '',
             'We trained on two task categories and tested on the third, rotating explanation, advice, and reasoning. Only response text was used. Each run trained a new CNN or LSTM on 24 responses and tested on 12 responses from four unseen questions. The vocabulary was fitted on training data only. Architecture, optimizer, and other settings were reused from main.py.', '',
             '| Classifier | Held-out task | Correct / 12 | Accuracy | Macro F1 |',
             '|---|---|---:|---:|---:|']
    for e in rq3:
        correct = sum(p['actual'] == p['predicted'] for p in e['predictions'])
        lines.append(f"| {e['model']} | {e['held_out_task']} | {correct} | {e['accuracy']:.1%} | {e['macro_f1']:.3f} |")
    lines += ['', '#### Comparison with grouped response-only testing', '',
              '| Classifier | Grouped response-only accuracy | Pooled unseen-task accuracy |',
              '|---|---:|---:|']
    for baseline in rq4['full_response']:
        predictions = [p for e in rq3 if e['model'] == baseline['model'] for p in e['predictions']]
        accuracy = sum(p['actual'] == p['predicted'] for p in predictions) / len(predictions)
        lines.append(f"| {baseline['model']} | {baseline['accuracy']:.1%} | {accuracy:.1%} |")
    lines += ['', 'The grouped response-only comparison was rerun using this same data file and settings. RQ3 changes both the task distribution and training-set size (24 instead of 27). Differences cannot therefore be attributed exclusively to domain shift. Each task test contains only four questions; training sets overlap across runs. No statistical significance or universal fingerprint claim is made.', '',
              '#### Recognition by source family', '',
              '| Classifier | Held-out task | GPT recall | Claude recall | Gemini recall |',
              '|---|---|---:|---:|---:|']
    for e in rq3:
        recall = [f"{e['per_class'][label]['recall']:.1%}" for label in project.LABELS]
        lines.append('| ' + ' | '.join([e['model'], e['held_out_task']] + recall) + ' |')
    lines += ['', '### RQ4: What characteristics distinguish different LLMs?', '',
              'We examined response token counts and lexical diversity using the same tokenizer as the classifiers. Lexical diversity is the number of unique tokens divided by the total token count; it depends on response length and is descriptive rather than a controlled comparison.', '',
              '| Family | Mean tokens | Median tokens | Minimum | Maximum | Mean lexical diversity |',
              '|---|---:|---:|---:|---:|---:|']
    for label, stats in rq4['descriptive'].items():
        lines.append(f"| {label} | {stats['mean_tokens']:.1f} | {stats['median_tokens']:.1f} | {stats['min_tokens']} | {stats['max_tokens']} | {stats['mean_lexical_diversity']:.3f} |")
    length = rq4['length_only']
    lines += ['', f"A classifier using only response length achieved {length['accuracy']:.1%} accuracy and {length['macro_f1']:.3f} macro F1. Within each training fold, it calculated each family's mean log token count and predicted the closest mean for held-out responses. It used the same grouped folds without test-based threshold tuning.", '',
              f"We also retrained both networks on only the first {prefix} response tokens. Every record has at least {prefix} tokens, so all ablation inputs have the same length. Folds and training seeds were unchanged.", '',
              '| Classifier | Full-response accuracy | Equal-length prefix accuracy | Change (percentage points) |',
              '|---|---:|---:|---:|']
    for full, shortened in zip(rq4['full_response'], rq4['prefix_ablation']):
        lines.append(f"| {full['model']} | {full['accuracy']:.1%} | {shortened['accuracy']:.1%} | {100*(shortened['accuracy']-full['accuracy']):+.1f} |")
    lines += ['', 'Length-only performance indicates how much source information is present in response length in this collection. The prefix ablation changes content and vocabulary as well as length, so it cannot isolate a causal effect of length or prove what the original networks used. The tokenizer removes punctuation, capitalization, and most formatting; those signals were not directly evaluated. Shared chat history, extraction artifacts, the clipped Gemini response, and the small dataset remain limitations.', '',
              '#### Reproducibility', '',
              f"This run used {results['config']['epochs']} epochs and the settings imported from main.py. Dataset SHA-256: `{results['config']['data_sha256']}`. `results/rq3_rq4.json` stores the complete configuration, predictions, confusion matrices, per-class metrics, and training losses."]
    (output_dir / 'rq3_rq4_sections.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--prefix-tokens', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.epochs < 1 or args.prefix_tokens < 1 or args.prefix_tokens > project.MAX_LENGTH:
        parser.error('Use positive epochs and prefix length no greater than MAX_LENGTH.')
    try:
        rows = project.load_data(args.data)
    except ValueError as error:
        parser.error(str(error))
    folds = project.make_folds(rows)
    tokens = {r['record_id']: project.get_tokens(r, 'output_only') for r in rows}
    lengths = {key: len(value) for key, value in tokens.items()}
    if min(lengths.values()) < args.prefix_tokens:
        parser.error('A response is shorter than the prefix length. Choose a documented smaller value.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    descriptive = {}
    for label in project.LABELS:
        group = [r for r in rows if r['LLM_name'] == label]
        counts = [lengths[r['record_id']] for r in group]
        diversity = [len(set(tokens[r['record_id']])) / lengths[r['record_id']] for r in group]
        descriptive[label] = {'mean_tokens': statistics.mean(counts), 'median_tokens': statistics.median(counts),
                              'min_tokens': min(counts), 'max_tokens': max(counts),
                              'mean_lexical_diversity': statistics.mean(diversity)}
    config = {'epochs': args.epochs, 'seed': project.SEED, 'prefix_tokens': args.prefix_tokens,
              'labels': project.LABELS, 'batch_size': project.BATCH_SIZE,
              'learning_rate': project.LEARNING_RATE, 'max_length': project.MAX_LENGTH,
              'grouped_test_prompt_ids': folds, 'python': sys.version.split()[0],
              'torch': torch.__version__, 'numpy': np.__version__,
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'main_py_sha256': hashlib.sha256((ROOT / 'main.py').read_bytes()).hexdigest(),
              'truncated_response_ids': [key for key, length in lengths.items() if length > project.MAX_LENGTH]}
    results = {'config': config, 'rq3': run_rq3(rows, args.epochs),
               'rq4': {'descriptive': descriptive, 'length_only': run_length_only(rows, folds, lengths),
                       'full_response': run_grouped(rows, folds, args.epochs),
                       'prefix_ablation': run_grouped(rows, folds, args.epochs, args.prefix_tokens)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'rq3_rq4.json').write_text(json.dumps(results, indent=2) + '\n')
    write_sections(results, args.output_dir)
    print('Saved rq3_rq4.json and rq3_rq4_sections.md.', flush=True)


if __name__ == '__main__':
    main()
