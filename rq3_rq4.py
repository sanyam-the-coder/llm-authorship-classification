import argparse
from collections import Counter
import math
from pathlib import Path
import statistics

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--prefix-tokens', type=int, default=20)
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
    rq3 = run_rq3(rows, args.epochs)
    length_result = run_length_only(rows, folds, lengths)
    full_results = run_grouped(rows, folds, args.epochs)
    prefix_results = run_grouped(rows, folds, args.epochs, args.prefix_tokens)

    print('\nRQ3 results')
    for result in rq3:
        print(f"{result['model']} tested on {result['held_out_task']}: "
              f"accuracy {result['accuracy']:.1%}, macro F1 {result['macro_f1']:.3f}")
        for label in project.LABELS:
            print(f"  {label} recall: {result['per_class'][label]['recall']:.1%}")

    print('\nRQ4 response characteristics')
    for label, stats in descriptive.items():
        print(f"{label}: average tokens {stats['mean_tokens']:.1f}, "
              f"average lexical diversity {stats['mean_lexical_diversity']:.3f}")
    print(f"Length only: accuracy {length_result['accuracy']:.1%}, "
          f"macro F1 {length_result['macro_f1']:.3f}")
    for full, prefix in zip(full_results, prefix_results):
        print(f"{full['model']}: full answer accuracy {full['accuracy']:.1%}, "
              f"first {args.prefix_tokens} tokens accuracy {prefix['accuracy']:.1%}, "
              f"drop {100 * (full['accuracy'] - prefix['accuracy']):.1f} percentage points")


if __name__ == '__main__':
    main()
