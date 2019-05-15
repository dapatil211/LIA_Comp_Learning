import csv
import os
import argparse
from pprint import pprint
import json


def parse_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        fieldnames = [
            name for name in reader.fieldnames
            if 'loss' in name or 'accuracy' in name
        ]
        vals = [
            {field: float(row[field]) for field in fieldnames} for row in reader
        ]
        return vals[-1]


def extract_results(folder):
    results = {'train': {}, 'outsample_val': {}, 'insample_val': {}, 'test': {}}
    for root, subdirs, files in os.walk(folder):
        if len(files) == 1:
            result = parse_csv(os.path.join(root, files[0]))
            if 'outsample_val' in root:
                results['outsample_val'][root] = result
            elif 'insample_val' in root:
                results['insample_val'][root] = result
            elif 'test' in root:
                results['test'][root] = result
            else:
                results['train'][root] = result

    return results


def best_val(results, field, comparator, init):
    best = init
    best_result = None
    for run, result in results.items():
        val = result[field]
        new_best = comparator(best, val)
        if new_best != best and abs(new_best - 1.0) > 1e-5:
            best = new_best
            best_result = {run: result}
    return best_result


def get_comp(field):
    # print(field)
    if 'loss' in field:
        return min, 100000000
    else:
        return max, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()
    results = extract_results(args.folder)
    fields = {split: results[split].values()[0].keys() for split in results}
    best_results = {
        split: {
            field: best_val(results[split], field, *get_comp(field))
            for field in fields[split]
        } for split in results
    }
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'best_results.json'), 'w') as f:
        json.dump(best_results, f, indent=2)

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()