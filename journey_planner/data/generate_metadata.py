import yaml
import json
import csv
import os

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def flatten_plan(plan):
    return {
        'start_point': plan['start_point'],
        'month': plan['month'],
        'num_days': len(plan['days']),
        'num_attractions': sum(len(day['attractions']) for day in plan['days']),
    }

if __name__ == '__main__':
    folder = 'data/'
    all_data = []
    for file in os.listdir(folder):
        if file.endswith('.yaml'):
            data = load_yaml(os.path.join(folder, file))
            all_data.append(data)

    # Write combined JSON for frontend
    with open(os.path.join(folder, 'plans.json'), 'w') as f:
        json.dump(all_data, f, indent=2)

    # Generate metadata CSV
    with open(os.path.join(folder, 'metadata.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start_point', 'month', 'num_days', 'num_attractions'])
        writer.writeheader()
        for plan in all_data:
            writer.writerow(flatten_plan(plan))
