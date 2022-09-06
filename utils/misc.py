import csv
import pandas as pd
with open('/datamirror/videos_dataset/kinetics/kinetics_400_labels.csv', mode='r') as f:
    reader = csv.reader(f)
    label_to_name = {rows[0]:rows[1] for rows in reader}
    name_to_label = {v: k for k, v in label_to_name.items()}

def get_name(label: int or str) -> str:
    return label_to_name[str(label)]
    
def get_label(name: str) -> int: 
    return int(name_to_label[name])


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    y = x - np.max(x)
    return np.exp(y) / np.exp(y).sum()