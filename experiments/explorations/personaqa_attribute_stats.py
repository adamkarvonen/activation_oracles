# %%
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

# %%
DATA_PATH = Path("datasets/personaqa_data/shuffled/personas.jsonl")
ATTRIBUTES = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not locate personas file at {DATA_PATH!s}")

# %%
records: list[dict[str, str]] = []
with DATA_PATH.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

print(f"Loaded {len(records)} persona records")

# %%
unique_values: dict[str, set[str]] = {attribute: set() for attribute in ATTRIBUTES}

for record in records:
    for attribute in ATTRIBUTES:
        value = record.get(attribute)
        if value is not None:
            unique_values[attribute].add(value)

unique_counts = {attribute: len(values) for attribute, values in unique_values.items()}

print("Unique value counts by attribute:")
for attribute, count in unique_counts.items():
    print(f"  {attribute}: {count}")

# %%
attribute_labels = list(unique_counts.keys())
unique_values_counts = [unique_counts[label] for label in attribute_labels]

positions = range(len(attribute_labels))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(attribute_labels, unique_values_counts, color="C0")
ax.set_title("Number of Unique Values per Attribute")
ax.set_ylabel("Unique value count")
ax.set_xlabel("Attribute")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.bar_label(bars, padding=3, fmt="%d")
ax.set_ylim(0, max(unique_values_counts) * 1.15)
fig.tight_layout()
plt.show()

# %%
num_attributes = len(ATTRIBUTES)
num_cols = 2
num_rows = (num_attributes + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False)
fig.suptitle("Value Frequency (Top 15) per Attribute")
axes_list = axes.ravel()

for attribute, ax in zip(ATTRIBUTES, axes_list):
    counts = Counter(record[attribute] for record in records if record.get(attribute) is not None)
    most_common = counts.most_common(15)
    labels = [value for value, _ in most_common]
    values = [count for _, count in most_common]

    ax.barh(labels, values, color="#61DDAA")
    ax.set_title(attribute)
    ax.invert_yaxis()

for ax in axes_list[num_attributes:]:
    ax.axis("off")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

# %%
