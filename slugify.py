import re
from pathlib import Path

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def generate_toc(md_file, output_file):
    md_path = Path(md_file)
    base_name = md_path.stem

    toc_lines = [f"\n## {base_name}\n\n"]  # section header per file

    counters = {1: 0, 2: 0, 3: 0}

    with open(md_file, 'r') as f:
        for line in f:
            match = re.match(r'^(#{2,3})\s+(.*)', line)
            if not match:
                continue

            level = len(match.group(1))
            title = match.group(2).strip()

            if level > 3:
                continue

            counters[level] += 1
            for l in range(level + 1, 4):
                counters[l] = 0

            number = ".".join(
                str(counters[i]) for i in range(1, level + 1) if counters[i] > 0
            )

            indent = "   " * (level - 1)
            anchor = slugify(title)

            toc_lines.append(
                f"{indent}{number}. [{title}]({base_name}#{anchor})<br>\n"
            )

    with open(output_file, 'a') as f:
        f.write("".join(toc_lines))


files = [
    "less_important_stuff.md",
    "data_analysis_probability_statistics.md",
    "ml_maths.md",
    "data_prep.md",
    "supervised_learning.md",
    "unsupervised_learning.md",
    "kan.md",
    "transformers.md",
    "MoE.md",
    "famous_problems_machine_learning.md",
    "model_training_techniques.md",
    "performance_metrics.md",
    "NLP.md",
    "reinforcement_learning.md",
    "attention_mechanism.md",
    "llm.md"
]

with open("contents.md", "w") as f:
    f.write("# Contents\n")

for f in files:
    generate_toc(f, "contents.md")