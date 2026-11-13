import argparse
from pathlib import Path
from needle_bench.charts import generate_all_charts

p = argparse.ArgumentParser()
p.add_argument("csv", type=Path)
p.add_argument("--out", type=Path, default=Path("reports/figures/manual"))
a = p.parse_args()
generate_all_charts(a.csv, a.out)
print(a.out)
