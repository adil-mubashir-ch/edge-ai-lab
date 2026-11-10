#!/bin/zsh
# Same memory budget, different bits: Qwen3 4B/8B/14B GGUFs (bartowski, imatrix quants) on the M5.
# For each model: speed (llama-bench), perplexity on wikitext-2 test (100 chunks of 512 tokens),
# HellaSwag accuracy (400 tasks). Logs go to logs/<model>.{bench,ppl,hs}.txt.
# usage: ./run.sh [model-file ...]   (default: all of MODELS)
set -u
cd "${0:A:h}"
mkdir -p models logs

MODELS=(
  Qwen_Qwen3-4B-Q4_K_M.gguf:bartowski/Qwen_Qwen3-4B-GGUF
  Qwen_Qwen3-8B-IQ2_M.gguf:bartowski/Qwen_Qwen3-8B-GGUF
  Qwen_Qwen3-4B-Q8_0.gguf:bartowski/Qwen_Qwen3-4B-GGUF
  Qwen_Qwen3-8B-Q4_K_M.gguf:bartowski/Qwen_Qwen3-8B-GGUF
  Qwen_Qwen3-14B-IQ2_M.gguf:bartowski/Qwen_Qwen3-14B-GGUF
  Qwen_Qwen3-14B-Q2_K.gguf:bartowski/Qwen_Qwen3-14B-GGUF
  Qwen_Qwen3-14B-Q4_K_M.gguf:bartowski/Qwen_Qwen3-14B-GGUF
)

for entry in $MODELS; do
  file=${entry%%:*}; repo=${entry#*:}
  if (( $# )) && [[ ! " $* " == *" $file "* ]]; then continue; fi
  name=${file%.gguf}
  echo "=== $name  $(date +%H:%M:%S)"
  [[ -f models/$file ]] || hf download $repo $file --local-dir models >/dev/null || { echo "download failed: $file"; continue; }
  ls -l models/$file | awk '{print "bytes", $5}' > logs/$name.size.txt
  [[ -s logs/$name.bench.txt ]] || llama-bench -m models/$file -p 512 -n 128 -r 3 -o md > logs/$name.bench.txt 2>&1
  [[ -s logs/$name.ppl.txt ]]   || llama-perplexity -m models/$file -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks 100 > logs/$name.ppl.txt 2>&1
  [[ -s logs/$name.hs.txt ]]    || llama-perplexity -m models/$file -f data/hellaswag_val_full.txt --hellaswag --hellaswag-tasks 400 -c 1024 > logs/$name.hs.txt 2>&1
  grep -E "^\|.*(pp512|tg128)" logs/$name.bench.txt
  grep -E "Final estimate" logs/$name.ppl.txt
  tail -2 logs/$name.hs.txt | head -1
done
echo "=== done $(date +%H:%M:%S)"
