# vLLM Benchmarks on AAC Slurm

This directory now contains a Slurm-aware runner, [`vllm_runs_slurm.sh`](/home/tarun/run_scr_singularity/vllm_runs_slurm.sh), that automates the flow described in [`manual_benchmarks_guide.md`](/home/tarun/run_scr_singularity/manual_benchmarks_guide.md):

1. Request a Slurm GPU allocation from the AAC login node.
2. Set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR`.
3. Start a Singularity instance from the ROCm vLLM image.
4. Launch `vllm serve` inside the instance.
5. Run `vllm bench serve` inside the same instance.
6. Stop the server and the instance when the run completes.


## Quick Start

From the AAC login node:

```bash
ssh <User Name>@<host>
cd /path/to/this/directory
cp vllm_runs_slurm.conf my_benchmark.conf
vim my_benchmark.conf
chmod +x vllm_runs_slurm.sh
./vllm_runs_slurm.sh -c my_benchmark.conf
```

If you want to skip the confirmation prompt:

```bash
./vllm_runs_slurm.sh -c my_benchmark.conf --yes
```

To resume the latest interrupted run:

```bash
./vllm_runs_slurm.sh -c my_benchmark.conf --resume --yes
```

To resume a specific run directory:

```bash
./vllm_runs_slurm.sh -c my_benchmark.conf --resume ./benchmark_results_slurm/run_YYYYMMDD_HHMMSS --yes
```

## Config Fields

### Model and benchmark settings

- `MODEL_CONFIGS`: one or more entries in the format `"model_path | precisions | tp_sizes | extra_vllm_args"`.
- `INPUT_LENS`: prompt lengths to sweep.
- `OUTPUT_LENS`: output lengths to sweep.
- `CONCURRENCIES`: concurrency values to sweep.
- `NUM_PROMPTS`: number of prompts per benchmark command.
- `REPEATS`: repetitions for each benchmark point.
- `EXTRA_SERVER_ARGS`: extra global flags for `vllm serve`.
- `EXTRA_BENCH_ARGS`: extra global flags for `vllm bench serve`.

Example:

```bash
MODEL_CONFIGS=(
  "meta-llama/Llama-3.1-8B-Instruct | bf16 | 1"
  "Qwen/Qwen2.5-7B-Instruct | bf16,fp16 | 1 | --enforce-eager"
)

INPUT_LENS=(1024 2048)
OUTPUT_LENS=(128 512)
CONCURRENCIES=(8 16 32)
NUM_PROMPTS=200
REPEATS=2
```

For the sweep arrays, use Bash array syntax with spaces between elements. These are all valid:

```bash
INPUT_LENS=(1024 2048)
OUTPUT_LENS=(128 512)
CONCURRENCIES=(8 16 32)
```

The runner also normalizes comma-separated entries if you write them by mistake.

### Slurm settings

- `SLURM_PARTITION`: AAC partition name. The current default is `256C8G1H_MI355X_Ubuntu22`.
- `SLURM_GPUS`: GPUs to allocate.
- `SLURM_TIME`: wall time for the allocation.
- `SLURM_MEM`: memory request. `0` preserves the guide’s “all memory” behavior.
- `SLURM_RESERVATION`: optional reservation if your benchmark window needs one.
- `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_EXTRA_ARGS`: optional site-specific additions.

The script runs `salloc` automatically when started from the login node. If you are already inside a Slurm allocation, it skips `salloc` and starts the benchmark directly on the compute node.

## Output Layout

Results are written under `BASE_OUT_DIR`, for example:

```text
benchmark_results_slurm/
  run_20260403_154500/
    Llama-3.1-8B-Instruct/
      bf16_tp1/
        server.log
        server_cmd.txt
        bench_c16_in1024_out512_rep1.log
        bench_c16_in1024_out512_rep1_cmd.txt
```

While the job is running, the script also prints live progress like:

```text
[05:40:12] Planned benchmark points: 12
[05:40:12] Progress: completed=0 failed=0 remaining=12 total=12
...
[05:44:03] Progress: completed=5 failed=0 remaining=7 total=12
```

Each individual benchmark log also includes `PROGRESS_BEFORE` and `PROGRESS_AFTER` markers so you can inspect progress from the saved logs.

The runner also writes `results_tracker.txt` in each run directory. Resume mode reloads successful benchmark points from that file, skips completed points, and only reruns the remaining ones.

## Running Any Benchmark

1. Pick the model in `MODEL_CONFIGS`.
2. Set `SLURM_GPUS` to match the tensor parallel size you want to test.
3. Set the benchmark sweep arrays (`INPUT_LENS`, `OUTPUT_LENS`, `CONCURRENCIES`).
4. Increase `SLURM_TIME` and `NUM_PROMPTS` for longer production runs.
5. Add any model-specific serve flags in the fourth field of the `MODEL_CONFIGS` entry.
6. Run the script from the AAC login node.

Examples:

```bash
# 1 GPU, small smoke test
./vllm_runs_slurm.sh -c vllm_runs_slurm_smoke.conf --yes

# 4 GPU tensor-parallel run
./vllm_runs_slurm.sh -c tp4_llama.conf --yes
```

For a TP=4 run, set both:

```bash
MODEL_CONFIGS=(
  "meta-llama/Llama-3.1-70B-Instruct | bf16 | 4"
)
SLURM_GPUS=4
```

## Notes

- The checked-in smoke config was validated on AAC with `Qwen/Qwen2.5-0.5B-Instruct`, `SLURM_GPUS=1`, `NUM_PROMPTS=4`, `INPUT_LENS=(32)`, `OUTPUT_LENS=(16)`, and `CONCURRENCIES=(1)`.
- Resume was validated on AAC in two cases: a fully completed 1-point run and a partial 2-point run where one completed point was skipped and only the remaining point was rerun.
- The script assumes `singularity`, `salloc`, `curl`, and cluster network access to the model source are available on AAC.
- If the server never becomes healthy, inspect `server.log` in the run directory first.
- If your team uses a reservation, set `SLURM_RESERVATION` explicitly in the config.
