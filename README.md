What is this?
=============
Python script to optimize block sizes for blis. Requires a modified BLIS source that allows setting block sizes dynamically via environment variables.


Usage:
------

```
usage: blistune.py [-h] -t THREADS [-o {gemm,hemm,herk,trmm,trsm}] [-P {fp32,fp64}] [-s LAYOUT] [-p PROBLEM_SIZE] [-r REPEATS] [-n N_TRIALS] [--seed SEED] [--timeout TIMEOUT] [--omp-places OMP_PLACES] [--omp-proc-bind OMP_PROC_BIND]
                   [-e KEY=VALUE] [--save-best SAVE_BEST] [--quiet] [--kc-min KC_MIN] [--kc-max KC_MAX] [--kc-step KC_STEP] [--mr-in-mc-min MR_IN_MC_MIN] [--mr-in-mc-max MR_IN_MC_MAX] [--mr-in-mc-step MR_IN_MC_STEP]
                   [--nr-in-nc-min NR_IN_NC_MIN] [--nr-in-nc-max NR_IN_NC_MAX] [--nr-in-nc-step NR_IN_NC_STEP]
                   blis_build_dir

Autotune BLIS block sizes and thread decomposition with Optuna.

positional arguments:
  blis_build_dir        BLIS build directory (drivers are looked up under <dir>/test/3)

options:
  -h, --help            show this help message and exit
  -t, --threads THREADS
                        total thread count; the JC/IC/JR/IR product must equal this
  -o, --operation {gemm,hemm,herk,trmm,trsm}
                        which test_<op>_blis_mt.x driver to benchmark (default: gemm)
  -P, --precision {fp32,fp64}
                        datatype passed to the driver via -d, and the block-size env var suffix (default: fp32)
  -s, --layout LAYOUT   storage layout passed through to the driver as '-s <layout>' (e.g. 'rrr', 'ccc') (default: None)
  -p, --problem-size PROBLEM_SIZE
                        problem size sweep passed to the driver as '-p <value>' (default: 4000 8000 500)
  -r, --repeats REPEATS
                        repetitions per problem size, passed as '-r <value>' (default: 3)
  -n, --n-trials N_TRIALS
                        number of Optuna trials (default: 150)
  --seed SEED           seed for Optuna's sampler (reproducible search) (default: None)
  --timeout TIMEOUT     per-benchmark wall-clock limit in seconds (default: None)
  --omp-places OMP_PLACES
                        OMP_PLACES value (default: '{0}:<threads>:1') (default: None)
  --omp-proc-bind OMP_PROC_BIND
                        OMP_PROC_BIND value (default: true)
  -e, --env KEY=VALUE   extra environment variable for the benchmark (repeatable) (default: [])
  --save-best SAVE_BEST
                        write the winning configuration to this file as shell exports (default: None)
  --quiet               silence Optuna's per-trial log lines (default: False)

block size search space:
  Per-parameter search bounds. --*-min is where the search range starts, --*-max where it ends, --*-step the granularity.

  --kc-min KC_MIN       lowest BLIS_KC_<S|D> value to try (default: 256)
  --kc-max KC_MAX       highest BLIS_KC_<S|D> value to try (default: 3072)
  --kc-step KC_STEP     BLIS_KC_<S|D> increment (default: 32)
  --mr-in-mc-min MR_IN_MC_MIN
                        lowest BLIS_MR_IN_MC_<S|D> value to try (default: 16)
  --mr-in-mc-max MR_IN_MC_MAX
                        highest BLIS_MR_IN_MC_<S|D> value to try (default: 128)
  --mr-in-mc-step MR_IN_MC_STEP
                        BLIS_MR_IN_MC_<S|D> increment (default: 4)
  --nr-in-nc-min NR_IN_NC_MIN
                        lowest BLIS_NR_IN_NC_<S|D> value to try (default: 128)
  --nr-in-nc-max NR_IN_NC_MAX
                        highest BLIS_NR_IN_NC_<S|D> value to try (default: 2048)
  --nr-in-nc-step NR_IN_NC_STEP
                        BLIS_NR_IN_NC_<S|D> increment (default: 32)
```



Example running blistune:
-------------------------

```
python ./blistune.py -t 8 -o gemm -P fp64 -p "1000 2000 500" -r 4 -n 300 --omp-places "{0}:8:1" --omp-proc-bind=false --save-best avx512_3vx8.sh /data/tmp/blis-avx512-3vx8-with-knl-packm --kc-min 32 --kc-max 256 --kc-step 4 --mr-in-mc-min 40 --mr-in-mc-max 120 --mr-in-mc-step 2
driver      : /data/tmp/blis-avx512-3vx8-with-knl-packm/test/3/test_gemm_blis_mt.x
command     : ./test_gemm_blis_mt.x -d d -p '1000 2000 500' -r 4
environment : OMP_PLACES={0}:8:1  OMP_PROC_BIND=false
threads     : 8 -> 20 valid JC/IC/JR/IR decompositions
BLIS_KC_D           : 32..256 step 4
BLIS_MR_IN_MC_D     : 40..120 step 2
BLIS_NR_IN_NC_D     : 128..2048 step 32
trials      : 300

[I 2026-09-11 14:00:34,601] A new study created in memory with name: no-name-5f790061-3987-4de4-974d-cbfa45372ac1
[I 2026-09-11 14:00:35,159] Trial 0 finished with value: 449.29 and parameters: {'thread_config': '1_4_2_1', 'BLIS_KC_D': 104, 'BLIS_MR_IN_MC_D': 66, 'BLIS_NR_IN_NC_D': 800}. Best is trial 0 with value: 449.29.
[I 2026-09-11 14:00:35,691] Trial 1 finished with value: 508.09 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 212, 'BLIS_MR_IN_MC_D': 66, 'BLIS_NR_IN_NC_D': 224}. Best is trial 1 with value: 508.09.
[...]
[I 2026-09-11 14:03:15,823] Trial 298 finished with value: 509.36 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 208, 'BLIS_MR_IN_MC_D': 44, 'BLIS_NR_IN_NC_D': 1312}. Best is trial 202 with value: 533.02.
[I 2026-09-11 14:03:16,362] Trial 299 finished with value: 518.96 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 224, 'BLIS_MR_IN_MC_D': 50, 'BLIS_NR_IN_NC_D': 736}. Best is trial 202 with value: 533.02.

==================================================
BEST GFLOPs: 533.02  (trial 202)
BEST CONFIGURATION:
  BLIS_JC_NT=2
  BLIS_IC_NT=2
  BLIS_JR_NT=2
  BLIS_IR_NT=1
  BLIS_KC_D=256
  BLIS_MR_IN_MC_D=50
  BLIS_NR_IN_NC_D=1408
==================================================

wrote avx512_3vx8.sh
```

# License

blistune is distributed under the terms of both the MIT license and the GNU General Public License v3.0. Users may choose either license, at their option.

All new contributions must be made under both the MIT and GNU General Public License v3.0.

See LICENSE-GPL-3.0, LICENSE-MIT for details.

SPDX-License-Identifier: MIT OR GPL-3.0-or-later
