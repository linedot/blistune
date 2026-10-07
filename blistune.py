#!/usr/bin/env python3
"""
BLIS block size / threading autotuner.

Drives one of the BLIS multithreaded test drivers
(<blis_build_dir>/test/3/test_<op>_blis_mt.x) under Optuna, searching over

  * the 4-way thread decomposition (BLIS_JC_NT / BLIS_IC_NT / BLIS_JR_NT / BLIS_IR_NT)
  * the cache block sizes (BLIS_KC_{S,D}, BLIS_MR_IN_MC_{S,D}, BLIS_NR_IN_NC_{S,D})
  * optionally, their max (edge-block) values (BLIS_KC_MAX_{S,D}, ...), searched
    as a headroom above the default so that max >= default always holds

and maximising reported GFLOPs.

Requires a BLIS built with runtime-configurable block sizes; the script checks
the driver binary for the relevant environment variable names (or the format
strings the bli_gks.c override code builds them from) before starting.
"""

import argparse
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError:  # pragma: no cover
    sys.exit("optuna is required: pip install optuna")


# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #

OPERATIONS = ("gemm", "hemm", "herk", "trmm", "trsm")

# precision -> (driver -d char, BLIS env var suffix)
PRECISIONS = {
    "fp32": ("s", "S"),
    "fp64": ("d", "D"),
}

# Thread loops, in the order used by the "jc_ic_jr_ir" categorical encoding.
THREAD_LOOPS = ("JC", "IC", "JR", "IR")

# cli stem -> (BLIS env var stem, default min, default max, default step)
BLOCK_PARAMS = {
    "kc": ("KC", 256, 3072, 32),
    "mr-in-mc": ("MR_IN_MC", 16, 128, 4),
    "nr-in-nc": ("NR_IN_NC", 128, 2048, 32),
}

# Matches a BLIS test-driver result row, e.g.
#   data_gemm_blis( 3, 1:4 ) = [ 4000 4000 4000  123.45 ];
# capturing the dimension columns and the trailing GFLOPs value. Tolerates 1-4
# leading dimension columns (herk/trmm/trsm print fewer). The drivers also print
# an all-zero placeholder row before the results; the parser skips it.
RESULT_RE = re.compile(r"\[\s*((?:\d+\s+){1,4})([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\]")

# How the GFLOPs of the problem-size sweep are combined into one trial score.
AGGREGATES = {
    "max": max,
    "mean": statistics.fmean,
    "min": min,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def thread_configs(num_threads):
    """All (jc, ic, jr, ir) tuples whose product is exactly num_threads."""
    configs = []
    for jc in divisors(num_threads):
        rem_jc = num_threads // jc
        for ic in divisors(rem_jc):
            rem_ic = rem_jc // ic
            for jr in divisors(rem_ic):
                ir = rem_ic // jr
                configs.append((jc, ic, jr, ir))
    return configs


def binary_contains(path, needles, chunk_size=1 << 20):
    """Return the subset of `needles` (bytes) present in the file at `path`."""
    needles = list(needles)
    overlap = max(len(n) for n in needles) - 1
    found = set()
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            buf = tail + chunk
            for needle in needles:
                if needle not in found and needle in buf:
                    found.add(needle)
            if len(found) == len(needles):
                break
            tail = buf[-overlap:] if overlap else b""
    return found


def align_range(low, high, step, label):
    """Clamp `high` down to low + k*step so Optuna doesn't silently adjust it."""
    if high < low:
        raise SystemExit(f"--{label}-max ({high}) must be >= --{label}-min ({low})")
    if step <= 0:
        raise SystemExit(f"--{label}-step must be positive")
    span = ((high - low) // step) * step
    return low, low + span


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="Autotune BLIS block sizes and thread decomposition with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("blis_build_dir", type=Path,
                   help="BLIS build directory (drivers are looked up under <dir>/test/3)")

    p.add_argument("-t", "--threads", type=int, required=True,
                   help="total thread count; the JC/IC/JR/IR product must equal this")
    p.add_argument("-o", "--operation", choices=OPERATIONS, default="gemm",
                   help="which test_<op>_blis_mt.x driver to benchmark")
    p.add_argument("-P", "--precision", choices=sorted(PRECISIONS), default="fp32",
                   help="datatype passed to the driver via -d, and the block-size env var suffix")
    p.add_argument("-s", "--layout", default=None,
                   help="storage layout passed through to the driver as '-s <layout>' (e.g. 'rrr', 'ccc')")

    p.add_argument("-p", "--problem-size", default="4000 8000 500",
                   help="problem size sweep passed to the driver as '-p <value>'")
    p.add_argument("-r", "--repeats", type=int, default=3,
                   help="repetitions per problem size, passed as '-r <value>'")
    p.add_argument("--aggregate", choices=sorted(AGGREGATES), default="max",
                   help="how the GFLOPs of the problem-size sweep are combined into the "
                        "trial score; prefer 'mean' when tuning max block sizes "
                        "(--*-headroom), which only change edge blocks")

    p.add_argument("-n", "--n-trials", type=int, default=150,
                   help="number of Optuna trials")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for Optuna's sampler (reproducible search)")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-benchmark wall-clock limit in seconds")

    p.add_argument("-c", "--cpu-mask", default=None,
                   help="CPU mask for taskset, e.g. '0-35' or '0,2,4-7'. "
                        "If set, the driver is launched via "
                        "'taskset -c <mask>'. Python/Optuna are not constrained.")

    p.add_argument("--omp-places", default=None,
                   help="OMP_PLACES value (default: '{0}:<threads>:1')")
    p.add_argument("--omp-proc-bind", default="true",
                   help="OMP_PROC_BIND value")
    p.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for the benchmark (repeatable)")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="forward the benchmark driver's stdout/stderr to the terminal "
                        "(useful for OMP_DISPLAY_ENV, OMP_DISPLAY_AFFINITY, etc.)")

    p.add_argument("--save-best", type=Path, default=None,
                   help="write the winning configuration to this file as shell exports")
    p.add_argument("--quiet", action="store_true",
                   help="silence Optuna's per-trial log lines")

    group = p.add_argument_group(
        "block size search space",
        "Per-parameter search bounds. --*-min is where the search range starts, "
        "--*-max where it ends, --*-step the granularity. --*-headroom > 0 also "
        "tunes the max (edge-block) value as default + 0..headroom in --*-step "
        "increments; 0 leaves max == default.",
    )
    for stem, (blis_stem, dmin, dmax, dstep) in BLOCK_PARAMS.items():
        var = f"BLIS_{blis_stem}_<S|D>"
        group.add_argument(f"--{stem}-min", type=int, default=dmin,
                           help=f"lowest {var} value to try")
        group.add_argument(f"--{stem}-max", type=int, default=dmax,
                           help=f"highest {var} value to try")
        group.add_argument(f"--{stem}-step", type=int, default=dstep,
                           help=f"{var} increment")
        group.add_argument(f"--{stem}-headroom", type=int, default=0,
                           help=f"how far BLIS_{blis_stem}_MAX_<S|D> may exceed {var} "
                                f"(0 = max not tuned)")

    return p


def parse_extra_env(pairs):
    env = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

class Benchmark:
    def __init__(self, exe, workdir, cmd, base_env, timeout, verbose=False,
                 aggregate="max"):
        self.exe = exe
        self.workdir = workdir
        self.cmd = cmd
        self.base_env = base_env
        self.timeout = timeout
        self.verbose = verbose
        self.aggregate = AGGREGATES[aggregate]

    def run(self, config):
        env = os.environ.copy()
        env.update(self.base_env)
        env.update({k: str(v) for k, v in config.items()})

        try:
            result = subprocess.run(
                self.cmd,
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"  ! timed out after {self.timeout}s", file=sys.stderr)
            if self.verbose:
                if exc.stdout:
                    sys.stdout.write(exc.stdout)
                    sys.stdout.flush()
                if exc.stderr:
                    sys.stderr.write(exc.stderr)
                    sys.stderr.flush()
            return 0.0
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or exc.stdout or "").strip().splitlines()[-3:]
            print(f"  ! driver exited {exc.returncode}: {' / '.join(tail)}", file=sys.stderr)
            if self.verbose:
                if exc.stdout:
                    sys.stdout.write(exc.stdout)
                    sys.stdout.flush()
                if exc.stderr:
                    sys.stderr.write(exc.stderr)
                    sys.stderr.flush()
            return 0.0

        # ---- verbose forwarding ------------------------------------------ #
        # The driver's full stdout/stderr is normally swallowed here. When
        # --verbose is set, write it through so OpenMP diagnostics
        # (OMP_DISPLAY_ENV, OMP_DISPLAY_AFFINITY, OMP_AFFINITY_FORMAT) and the
        # driver's own tables show up in the terminal.
        if self.verbose:
            if result.stdout:
                sys.stdout.write(result.stdout)
                sys.stdout.flush()
            if result.stderr:
                sys.stderr.write(result.stderr)
                sys.stderr.flush()

        # ---- parse GFLOP/s from stdout (always) --------------------------- #
        # Rows whose dimensions are all zero are the drivers' placeholders;
        # they'd drag 'mean' and 'min' down.
        values = [float(gflops) for dims, gflops in RESULT_RE.findall(result.stdout)
                  if any(int(d) for d in dims.split())]
        if not values:
            print("  ! no parseable result rows in driver output", file=sys.stderr)
            return 0.0
        return self.aggregate(values)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    dtype_char, suffix = PRECISIONS[args.precision]

    # ---- locate the driver ------------------------------------------------ #
    workdir = (args.blis_build_dir / "test" / "3").resolve()
    exe_name = f"test_{args.operation}_blis_mt.x"
    exe = workdir / exe_name
    if not exe.is_file():
        raise SystemExit(
            f"benchmark not found: {exe}\n"
            f"(is {args.blis_build_dir} a BLIS build dir, and were the test drivers built?)"
        )
    if not os.access(exe, os.X_OK):
        raise SystemExit(f"benchmark is not executable: {exe}")

    # ---- confirm the binary understands the env vars we intend to set ----- #
    # Per-config overrides leave the literal names (e.g. "BLIS_KC_D") in the
    # binary. The generic override in bli_gks.c builds them at run time from
    # "BLIS_%s_%s" / "BLIS_%s_MAX_%s", so look for those format strings too;
    # unlike a static function's name, they survive inlining and stripping.
    block_env_names = {
        stem: f"BLIS_{blis_stem}_{suffix}"
        for stem, (blis_stem, _, _, _) in BLOCK_PARAMS.items()
    }
    max_env_names = {
        stem: f"BLIS_{blis_stem}_MAX_{suffix}"
        for stem, (blis_stem, _, _, _) in BLOCK_PARAMS.items()
    }
    needles = [name.encode() for name in block_env_names.values()]
    max_needles = [name.encode() for name in max_env_names.values()]
    fmt_needles = [b"BLIS_%s_%s", b"MR_IN_MC"]
    fmt_max_needle = b"BLIS_%s_MAX_%s"
    present = binary_contains(exe, needles + max_needles + fmt_needles + [fmt_max_needle])
    missing = sorted(n.decode() for n in needles if n not in present)
    has_gks_env = all(n in present for n in fmt_needles)
    has_max_env = fmt_max_needle in present or all(n in present for n in max_needles)
    if missing and not has_gks_env:
        raise SystemExit(
            f"this blis library doesn't support dynamic block sizes\n"
            f"  binary:  {exe}\n"
            f"  missing: {', '.join(missing)}\n"
            f"Rebuild BLIS with runtime-configurable block sizes, or point "
            f"--precision at the datatype this build was configured for."
        )

    # ---- search space ----------------------------------------------------- #
    thread_space = thread_configs(args.threads)
    if not thread_space:
        raise SystemExit(f"no valid 4-way thread decomposition for {args.threads} threads")
    thread_choices = ["_".join(str(v) for v in cfg) for cfg in thread_space]

    block_space = {}
    for stem in BLOCK_PARAMS:
        attr = stem.replace("-", "_")
        low = getattr(args, f"{attr}_min")
        high = getattr(args, f"{attr}_max")
        step = getattr(args, f"{attr}_step")
        low, high = align_range(low, high, step, stem)
        block_space[stem] = (low, high, step)

    # Max (edge-block) values are searched as a headroom above the default, in
    # the default's step. That keeps each parameter's range fixed across
    # trials, and max >= default holds by construction (the patched BLIS
    # aborts at init otherwise). Headroom 0 is the old max == default.
    max_space = {}
    for stem in BLOCK_PARAMS:
        headroom = getattr(args, f"{stem.replace('-', '_')}_headroom")
        step = block_space[stem][2]
        if headroom < 0:
            raise SystemExit(f"--{stem}-headroom must be >= 0")
        if headroom == 0:
            continue
        if headroom < step:
            raise SystemExit(f"--{stem}-headroom ({headroom}) must be 0 or at least "
                             f"--{stem}-step ({step})")
        max_space[stem] = (0, (headroom // step) * step, step)

    if max_space and not has_max_env:
        raise SystemExit(
            f"this blis library doesn't read the max block size variables\n"
            f"  binary:  {exe}\n"
            f"  wanted:  {', '.join(max_env_names[stem] for stem in max_space)}\n"
            f"Rebuild BLIS with the MAX-aware bli_gks.c patch, or drop --*-headroom."
        )

    # ---- command + environment -------------------------------------------- #
    cmd = [f"./{exe_name}", "-d", dtype_char, "-p", args.problem_size, "-r", str(args.repeats)]
    if args.layout:
        cmd += ["-s", args.layout]

    # Wrap the driver in taskset if a CPU mask was given. Only the driver is
    # constrained; Python and Optuna keep their normal scheduling freedom.
    if args.cpu_mask:
        cmd = ["taskset", "-c", args.cpu_mask] + cmd

    base_env = {
        "OMP_PLACES": args.omp_places or f"{{0}}:{args.threads}:1",
        "OMP_PROC_BIND": args.omp_proc_bind,
    }
    base_env.update(parse_extra_env(args.env))

    benchmark = Benchmark(exe, workdir, cmd, base_env, args.timeout,
                          verbose=args.verbose, aggregate=args.aggregate)

    # ---- objective --------------------------------------------------------- #
    def objective(trial):
        thread_str = trial.suggest_categorical("thread_config", thread_choices)
        values = dict(zip(THREAD_LOOPS, map(int, thread_str.split("_"))))

        config = {f"BLIS_{loop}_NT": values[loop] for loop in THREAD_LOOPS}
        for stem, (low, high, step) in block_space.items():
            env_name = block_env_names[stem]
            config[env_name] = trial.suggest_int(env_name, low, high, step=step)

        for stem, (low, high, step) in max_space.items():
            headroom = trial.suggest_int(f"{stem}-headroom", low, high, step=step)
            config[max_env_names[stem]] = config[block_env_names[stem]] + headroom

        # The max values are derived rather than Optuna parameters, so keep the
        # exact environment of every trial for the report.
        trial.set_user_attr("config", config)

        if args.verbose:
            print(f"\n[trial {trial.number}] config: "
                  + " ".join(f"{k}={v}" for k, v in sorted(config.items())),
                  file=sys.stderr)

        return benchmark.run(config)

    # ---- run --------------------------------------------------------------- #
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"driver      : {exe}")
    print(f"command     : {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"environment : " + "  ".join(f"{k}={v}" for k, v in base_env.items()))
    if args.cpu_mask:
        print(f"taskset     : -c {args.cpu_mask}")
    print(f"threads     : {args.threads} -> {len(thread_space)} valid JC/IC/JR/IR decompositions")
    for stem, (low, high, step) in block_space.items():
        print(f"{block_env_names[stem]:<20}: {low}..{high} step {step}")
    for stem, (low, high, step) in max_space.items():
        print(f"{max_env_names[stem]:<20}: {block_env_names[stem]} + {low}..{high} step {step}")
    print(f"score       : {args.aggregate} GFLOPs over the problem-size sweep")
    if max_space and args.aggregate == "max":
        print("note        : max block sizes only change edge blocks, which '--aggregate max' "
              "mostly ignores; consider --aggregate mean")
    print(f"trials      : {args.n_trials}")
    if args.verbose:
        print(f"verbose     : on (forwarding driver stdout/stderr)")
    print()

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.seed is not None else None
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # ---- report ------------------------------------------------------------ #
    # Export the environment the best trial actually ran with: thread ways,
    # block sizes and any derived max values (default + headroom).
    exports = [f"{k}={v}" for k, v in study.best_trial.user_attrs["config"].items()]

    print("\n" + "=" * 50)
    print(f"BEST GFLOPs: {study.best_value:.2f}  (trial {study.best_trial.number})")
    print("BEST CONFIGURATION:")
    for line in exports:
        print(f"  {line}")
    print("=" * 50)

    if args.save_best:
        args.save_best.write_text(
            "\n".join(
                [
                    f"# {args.operation} / {args.precision} / {args.threads} threads",
                    f"# {study.best_value:.2f} GFLOPs",
                    f"export OMP_PLACES='{base_env['OMP_PLACES']}'",
                    f"export OMP_PROC_BIND={base_env['OMP_PROC_BIND']}",
                ]
                + [f"export {line}" for line in exports]
            )
            + "\n"
        )
        print(f"\nwrote {args.save_best}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
