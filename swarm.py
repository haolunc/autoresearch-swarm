"""
Population-based evolutionary search for autoresearch.

Setup tool that creates worktrees and generates per-island program.md files.
The user manually launches Claude in each worktree.

Usage:
    python swarm.py launch --tag mar9 --num-islands 4
    python swarm.py status
    python swarm.py cleanup --tag mar9
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORKTREES_DIR = REPO_ROOT / "worktrees"
POPULATION_FILE = REPO_ROOT / "population.jsonl"

DEFAULT_FOCUSES = [
    "Architecture — attention variants, layer structure, embeddings, normalization, activations, width/depth ratio",
    "Optimization — learning rates, optimizer params, LR schedules, weight decay, momentum, betas",
    "Training dynamics — batch size, grad accumulation, warmup/warmdown, initialization, regularization",
    "Wild card — anything unconventional, radical departures, novel combinations",
]


def get_focus(island_idx: int) -> str:
    if island_idx < len(DEFAULT_FOCUSES):
        return DEFAULT_FOCUSES[island_idx]
    return DEFAULT_FOCUSES[-1]  # wild card for extras


def run_git(*args, check=True, capture=True):
    cmd = ["git"] + list(args)
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=capture, text=True, check=check
    )
    return result.stdout.strip() if capture else None


def generate_program_md(original: str, island: int, num_islands: int) -> str:
    focus = get_focus(island)
    population_path = os.path.relpath(POPULATION_FILE, WORKTREES_DIR / f"island-{island}")

    swarm_section = f"""

---

## Swarm Mode — Island {island}

You are **island-{island}** in a swarm of **{num_islands} islands** doing population-based search.

### Your Exploration Focus

> {focus}

Other islands are exploring different directions — you don't need to cover everything. Lean into your focus area, but don't ignore obvious wins outside it.

### Population Tracking

After each experiment, append a single JSONL line to the shared population file:

```bash
echo '{{"island": {island}, "commit": "<7-char>", "val_bpb": <float>, "memory_gb": <float>, "status": "<keep|discard|crash>", "step": <int>, "summary": "<description>", "timestamp": "<ISO8601>"}}' >> {population_path}
```

Use the same commit hash, val_bpb, memory_gb, status, and description you log to results.tsv. The `step` field is your experiment number (1, 2, 3, ...). The `timestamp` is ISO 8601 format.

### Crossover Protocol (Every 5th Experiment)

On every 5th experiment (step 5, 10, 15, ...), do a **crossover** before your normal experiment:

1. **Read the population file**: `cat {population_path}`
2. **Tournament selection**: From entries belonging to *other* islands (not island-{island}), randomly sample 3 entries and pick the one with the lowest val_bpb.
3. **Read that island's train.py**: `cat ../island-<winner>/train.py`
4. **Combine**: Understand both your current train.py and the winner's. Write a new train.py that combines the best ideas from both.
5. **Mark it**: In your results description and population summary, prefix with "crossover: ..." to indicate this was a crossover experiment.

If the population file is empty or has no entries from other islands, skip crossover and do a normal experiment instead.
"""
    return original + swarm_section


def cmd_launch(args):
    tag = args.tag
    num_islands = args.num_islands

    # Read original program.md
    program_path = REPO_ROOT / "program.md"
    if not program_path.exists():
        print("Error: program.md not found in repo root.", file=sys.stderr)
        sys.exit(1)
    original_program = program_path.read_text()

    # Create worktrees dir
    WORKTREES_DIR.mkdir(exist_ok=True)

    current_head = run_git("rev-parse", "HEAD")

    for i in range(num_islands):
        branch = f"swarm/{tag}/island-{i}"
        worktree = WORKTREES_DIR / f"island-{i}"

        if worktree.exists():
            print(f"Error: {worktree} already exists. Run cleanup first.", file=sys.stderr)
            sys.exit(1)

        # Create branch from current HEAD
        run_git("branch", branch, current_head)
        print(f"  Created branch {branch}")

        # Create worktree
        run_git("worktree", "add", str(worktree), branch)
        print(f"  Created worktree {worktree}")

        # Generate program.md in worktree
        island_program = generate_program_md(original_program, i, num_islands)
        (worktree / "program.md").write_text(island_program)
        print(f"  Generated program.md for island-{i} (focus: {get_focus(i).split(' — ')[0]})")
        print()

    # Initialize population.jsonl if it doesn't exist
    if not POPULATION_FILE.exists():
        POPULATION_FILE.touch()
        print(f"Initialized {POPULATION_FILE}")

    print()
    print("=" * 60)
    print("Swarm ready! Launch Claude in each worktree:")
    print("=" * 60)
    for i in range(num_islands):
        print(f"  cd worktrees/island-{i} && claude")
    print()
    print('Tell each Claude: "follow program.md"')
    print()
    print("With multiple GPUs, run islands in parallel. With one GPU, run one at a time.")
    print()
    print("Check progress:  python swarm.py status")
    print(f"Cleanup:          python swarm.py cleanup --tag {tag}")


def cmd_status(args):
    if not POPULATION_FILE.exists() or POPULATION_FILE.stat().st_size == 0:
        print("No population data yet. (population.jsonl is empty or missing)")
        return

    entries = []
    for line in POPULATION_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        print("No valid entries in population.jsonl")
        return

    # Group by island
    islands = {}
    for e in entries:
        island = e.get("island", "?")
        islands.setdefault(island, []).append(e)

    print(f"Population: {len(entries)} total experiments")
    print("=" * 60)

    global_best = None
    global_best_island = None

    for island_id in sorted(islands.keys()):
        island_entries = islands[island_id]
        valid = [e for e in island_entries if e.get("val_bpb", 0) > 0]
        best = min(valid, key=lambda e: e["val_bpb"]) if valid else None
        latest = island_entries[-1]

        print(f"\nIsland {island_id}: {len(island_entries)} experiments")
        if best:
            print(f"  Best val_bpb: {best['val_bpb']:.6f} (commit {best.get('commit', '?')})")
            if global_best is None or best["val_bpb"] < global_best["val_bpb"]:
                global_best = best
                global_best_island = island_id
        else:
            print("  No successful runs yet")
        print(f"  Latest: {latest.get('summary', '?')}")

    if global_best:
        print()
        print("=" * 60)
        print(f"Global best: {global_best['val_bpb']:.6f} from island-{global_best_island}")
        print(f"  Commit: {global_best.get('commit', '?')}")
        print(f"  Summary: {global_best.get('summary', '?')}")


def cmd_cleanup(args):
    tag = args.tag

    # Find and remove worktrees for this tag
    # List all worktrees to find matching ones
    worktree_output = run_git("worktree", "list", "--porcelain")
    worktree_paths = []
    for line in worktree_output.split("\n"):
        if line.startswith("worktree "):
            path = line[len("worktree "):]
            if "/worktrees/island-" in path:
                worktree_paths.append(path)

    for wt_path in worktree_paths:
        print(f"Removing worktree {wt_path}")
        run_git("worktree", "remove", wt_path, "--force", check=False)

    # Prune worktree references
    run_git("worktree", "prune")

    # Delete branches
    branches_output = run_git("branch", "--list", f"swarm/{tag}/*")
    for branch in branches_output.split("\n"):
        branch = branch.strip()
        if branch:
            print(f"Deleting branch {branch}")
            run_git("branch", "-D", branch, check=False)

    # Clean up empty worktrees dir
    if WORKTREES_DIR.exists() and not any(WORKTREES_DIR.iterdir()):
        WORKTREES_DIR.rmdir()
        print("Removed empty worktrees/ directory")

    print("Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Population-based evolutionary search for autoresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # launch
    launch_parser = subparsers.add_parser("launch", help="Create islands and worktrees")
    launch_parser.add_argument("--tag", required=True, help="Run tag (e.g. mar9)")
    launch_parser.add_argument("--num-islands", type=int, default=4, help="Number of islands (default: 4)")

    # status
    subparsers.add_parser("status", help="Show population status")

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove worktrees and branches")
    cleanup_parser.add_argument("--tag", required=True, help="Run tag to clean up")

    args = parser.parse_args()

    if args.command == "launch":
        cmd_launch(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)


if __name__ == "__main__":
    main()
