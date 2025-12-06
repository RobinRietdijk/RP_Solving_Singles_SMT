from statistics import mean, stdev
from utils import format_elapsed

def print_outliers(results: dict, z_time_threshold: float = 2.0, z_conflict_threshold: float = 2.0) -> None:
    if results is None:
        return 
    
    by_puzzle = {}
    for rs in results:
        key = (rs["solver"], rs["size"], rs["puzzle"])
        if key not in by_puzzle:
            by_puzzle[key] = {"elapsed": [], "conflicts": []}
        by_puzzle[key]["elapsed"].append(rs["elapsed"])
        by_puzzle[key]["conflicts"].append(rs["statistics"].get("conflicts", 0))

    flattened_results = []
    for (solver, size, puzzle), rs in by_puzzle.items():
        avg_elapsed = mean(rs["elapsed"])
        avg_conflicts = mean(rs["conflicts"])
        flattened_results.append({
            "solver": solver,
            "size": size,
            "puzzle": puzzle,
            "elapsed": avg_elapsed,
            "conflicts": avg_conflicts,
            "runs": len(rs["elapsed"])
        })

    solvers = set()
    by_solver = {}
    for rs in flattened_results:
        key = (rs["solver"], rs["size"])
        if key not in by_solver:
            by_solver[key] = []
        by_solver[key].append(rs)
        solvers.add(rs["solver"])

    for solver in sorted(solvers):
        outliers = []
        for (s, size), rs in by_solver.items():
            if s != solver:
                continue

            if len(rs) < 2:
                continue

            times = [r["elapsed"] for r in rs]
            mn_times = mean(times)
            std_times = stdev(times)
            conflicts = [r["conflicts"] for r in rs]
            mn_conflicts = mean(conflicts)
            std_conflicts = stdev(conflicts)

            if std_times == 0 and std_conflicts == 0:
                continue

            for r in rs:
                z_time = ((r["elapsed"] - mn_times) / std_times) if std_times > 0 else 0.0
                z_conflicts = ((r["conflicts"] - mn_conflicts) / std_conflicts) if std_conflicts > 0 else 0.0
                if z_time >= z_time_threshold or z_conflicts >= z_conflict_threshold:
                    outliers.append({
                        "puzzle": r["puzzle"],
                        "size": size,
                        "runs": r["runs"],
                        "time": r["elapsed"],
                        "conflicts": r["conflicts"],
                        "z_times": z_time,
                        "mean_times": mn_times,
                        "std_times": std_times,
                        "z_conflicts": z_conflicts,
                        "mean_conflicts": mn_conflicts,
                        "std_conflicts": std_conflicts
                    })
        
        if outliers:
            print(f"\nHard puzzles for solver {solver} (z_time >= {z_time_threshold} or z_conflicts >= {z_conflict_threshold})")
            outliers.sort(key=lambda x: (x["size"], -x["z_conflicts"], -x["z_times"]))
            by_size = {}
            for o in outliers:
                key = o["size"]
                if key not in by_size:
                    by_size[key] = []
                by_size[key].append(o)
            
            for size, os in by_size.items():
                sample = os[0]
                print(f"\n{sample['runs']} run(s) on puzzles of size {size}x{size}: mean_time={format_elapsed(sample['mean_times'])} std_time={format_elapsed(sample['std_times'])} " \
                      f"mean_conflicts={sample['mean_conflicts']:.2f} std_conflicts={sample['std_conflicts']:.2f}:\n")
                for o in os:
                    print(f"{o['puzzle']} ({o['size']}x{o['size']}): "
                        f"time={format_elapsed(o['time'])} z_time={o['z_times']:.2f} conflicts={o['conflicts']} z_conflicts={o['z_conflicts']:.2f}")
        else:
            print(f"\nNo outliers found for solver {solver}")