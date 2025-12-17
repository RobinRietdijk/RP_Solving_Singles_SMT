import os
import math
from statistics import mean, stdev
from utils import format_elapsed
from collections import Counter

def _zscore(x: int|float, mu: float, sigma: float) -> float:
    return (x-mu)/sigma if sigma > 0 else 0.0

def _safe_div(x: int|float, y: int|float) -> float:
    return x/y if y not in (0, 0.0) else 0.0

def rel(a: int|float, b: int|float) -> float:
    return 100*(b-a)/a if a != 0 else float("inf")

def _flatten_results(results: dict) -> list:
    flattened_results = []
    for (solver, size, puzzle), rs in results.items():
        entry = {
            "solver": solver,
            "size": size,
            "puzzle": puzzle,
            "path": rs["path"],
            "runs": len(rs["elapsed"])
        }
        for key in rs.keys(): 
            if key not in entry:
                entry[key] = mean(rs[key])
        flattened_results.append(entry)
    return flattened_results

def _group_by_size(results: list, fields: list) -> dict:
    by_size = {}
    for rs in results:
        key = (rs["solver"], rs["size"])
        if key not in by_size:
            by_size[key] = {
                "solver": rs["solver"],
                "size": rs["size"],
                "puzzles": 0
            }
            for k in fields:
                by_size[key][k] = []
            
        by_size[key]["puzzles"] += 1
        for k in fields:
            by_size[key][k].append(rs.get(k, 0))
    return by_size

def _calculate_stats(results: dict, fields: list) -> list:
    stats = []
    for rs in results.values():
        entry = {
            "solver": rs["solver"],
            "size": rs["size"],
            "puzzles": rs["puzzles"]
        }
        for k in fields:
            values = rs[k]
            entry[f"{k}_mean"] = mean(values) if values else 0.0
            entry[f"{k}_std"] = stdev(values) if len(values) > 1 else 0.0
        stats.append(entry)
    
    stats.sort(key=lambda x: (x["size"]))
    return stats

def _index_by_solver_size(results: list) -> dict:
    return {(rs["solver"], rs["size"]): rs for rs in results}

def find_difficult(results: dict, hard_threshold: float = 3.0, easy_threshold: float = 3.0, do_print: bool = True) -> tuple[dict, dict]:
    if results is None:
        return 
    
    by_puzzle = {}
    for rs in results:
        key = (rs["solver"], rs["size"], rs["puzzle"])
        if key not in by_puzzle:
            by_puzzle[key] = {
                "path": rs["path"], 
                "elapsed": [], 
                "conflicts": [], 
                "decisions": [], 
                "propagations": [], 
                "rlimit_count": [], 
                "max_memory": [],
            }
        by_puzzle[key]["elapsed"].append(rs["elapsed"])
        by_puzzle[key]["conflicts"].append(rs["statistics"].get("conflicts", 0))
        by_puzzle[key]["decisions"].append(rs["statistics"].get("decisions", 0))
        by_puzzle[key]["propagations"].append(rs["statistics"].get("propagations", 0))
        by_puzzle[key]["rlimit_count"].append(rs["statistics"].get("rlimit_count", 0))
        by_puzzle[key]["max_memory"].append(rs["statistics"].get("max_memory", 0))

    flattened_results = _flatten_results(by_puzzle)

    weights = {
        "decisions": 1.0,
        "conflicts": 0.7,
        "propagations": 0.5,
    }

    solvers = set()
    by_solver = {}
    for rs in flattened_results:
        key = (rs["solver"], rs["size"])
        if key not in by_solver:
            by_solver[key] = []
        by_solver[key].append(rs)
        solvers.add(rs["solver"])

    hard_per_solver = {}
    easy_per_solver = {}
    for solver in sorted(solvers):
        hard_per_solver[solver] = []
        easy_per_solver[solver] = []
        hard = []
        easy = []

        for (s, size), rs in by_solver.items():
            if s != solver:
                continue
            if len(rs) < 2:
                continue

            times = [r["elapsed"] for r in rs]
            conflicts = [r["conflicts"] for r in rs]
            decisions = [r["decisions"] for r in rs]
            propagations = [r["propagations"] for r in rs]
            rlimit_counts = [r["rlimit_count"] for r in rs]
            max_memories = [r["max_memory"] for r in rs]

            mu_t = mean(times)
            mu_c = mean(conflicts)
            mu_d = mean(decisions)
            mu_p = mean(propagations)
            mu_r = mean(rlimit_counts)
            mu_m = mean(max_memories)

            std_t = stdev(times) if len(times) > 1 else 0.0
            std_c = stdev(conflicts) if len(conflicts) > 1 else 0.0
            std_d = stdev(decisions) if len(decisions) > 1 else 0.0
            std_p = stdev(propagations) if len(propagations) > 1 else 0.0
            std_r = stdev(rlimit_counts) if len(rlimit_counts) > 1 else 0.0
            std_m = stdev(max_memories) if len(max_memories) > 1 else 0.0

            # High => Bad decision quality
            ratio_c_d = [math.log1p(_safe_div(r["conflicts"], r["decisions"])) for r in rs]
            # Low => Ineffective search/weak propagation
            ratio_p_d = [math.log1p(_safe_div(r["propagations"], r["decisions"])) for r in rs]
            # High => Expensive search
            ratio_r_d = [math.log1p(_safe_div(r["rlimit_count"], r["decisions"])) for r in rs]
            # High => Inefficient solver work
            ratio_t_r = [math.log1p(_safe_div(r["elapsed"], r["rlimit_count"])) for r in rs]

            mu_rcd = mean(ratio_c_d)
            mu_rpd = mean(ratio_p_d)
            mu_rrd = mean(ratio_r_d)
            mu_rtr = mean(ratio_t_r)

            std_rcd = stdev(ratio_c_d) if len(ratio_c_d) > 1 else 0.0
            std_rpd = stdev(ratio_p_d) if len(ratio_p_d) > 1 else 0.0
            std_rrd = stdev(ratio_r_d) if len(ratio_r_d) > 1 else 0.0
            std_rtr = stdev(ratio_t_r) if len(ratio_t_r) > 1 else 0.0

            if std_c == std_d == std_p == std_r == 0:
                continue

            for r in rs:
                z_t = _zscore(r["elapsed"], mu_t, std_t)
                z_c = _zscore(r["conflicts"], mu_c, std_c)
                z_d = _zscore(r["decisions"], mu_d, std_d)
                z_p = _zscore(r["propagations"], mu_p, std_p)
                z_r = _zscore(r["rlimit_count"], mu_r, std_r)
                z_m = _zscore(r["max_memory"], mu_m, std_m)

                z_rcd = _zscore(math.log1p(_safe_div(r["conflicts"], r["decisions"])), mu_rcd, std_rcd)
                z_rpd = _zscore(math.log1p(_safe_div(r["propagations"], r["decisions"])), mu_rpd, std_rpd)
                z_rrd = _zscore(math.log1p(_safe_div(r["rlimit_count"], r["decisions"])), mu_rrd, std_rrd)
                z_rtr = _zscore(math.log1p(_safe_div(r["elapsed"], r["rlimit_count"])), mu_rtr, std_rtr)

                z = {
                    "conflicts": z_c,
                    "decisions": z_d,
                    "propagations": z_p,
                }
                z_ratios = {
                    "z_rcd": z_rcd,
                    "z_rpd": z_rpd,
                    "z_rrd": z_rrd,
                    "z_rtr": z_rtr,
                }
                
                hard_score = math.sqrt(sum(weights[i]*(max(0.0, z[i])**2) for i in weights))
                easy_score = math.sqrt(sum(weights[i]*(max(0.0, -z[i])**2) for i in weights))

                z["elapsed"] = z_t
                z["max_memory"] = z_m
                z["rlimit_count"] = z_r

                stats = {
                    "puzzle": r["puzzle"],
                    "path": r["path"],
                    "size": size,
                    "runs": r["runs"],
                    "time": r["elapsed"],
                    "conflicts": r["conflicts"],
                    "decisions": r["decisions"],
                    "propagations": r["propagations"],
                    "rlimit": r["rlimit_count"],
                    "max_memory": r["max_memory"],
                    "z": z,
                    "hard_score": hard_score,
                    "easy_score": easy_score,
                    "ratios": {
                        "cpd": _safe_div(r["conflicts"], r["decisions"]),
                        "ppd": _safe_div(r["propagations"], r["decisions"]),
                        "rpd": _safe_div(r["rlimit_count"], r["decisions"]),
                        "tpr": _safe_div(r["elapsed"], r["rlimit_count"]),
                    },
                    "z_ratios": z_ratios
                }

                if hard_score >= hard_threshold:
                    hard.append(stats)
                    hard_per_solver[solver].append(stats)
                if easy_score >= easy_threshold:
                    easy.append(stats)
                    easy_per_solver[solver].append(stats["path"])
        
        if not hard and not easy:
            print(f"\nNo outliers found for solver {solver}")
            continue
        
        def print_difficulty(title, items, score, threshold):
            if not items:
                return
            items.sort(key=lambda x: (x["size"], -x[score]))
            print(f"\n{title} for solver {solver} ({score} >= {threshold})")
            
            by_size = {}
            for o in items:
                key = o["size"]
                if key not in by_size:
                    by_size[key] = []
                by_size[key].append(o)
        
            for size, os in by_size.items():
                print(f"\n{os[0]['runs']} run(s) on puzzles of size {size}x{size}:\n")
                for o in os:
                    z = o["z"]
                    ratios = o["ratios"]
                    z_ratios = o["z_ratios"]

                    print(f"{o['puzzle']} ({o['size']}x{o['size']}): \n"
                        f"{score}={o[score]:.2f}\n"
                        f"time={format_elapsed(o['time'])} "
                        f"conflicts={int(o['conflicts'])} "
                        f"decisions={int(o['decisions'])} "
                        f"propagations={int(o['propagations'])} "
                        f"rlimit_count={int(o['rlimit'])} "
                        f"max_memory={o['max_memory']:.2f} \n"
                        f"zT={z['elapsed']:.2f} "
                        f"zC={z['conflicts']:.2f} "
                        f"zD={z['decisions']:.2f} "
                        f"zP={z['propagations']:.2f} "
                        f"zR={z['rlimit_count']:.2f} "
                        f"zM={z['max_memory']:.2f} \n"
                        f"conflicts/decisions={ratios['cpd']:.3g} "
                        f"propagations/decisions={ratios['ppd']:.3g} "
                        f"rlimit_count/decisions={ratios['rpd']:.3g} "
                        f"time/rlimit_count={ratios['tpr']:.3g}\n"
                        f"z_C/D={z_ratios['z_rcd']:.2f} "
                        f"z_P/D={z_ratios['z_rpd']:.2f} "
                        f"z_R/D={z_ratios['z_rrd']:.2f} "
                        f"z_T/R={z_ratios['z_rtr']:.2f}\n")
        
        if do_print:
            print_difficulty("Hard puzzles", hard, "hard_score", hard_threshold)
            print_difficulty("Easy puzzles", easy, "easy_score", easy_threshold)

    return hard_per_solver, easy_per_solver

def print_outlying_puzzles(threshold: float, puzzle_statistics: dict) -> None:
    for n, size in puzzle_statistics.items():
        outliers = []

        for fname, stats in size["puzzles"].items():
            hits = []
            for k, v in stats.items():
                if k.startswith("z_") and isinstance(v, (int, float)) and abs(v) >= threshold:
                    prop = k[2:]
                    raw = stats.get(prop, None)
                    hits.append((k, v, prop, raw))
            
            if hits:
                hits.sort(key=lambda t: abs(t[1]), reverse=True)
                outliers.append((fname, hits, stats))
        if not outliers:
            continue

        print(f"\nOutliers for puzzles of size {n}x{n} (|z| >= {threshold:g}):\n")
        outliers.sort(key=lambda item: abs(item[1][0][1]), reverse=True)

        for fname, hits, stats in outliers:
            parts = []
            for z_key, z_val, prop, raw in hits:
                if raw is None:
                    parts.append(f"{z_key}={z_val:+.2f}")
                else:
                    parts.append(f"{prop}={raw} {z_key}={z_val:+.2f}")
            print(f"{fname} ({n}x{n}): " + " ".join(parts))

def _find_pairs_and_triplets(puzzle: list, n: int) -> tuple[int, int]:
    pairs = 0
    triplets = 0
    for i in range(n):
        j = 0
        while j < n-1:
            if puzzle[i][j] == puzzle[i][j+1]:
                if j < n-2 and puzzle[i][j] == puzzle[i][j+2]:
                    triplets += 1
                    j += 3
                else:
                    pairs += 1
                    j += 2
            else:
                j += 1
        j = 0
        while j < n-1:
            if puzzle[j][i] == puzzle[j+1][i]:
                if j < n-2 and puzzle[j][i] == puzzle[j+2][i]:
                    triplets += 1
                    j += 3
                else:
                    pairs += 1
                    j += 2
            else:
                j += 1
    return pairs, triplets

def _find_isolated(puzzle: list, n: int) -> int:
    isolated = 0
    for i in range(n):
        row_map = {}
        col_map = {}
        k = 0
        while k < n:
            nk = k
            if k < n-1 and puzzle[i][k] == puzzle[i][k+1]:
                if k < n-2 and puzzle[i][k] == puzzle[i][k+2]:
                    nk += 3
                else:
                    nk += 2
            else:
                nk += 1

            v = puzzle[i][k]
            row_map[v] = row_map.get(v, 0) + 1
            k = nk
        
        k = 0
        while k < n:
            nk = k
            if k < n-1 and puzzle[k][i] == puzzle[k+1][i]:
                if k < n-2 and puzzle[k][i] == puzzle[k+2][i]:
                    nk += 3
                else:
                    nk += 2
            else:
                nk += 1

            v = puzzle[k][i]
            col_map[v] = col_map.get(v, 0) + 1
            k = nk
        
        for _, val in row_map.items():
            if val > 1:
                isolated += 1
        for _, val in col_map.items():
            if val > 1:
                isolated += 1
    return isolated

def _cross_duplicates(puzzle: list, n: int) -> int:
    rows = [[False]*n for _ in range(n)]
    cols = [[False]*n for _ in range(n)]

    for i in range(n):
        count = Counter(puzzle[i])
        for j in range(n):
            if count[puzzle[i][j]] > 1:
                rows[i][j] = True

    for i in range(n):
        col = [puzzle[j][i] for j in range(n)]
        count = Counter(col)
        for j in range(n):
            if count[puzzle[j][i]] > 1:
                cols[j][i] = True
    
    return sum(1 for i in range(n) for j in range(n) if rows[i][j] and cols[i][j])

def _value_entropy(line: list) -> float:
    total = len(line)
    if total == 0:
        return 0.0
    c = Counter(line)
    entropy = 0.0
    for count in c.values():
        p = count/total
        entropy -= p*math.log2(p)
    return entropy

def _entropy_difficulty(puzzle: list, n: int) -> float:
    row_entropy = [_value_entropy(puzzle[i]) for i in range(n)]
    col_entropy = [_value_entropy([puzzle[i][j] for i in range(n)]) for j in range(n)]


    row_std = stdev(row_entropy) if n > 1 else 0.0
    col_std = stdev(col_entropy) if n > 1 else 0.0

    mean_entropy = mean(row_entropy+col_entropy)
    entropy_variation = row_std+col_std
    return mean_entropy-entropy_variation

def analyze_puzzle_statistics(puzzles: list) -> dict:
    puzzle_dict = {}
    for path, puzzle, _ in puzzles:
        fname = os.path.splitext(os.path.basename(path))[0]
        n = len(puzzle)
        if n not in puzzle_dict:
            puzzle_dict[n] = {"puzzles": {}}
        
        stats = {}
        stats["path"] = path

        pairs, triplets = _find_pairs_and_triplets(puzzle, n)
        isolated = _find_isolated(puzzle, n)
        cross_duplicates = _cross_duplicates(puzzle, n)
        entropy_difficulty = _entropy_difficulty(puzzle, n)

        stats["pairs"] = pairs
        stats["triplets"] = triplets
        stats["isolated"] = isolated
        stats["cross_duplicates"] = cross_duplicates
        stats["entropy"] = entropy_difficulty
        puzzle_dict[n]["puzzles"][fname] = stats
    for n, size in puzzle_dict.items():
        summary = {}
        for key in ("pairs", "triplets", "isolated", "cross_duplicates", "entropy"):
            values = [p[key] for p in size["puzzles"].values()]
            mu = mean(values)
            std = stdev(values) if len(values) > 1 else 0.0

            summary[key] = { "mean": mu, "std": std }
            for puzzle_stats in size["puzzles"].values():
                x = puzzle_stats[key]
                puzzle_stats[f"z_{key}"] = (x-mu)/std if std > 0 else 0.0
        size["summary"] = summary
    return puzzle_dict


def analyze_sets(results_a: list, results_b: list):
    if results_a is None or results_b is None:
        return 
    
    a_by_puzzle = {}
    for rs in results_a:
        key = (rs["solver"], rs["size"], rs["puzzle"])
        if key not in a_by_puzzle:
            a_by_puzzle[key] = {
                "path": rs["path"], 
                "elapsed": [], 
                "decisions": [],
                "conflicts": [],
                "propagations": [],
                "rlimit": [],
                "max_memory": [],
                "black_cells": []
            }
        a_by_puzzle[key]["elapsed"].append(rs["elapsed"])
        a_by_puzzle[key]["max_memory"].append(rs["statistics"].get("max_memory", 0))
        a_by_puzzle[key]["decisions"].append(rs["statistics"].get("decisions", 0))
        a_by_puzzle[key]["conflicts"].append(rs["statistics"].get("conflicts", 0))
        a_by_puzzle[key]["propagations"].append(rs["statistics"].get("propagations", 0))
        a_by_puzzle[key]["rlimit"].append(rs["statistics"].get("rlimit_count", 0))
        a_by_puzzle[key]["black_cells"].append(rs["puzzle_statistics"].get("black_cells", 0))

    b_by_puzzle = {}
    for rs in results_b:
        key = (rs["solver"], rs["size"], rs["puzzle"])
        if key not in b_by_puzzle:
            b_by_puzzle[key] = {
                "path": rs["path"], 
                "elapsed": [], 
                "decisions": [],
                "conflicts": [],
                "propagations": [],
                "rlimit": [],
                "max_memory": [],
                "black_cells": []
            }
        b_by_puzzle[key]["elapsed"].append(rs["elapsed"])
        b_by_puzzle[key]["max_memory"].append(rs["statistics"].get("max_memory", 0))
        b_by_puzzle[key]["decisions"].append(rs["statistics"].get("decisions", 0))
        b_by_puzzle[key]["conflicts"].append(rs["statistics"].get("conflicts", 0))
        b_by_puzzle[key]["propagations"].append(rs["statistics"].get("propagations", 0))
        b_by_puzzle[key]["rlimit"].append(rs["statistics"].get("rlimit_count", 0))
        b_by_puzzle[key]["black_cells"].append(rs["puzzle_statistics"].get("black_cells", 0))

    fields = ["elapsed", "max_memory", "decisions", "conflicts", "propagations", "rlimit", "black_cells"]
    flattened_results_a = _flatten_results(a_by_puzzle)
    flattened_results_b = _flatten_results(b_by_puzzle)

    a_by_size = _group_by_size(flattened_results_a, fields)
    b_by_size = _group_by_size(flattened_results_b, fields)

    calulated_a = _calculate_stats(a_by_size, fields)
    calulated_b = _calculate_stats(b_by_size, fields)
    
    indexed_a = _index_by_solver_size(calulated_a)
    indexed_b = _index_by_solver_size(calulated_b)

    for key in sorted(indexed_a.keys()):
        if key not in indexed_b:
            continue

        a = indexed_a[key]
        b = indexed_b[key]

        solver, n = key
        print(f"Solver {solver} | Size {n}x{n}\n" \
            f"Puzzles: A={a['puzzles']}, B={b['puzzles']}\n\n"
            f"Black cells: difference (a to b) - {rel(a['black_cells_mean'], b['black_cells_mean']):+.1f}%\n" \
            f"A: mu={a['black_cells_mean']:.2f} sigma={a['black_cells_std']:.2f}\n" \
            f"B: mu={b['black_cells_mean']:.2f} sigma={b['black_cells_std']:.2f}\n"
            f"Elapsed time: difference (a to b) = {rel(a['elapsed_mean'], b['elapsed_mean']):+.1f}%\n" \
            f"A: mu={format_elapsed(a['elapsed_mean'])} sigma={format_elapsed(a['elapsed_std'])}\n" \
            f"B: mu={format_elapsed(b['elapsed_mean'])} sigma={format_elapsed(b['elapsed_std'])}\n" \
            f"Max memory: difference (a to b) = {rel(a['max_memory_mean'], b['max_memory_mean']):+.1f}%\n" \
            f"A: mu={a['max_memory_mean']:.2f} sigma={a['max_memory_std']:.2f}\n" \
            f"B: mu={b['max_memory_mean']:.2f} sigma={b['max_memory_std']:.2f}\n"
            f"Conflicts: difference (a to b) = {rel(a['conflicts_mean'], b['conflicts_mean']):+.1f}%\n" \
            f"A: mu={a['conflicts_mean']:.2f} sigma={a['conflicts_std']:.2f}\n" \
            f"B: mu={b['conflicts_mean']:.2f} sigma={b['conflicts_std']:.2f}\n" \
            f"Propagations: difference (a to b) = {rel(a['propagations_mean'], b['propagations_mean']):+.1f}%\n" \
            f"A: mu={a['propagations_mean']:.2f} sigma={a['propagations_std']:.2f}\n" \
            f"B: mu={b['propagations_mean']:.2f} sigma={b['propagations_std']:.2f}\n" \
            f"Decisions: difference (a to b) = {rel(a['decisions_mean'], b['decisions_mean']):+.1f}%\n" \
            f"A: mu={a['decisions_mean']:.2f} sigma={a['decisions_std']:.2f}\n" \
            f"B: mu={b['decisions_mean']:.2f} sigma={b['decisions_std']:.2f}\n" \
            f"rlimit counts: difference (a to b) = {rel(a['rlimit_mean'], b['rlimit_mean']):+.1f}%\n" \
            f"A: mu={a['rlimit_mean']:.2f} sigma={a['rlimit_std']:.2f}\n" \
            f"B: mu={b['rlimit_mean']:.2f} sigma={b['rlimit_std']:.2f}\n")
