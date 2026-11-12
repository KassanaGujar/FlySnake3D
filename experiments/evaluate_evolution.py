"""Did evolution work, and how does the result compare? All on seeds never used in training.

A. Held-out progress   Each saved champion (and the un-evolved starting genome)
                       plays the same 64 held-out seeds, 300 moves each, with the
                       training rules. If evolution works, later champions score
                       higher on games that played no part in selecting them.
B. Controls            Random, hand-written reflex, evolved ANN, and (if present)
                       the evolved shuffled-wiring fly, on the same seeds and rules.
C. Time-step check     The final fly champion re-run at the published dt = 0.1 ms.
D. Full episodes       One whole game per seed with the A* benchmark's rules and
                       per-move measurements, so results line up with report 01.
E. Recording           One champion game: a replay for the game viewer plus every
                       spike for the brain view.
F. Figure              Learning curves and held-out scores.

Run:  python -m experiments.evaluate_evolution [--fly fly_main] [--ann ann_main] [--shuffled fly_shuffled]
Writes results/stage5/* , results/replays/stage5/* , results/brain/fly_champion/* ,
       docs/images/evolution_*.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from agents.policies import (ANN_INIT_MEAN, FLY_INIT_MEAN, AnnPolicy, FlyPolicy, RandomPolicy,
                             ReactivePolicy, fly_genome_names)
from evolution.rollout import play_budget, play_episodes
from experiments.evolve import shuffled_wiring, train_config
from game.environment import GameConfig

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "results" / "evolution"
OUT = ROOT / "results" / "stage5"
B = 64
HELD_OUT = 500_000 + np.arange(B)          # training used 10_000 + generation
DECISIONS = 300


def ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [round(float(v), 2) for v in np.percentile(m, [2.5, 97.5])]


def summarise(res: dict) -> dict:
    return {"fitness_mean": round(float(res["fitness"].mean()), 2), "fitness_ci95": ci(res["fitness"]),
            "food_mean": round(float(res["food"].mean()), 2), "food_ci95": ci(res["food"]),
            "deaths_mean": round(float(res["deaths"].mean()), 2),
            "max_length": int(res["max_length"].max())}


def checkpoints(run: Path, k: int = 6) -> list[tuple[str, np.ndarray]]:
    files = sorted(run.glob("champion_gen*.npy"))
    if len(files) > k:
        files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, k).round().astype(int))]
    out = [(f"gen {int(f.stem[-4:])}", np.load(f)) for f in files]
    return out + [("final", np.load(run / "champion.npy"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default="fly_main")
    ap.add_argument("--ann", default="ann_main")
    ap.add_argument("--shuffled", default="fly_shuffled")
    ap.add_argument("--episodes_cap", type=int, default=3000)
    ap.add_argument("--skip_dt_check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = train_config(20)
    fly_cfg = json.loads((EVO / a.fly / "config.json").read_text())
    dt, window = fly_cfg["dt"], fly_cfg["window_ms"]
    report: dict = {"held_out_seeds": [int(HELD_OUT[0]), int(HELD_OUT[-1])], "decisions": DECISIONS,
                    "fly_settings": {"dt_ms": dt, "window_ms": window}}

    from brain.connectome_loader import load_connectome
    conn = load_connectome()
    same = lambda g: np.repeat(np.asarray(g)[None], B, 0)
    fly_final = np.load(EVO / a.fly / "champion.npy")

    # ---- A. held-out progress of the fly
    policy = FlyPolicy(conn, same(FLY_INIT_MEAN), dt=dt, window_ms=window, seed=1)
    prog = {}
    raw = {}
    for name, g in [("start (not evolved)", FLY_INIT_MEAN)] + checkpoints(EVO / a.fly):
        policy.set_genomes(same(g))
        r = play_budget(policy, cfg, HELD_OUT, DECISIONS)
        prog[name], raw[name] = summarise(r), r["fitness"]
        print("fly", name, prog[name], flush=True)
    report["fly_held_out_progress"] = prog
    d = raw["final"] - raw["start (not evolved)"]              # paired by seed
    report["fly_final_minus_start"] = {"mean": round(float(d.mean()), 2), "ci95": ci(d),
                                       "seeds_improved": int((d > 0).sum()), "seeds": B}
    hist = pd.read_csv(EVO / a.fly / "history.csv")
    rho = hist[["generation", "champion_fitness"]].corr(method="spearman").iloc[0, 1]
    report["fly_training_trend"] = {
        "generations": int(len(hist)), "spearman_generation_vs_champion_fitness": round(float(rho), 3),
        "mean_fitness_first5": round(float(hist.mean_fitness.head(5).mean()), 2),
        "mean_fitness_last5": round(float(hist.mean_fitness.tail(5).mean()), 2),
        "champion_fitness_first5": round(float(hist.champion_fitness.head(5).mean()), 2),
        "champion_fitness_last5": round(float(hist.champion_fitness.tail(5).mean()), 2)}

    # ---- B. controls, same seeds and rules
    controls = {"random": RandomPolicy(B, seed=0), "reactive (hand-written)": ReactivePolicy(B)}
    if (EVO / a.ann / "champion.npy").exists():
        controls["ann (evolved)"] = AnnPolicy(same(np.load(EVO / a.ann / "champion.npy")))
        controls["ann start (not evolved)"] = AnnPolicy(same(ANN_INIT_MEAN))
    report["controls_held_out"] = {k: summarise(play_budget(p, cfg, HELD_OUT, DECISIONS))
                                   for k, p in controls.items()}
    has_shuf = (EVO / a.shuffled / "champion.npy").exists()
    if has_shuf:
        del policy
        sp_ = FlyPolicy(conn, same(np.load(EVO / a.shuffled / "champion.npy")), dt=dt,
                        window_ms=window, seed=1, W=shuffled_wiring(conn.W))
        report["controls_held_out"]["fly, shuffled wiring (evolved)"] = summarise(
            play_budget(sp_, cfg, HELD_OUT, DECISIONS))
        # the real-wiring champion's interface on shuffled wiring: does it depend on the wiring?
        sp_.set_genomes(same(fly_final))
        report["controls_held_out"]["fly champion's genome on shuffled wiring"] = summarise(
            play_budget(sp_, cfg, HELD_OUT, DECISIONS))
        del sp_
        policy = FlyPolicy(conn, same(fly_final), dt=dt, window_ms=window, seed=1)
    for k, v in report["controls_held_out"].items():
        print(k, v, flush=True)

    # ---- C. published time step
    if not a.skip_dt_check:
        del policy
        fine = FlyPolicy(conn, same(fly_final), dt=0.1, window_ms=window, seed=1)
        report["fly_final_at_dt_0.1ms"] = summarise(play_budget(fine, cfg, HELD_OUT, DECISIONS))
        print("dt 0.1:", report["fly_final_at_dt_0.1ms"], flush=True)
        del fine
        policy = FlyPolicy(conn, same(fly_final), dt=dt, window_ms=window, seed=1)

    # ---- D. full episodes, A* benchmark rules
    ecfg = GameConfig(width=20, height=20, max_steps=20000)
    seeds = np.arange(B)
    policy.set_genomes(same(fly_final))
    rows, _ = play_episodes(policy, ecfg, seeds, a.episodes_cap, name="fly (evolved)")
    for k, p in controls.items():
        rr, _ = play_episodes(p, ecfg, seeds, 20000, name=k)
        rows += rr
    ep = pd.DataFrame(rows)
    astar = pd.read_csv(ROOT / "results" / "stage2" / "episodes.csv")
    astar = astar[astar.seed < B][ep.columns.intersection(astar.columns)]
    ep = pd.concat([ep, astar], ignore_index=True)
    ep.to_csv(OUT / "episodes.csv", index=False)
    full = {}
    for name, dd in ep.groupby("agent", sort=False):
        full[name] = {"episodes": len(dd), "food_mean": round(float(dd.food_eaten.mean()), 1),
                      "food_ci95": ci(dd.food_eaten), "food_max": int(dd.food_eaten.max()),
                      "steps_mean": round(float(dd.steps.mean()), 1),
                      "outcomes": dd.outcome.value_counts().to_dict(),
                      "self_trap_death_rate": round(float(dd.self_trap_death.mean()), 3)}
    report["full_episodes"] = full

    # ---- E. one recorded champion game (replay + every spike)
    one = FlyPolicy(conn, fly_final[None], dt=dt, window_ms=window, seed=2, record_col=0)
    res, frames = play_episodes(one, ecfg, np.array([0]), a.episodes_cap, name="fly (evolved)",
                                record_col=0)
    rep = ROOT / "results" / "replays" / "stage5"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "fly_champion_seed0.jsonl", "w") as f:
        f.write(json.dumps({"type": "header", "config": asdict(ecfg), "result": res[0]}) + "\n")
        for fr in frames:
            f.write(json.dumps({"type": "frame", **fr}) + "\n")
    write_brain_recording(conn, one.spike_log, frames, window, res[0], ecfg)
    report["recorded_game"] = res[0]

    report["fly_champion_genome"] = dict(zip(fly_genome_names(), np.round(fly_final, 3).tolist()))
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=1))
    figure(a, report, has_shuf)
    print(json.dumps({k: report[k] for k in ("fly_final_minus_start", "fly_training_trend",
                                             "full_episodes")}, indent=1))


def write_brain_recording(conn, spike_log, frames, window_ms, result, cfg):
    out = ROOT / "results" / "brain" / "fly_champion"
    out.mkdir(parents=True, exist_ok=True)
    types = conn.annotations.cell_type.fillna(conn.annotations.hemibrain_type).fillna("?").to_numpy()
    names = ["LEFT", "STRAIGHT", "RIGHT"]
    from game.environment import action_to
    offsets, bins = [0], []
    for i, fired in enumerate(spike_log):
        offsets.append(offsets[-1] + len(fired))
        top = pd.Series(types[fired]).value_counts().head(6) if len(fired) else pd.Series(dtype=int)
        act = ""
        if i + 1 < len(frames):
            a = action_to(tuple(frames[i]["heading"]), tuple(frames[i + 1]["heading"]))
            act = names[int(a)] if a is not None else ""
        bins.append({"t_ms": i * window_ms, "n_spiking": int(len(fired)), "stimuli": [],
                     "top_types": [[k, int(v)] for k, v in top.items()],
                     "caption": f"move {i}: brain chose {act}" if act else f"move {i}"})
    n = len(spike_log)
    np.concatenate(spike_log + [np.zeros(0, np.int32)]).astype(np.int32).tofile(out / "spikes_idx.i32")
    np.asarray(offsets, np.int32).tofile(out / "spikes_off.i32")
    (out / "activity.json").write_text(json.dumps({
        "title": "Evolved fly brain playing Snake (one game, every spike)",
        "bin_ms": window_ms, "n_bins": n, "duration_ms": n * window_ms, "stimuli": [],
        "total_spiking_neuron_bins": int(offsets[-1]), "bins": bins, "result": result,
        "game": {"width": cfg.width, "height": cfg.height,
                 "frames": [{"body": fr["body"], "food": fr["food"]} for fr in frames[:n]]}}))


def figure(a, report, has_shuf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    runs = [("fly brain", a.fly, "tab:green"), ("ANN control", a.ann, "tab:gray")]
    if has_shuf:
        runs.append(("fly, shuffled wiring", a.shuffled, "tab:orange"))
    for label, run, col in runs:
        p = EVO / run / "history.csv"
        if p.exists():
            h = pd.read_csv(p)
            ax[0].plot(h.generation, h.mean_fitness, color=col, label=f"{label}: population mean")
            ax[0].plot(h.generation, h.champion_fitness.rolling(5, min_periods=1).mean(), color=col,
                       ls="--", label=f"{label}: champion (5-gen average)")
    ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness = food - 0.5 x deaths (300 moves)")
    ax[0].set_title("Training (fresh seeds every generation)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    prog = report["fly_held_out_progress"]
    names = list(prog)
    m = [prog[k]["fitness_mean"] for k in names]
    lo = [prog[k]["fitness_mean"] - prog[k]["fitness_ci95"][0] for k in names]
    hi = [prog[k]["fitness_ci95"][1] - prog[k]["fitness_mean"] for k in names]
    ax[1].bar(range(len(names)), m, yerr=[lo, hi], color="tab:green", capsize=3)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("fitness on 64 held-out seeds (95% CI)")
    ax[1].set_title("Fly champions on games never used in training"); ax[1].grid(alpha=.3, axis="y")
    for k, v in report["controls_held_out"].items():
        if k in ("random", "reactive (hand-written)", "ann (evolved)"):
            ax[1].axhline(v["fitness_mean"], ls=":", color="k", lw=.8)
            ax[1].text(len(names) - .5, v["fitness_mean"], " " + k, fontsize=7, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "evolution_results.png", dpi=130)


if __name__ == "__main__":
    main()
