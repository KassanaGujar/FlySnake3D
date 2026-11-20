"""Stage 8 pre-test: can per-move dopamine move the MBON readout enough to change moves?

Two copies of the fly_v7_learn champion, both with per-move teaching signals and
the MBON readout, run in LOCKSTEP on the same games: the learning copy chooses
the moves, the frozen copy (learning rate 0) sees exactly the same views, smells
and dopamine. Any difference in their MBON spikes is therefore caused by
learning alone.

MBON readout weights are still 0 in this genome, so play is the stage 7 play.
The question is only whether the signal learning creates is big enough to
matter once evolution gives the MBONs a weight:

  flip_possible   fraction of moves where the learned change in MBON (right -
                  left) spikes, summed over types with the largest allowed weight
                  (|w| = 1), exceeds the distance of the descending-neuron turn
                  signal from the nearest decision threshold.

Run:  python -m experiments.move_feedback_probe [--decisions 300]
Writes results/stage8/move_feedback_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.policies import FlyPolicy
from brain.motor_decoder import N_OUT
from brain.mushroom_body import N_MBON, food_distance, move_signals
from brain.visual_encoder import N_FEAT, view_features
from evolution.rollout import DEATHS, LIFE_SEED_STEP, act
from experiments.evolve import train_config
from game.environment import SnakeGame

ROOT = Path(__file__).resolve().parents[1]
RUN = "fly_v7_learn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--sanity", action="store_true",
                    help="both copies frozen: every 'learned change' must then be exactly 0")
    a = ap.parse_args()
    from brain.connectome_loader import load_connectome
    conn = load_connectome()
    rc = json.loads((ROOT / "results" / "evolution" / RUN / "config.json").read_text())
    B = a.B
    g = np.repeat(np.load(ROOT / "results" / "evolution" / RUN / "champion.npy")[None], B, 0)
    kw = dict(dt=rc["dt"], window_ms=rc["window_ms"], readout_from_ms=rc["readout_from_ms"],
              regular_input=rc["regular_input"], seed=1, learning=True, theta_scale=rc["theta_scale"],
              mbon_readout=True, move_feedback=True)
    on = FlyPolicy(conn, g, plasticity=not a.sanity, **kw)
    off = FlyPolicy(conn, g, plasticity=False, **kw)
    cfg = train_config(20)
    seeds = 600_000 + np.arange(B)                     # not used in training or other probes
    games = [SnakeGame(cfg, int(s)) for s in seeds]
    prev = [None] * B
    lives = np.zeros(B, np.int64)
    food = deaths = 0
    n_danger = n_closer = 0
    for p in (on, off):
        p.new_evaluation()
        p.reset(np.ones(B, bool))
    feats = np.zeros((B, 2, N_FEAT), np.float32)
    w_dn = (on.weights[:, :N_OUT]).cpu().numpy()
    th = (on.theta * on.theta_scale).cpu().numpy()
    rows = []                                          # per decision: arrays over B
    for t in range(a.decisions):
        for b, gm in enumerate(games):
            feats[b], prev[b] = view_features(gm, prev[b])
        acts = act(on, feats, games)
        act(off, feats, games)
        c_on = on.last_counts.view(N_OUT + N_MBON, 2, B).cpu().numpy()
        c_off = off.last_counts.view(N_OUT + N_MBON, 2, B).cpu().numpy()
        lr_on = c_on[:, 1] - c_on[:, 0]                # (types, B) right - left
        lr_off = c_off[:, 1] - c_off[:, 0]
        dn_turn = (lr_on[:N_OUT].T * w_dn).sum(1)      # (B,)
        margin = np.minimum(np.abs(dn_turn - th), np.abs(dn_turn + th))
        dmb = np.abs(lr_on[N_OUT:] - lr_off[N_OUT:]).sum(0)
        rows.append({"t": t, "mbon_spikes_on": c_on[N_OUT:].sum((0, 1)), "mbon_spikes_off": c_off[N_OUT:].sum((0, 1)),
                     "mbon_lr_abs_off": np.abs(lr_off[N_OUT:]).sum(0), "mbon_lr_change": dmb,
                     "dn_turn_abs": np.abs(dn_turn), "margin": margin, "flip_possible": dmb > margin})
        ate = np.zeros(B, bool)
        died = np.zeros(B, bool)
        danger = np.zeros(B, bool)
        closer = np.zeros(B, bool)
        restart = np.zeros(B, bool)
        for b, gm in enumerate(games):
            before = gm.food_eaten
            dist = food_distance(gm)
            gm.step(int(acts[b]))
            ate[b] = gm.food_eaten > before
            danger[b], closer[b] = move_signals(gm, dist)
            closer[b] = closer[b] and not ate[b]
            if gm.done:
                died[b] = gm.outcome in DEATHS
                food += gm.food_eaten
                deaths += died[b]
                lives[b] += 1
                games[b] = SnakeGame(cfg, int(seeds[b]) + LIFE_SEED_STEP * int(lives[b]))
                prev[b] = None
                restart[b] = True
        n_danger += danger.sum()
        n_closer += closer.sum()
        for p in (on, off):
            p.feedback(ate, died, danger, closer)
            if restart.any():
                p.reset(restart)
        if t % 50 == 0:
            r = rows[-1]
            print(f"move {t}: MBON spikes/brain {r['mbon_spikes_on'].mean():.1f}, learned change "
                  f"{r['mbon_lr_change'].mean():.2f}, DN margin {r['margin'].mean():.2f}, "
                  f"flip possible {r['flip_possible'].mean():.2%}", flush=True)
    food += sum(gm.food_eaten for gm in games)

    def seg(lo, hi):
        rr = rows[lo:hi]
        m = lambda k: round(float(np.mean([r[k].mean() for r in rr])), 4)   # noqa: E731
        return {k: m(k) for k in ("mbon_spikes_on", "mbon_spikes_off", "mbon_lr_abs_off", "mbon_lr_change",
                                  "dn_turn_abs", "margin", "flip_possible")}
    n = a.decisions
    ps = on.sim.pscale.float().cpu().numpy()
    out = {"run": RUN, "decisions": n, "brains": B, "seeds": [int(seeds[0]), int(seeds[-1])],
           "food_per_brain": round(food / B, 2), "deaths_per_brain": round(float(deaths) / B, 2),
           "danger_moves_per_brain": round(float(n_danger) / B, 1),
           "closer_moves_per_brain": round(float(n_closer) / B, 1),
           "first_50_moves": seg(0, 50), "last_100_moves": seg(max(0, n - 100), n),
           "plastic_synapses_below_0.9": round(float((ps < 0.9).mean()), 5),
           "mean_depression": round(float((1 - ps).mean()), 5),
           "note": "MBON weights are 0 here, so moves are the stage 7 champion's; the numbers say how much "
                   "learning COULD change a move once evolution weights the MBONs (|w| up to 1)."}
    print(json.dumps(out, indent=1), flush=True)
    if a.sanity:
        return
    d = ROOT / "results" / "stage8"
    d.mkdir(parents=True, exist_ok=True)
    (d / "move_feedback_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
