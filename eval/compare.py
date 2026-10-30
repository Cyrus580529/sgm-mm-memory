"""Paired comparison of two end-to-end runs (the *.answers.jsonl files written by the replays with --answer).

Questions are matched by id; for each question type it reports both accuracies, how many questions only one run got
right, and an exact two-sided sign test on those discordant questions. A difference is worth acting on only when the
discordant counts are clearly lopsided (small p); re-running an identical configuration also flips a few percent of
answers, so small nets are noise.

usage: python -m eval.compare BASELINE.answers.jsonl VARIANT.answers.jsonl
"""
import collections
import json
import math
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = r.get("qid") or r.get("q") or r.get("question")
        out[key] = r
    return out


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no questions in common")
    groups = collections.defaultdict(list)
    for k in common:
        t = a[k].get("type") or a[k].get("point") or "?"
        groups["ALL"].append(k)
        groups[t].append(k)
    print(f"{len(common)} paired questions ({len(a)} vs {len(b)} in the files)")
    print(f"{'type':26s} {'n':>4s} {'base':>6s} {'var':>6s} {'diff':>6s} {'var+':>5s} {'var-':>5s} {'p':>6s}")
    for t, ks in sorted(groups.items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        sa = [float(a[k]["score"]) for k in ks]
        sb = [float(b[k]["score"]) for k in ks]
        wins = sum(y > x for x, y in zip(sa, sb))
        losses = sum(y < x for x, y in zip(sa, sb))
        ma, mb = sum(sa) / len(ks), sum(sb) / len(ks)
        flag = "  <--" if sign_test(wins, losses) < 0.05 else ""
        print(f"{t:26s} {len(ks):4d} {ma:6.3f} {mb:6.3f} {mb - ma:+6.3f} {wins:5d} {losses:5d} "
              f"{sign_test(wins, losses):6.3f}{flag}")


if __name__ == "__main__":
    main()
