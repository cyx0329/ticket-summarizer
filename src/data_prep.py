"""Generate a synthetic dataset of network-incident tickets paired with
customer-facing notifications.

Real tickets can't leave the building, so this builds a structurally faithful
stand-in: noisy internal tickets (with PII, jargon, internal hostnames) and
the sanitized notification a human would send the customer.

Usage:
    python src/data_prep.py --train-size 400 --eval-size 100
    # or ingest your own CSV:
    python src/data_prep.py --csv tickets.csv --text-col ticket --target-col notification
"""
import argparse
import csv
import json
import random
from pathlib import Path

SERVICES = ["payments-api", "checkout", "auth-service", "search-indexer",
            "billing-worker", "cdn-edge", "notify-queue", "ledger-sync"]
REGIONS = ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-2"]
IMPACTS = ["elevated p99 latency", "5xx spike", "connection timeouts",
           "partial outage", "retry storms", "queue backlog"]
DURATIONS = ["12 minutes", "35 minutes", "1 hour", "2.5 hours", "47 minutes"]
CUSTOMERS = ["Acme Corp", "Globex", "Initech", "Umbrella Health", "Hooli"]

TICKET_TMPL = (
    "[INC-{tid}] {service} {impact} in {region}\n"
    "reporter: {reporter}@internal.example.com\n"
    "host: {host}.internal.example.com (10.{a}.{b}.{c})\n"
    "notes: {notes}\n"
    "customer impact: {customer} seeing {impact}; started ~{duration} ago.\n"
    "mitigation: {mitigation}"
)

NOTES = [
    "on-call paged, dashboard shows red across the board, coffee machine also broken",
    "suspect bad deploy v2.14.3, rolling back now, will update",
    "db primary failover took longer than expected, replicas catching up",
    "looks like a noisy neighbor on the shared cluster, moving workloads",
    "root cause TBD, mitigated via traffic shift, postmortem tomorrow",
]

MITIGATIONS = [
    "rolled back deploy v2.14.3",
    "failed over to standby region",
    "scaled out workers 4x",
    "shifted traffic away from affected AZ",
]

NOTIF_TMPL = (
    "We experienced {impact} affecting {service} in {region} for approximately "
    "{duration}. The issue has been mitigated ({mitigation_short}) and we are "
    "monitoring. No action is required on your side."
)


def make_pair(rng: random.Random) -> dict:
    service = rng.choice(SERVICES)
    region = rng.choice(REGIONS)
    impact = rng.choice(IMPACTS)
    duration = rng.choice(DURATIONS)
    customer = rng.choice(CUSTOMERS)
    tid = rng.randint(10000, 99999)
    reporter = rng.choice(["jdoe", "asmith", "klee", "rpatel"])
    host = f"{service}-{rng.randint(1, 40):02d}"
    a, b, c = rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)
    mitigation = rng.choice(MITIGATIONS)
    ticket = TICKET_TMPL.format(
        tid=tid, service=service, impact=impact, region=region,
        reporter=reporter, host=host, a=a, b=b, c=c,
        notes=rng.choice(NOTES), customer=customer, duration=duration,
        mitigation=mitigation,
    )
    notification = NOTIF_TMPL.format(
        impact=impact, service=service, region=region, duration=duration,
        mitigation_short=mitigation,
    )
    return {"ticket": ticket, "notification": notification}


def from_csv(path: str, text_col: str, target_col: str) -> list[dict]:
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append({"ticket": row[text_col], "notification": row[target_col]})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=400)
    ap.add_argument("--eval-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--text-col", default="ticket")
    ap.add_argument("--target-col", default="notification")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.csv:
        pairs = from_csv(args.csv, args.text_col, args.target_col)
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        split = int(len(pairs) * 0.8)
        train, evl = pairs[:split], pairs[split:]
    else:
        rng = random.Random(args.seed)
        train = [make_pair(rng) for _ in range(args.train_size)]
        evl = [make_pair(rng) for _ in range(args.eval_size)]

    for name, split_pairs in [("train.jsonl", train), ("eval.jsonl", evl)]:
        with open(out / name, "w") as f:
            for p in split_pairs:
                f.write(json.dumps(p) + "\n")
    print(f"wrote {len(train)} train / {len(evl)} eval pairs to {out}/")


if __name__ == "__main__":
    main()
