import argparse, json, pathlib
from .config import Config
from .experiment import run_experiment

def main():
    ap = argparse.ArgumentParser(prog="multiobj-algos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("--config", type=pathlib.Path, help="optional config JSON")
    r.add_argument("--out",    type=pathlib.Path, default="results.json")

    args = ap.parse_args()
    if args.cmd == "run":
        if args.config and args.config.exists():
            cfg = Config(**json.loads(args.config.read_text()))
        else:
            cfg = Config()  # defaults
        results = run_experiment(cfg)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
