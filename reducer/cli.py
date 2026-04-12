import argparse
import json
from dataclasses import asdict

from .config import Config
from .pipeline import compress_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Extractive-first prompt reducer")
    parser.add_argument("--input", type=str, help="Input text file path")
    parser.add_argument("--keep-ratio", type=float, default=0.70)
    parser.add_argument("--strict-mode", action="store_true")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = input("Prompt> ")

    cfg = Config(keep_ratio=args.keep_ratio, strict_mode=args.strict_mode)
    result = compress_prompt(text, cfg)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
