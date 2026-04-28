import argparse
import json
from dataclasses import asdict

from .config import Config
from .pipeline import compress_prompt, run_inference_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Extractive-first prompt reducer")
    parser.add_argument("--input", type=str, help="Input text file path")
    parser.add_argument("--keep-ratio", type=float, default=0.70)
    parser.add_argument("--strict-mode", action="store_true")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["general", "coding", "tool_use", "data", "legal_like"],
        default="general",
    )
    parser.add_argument("--disable-output-normalizer", action="store_true")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run compressed_input -> model -> normalize_output flow",
    )
    parser.add_argument(
        "--mock-model-output",
        type=str,
        default="",
        help="Mock model output text used when --full-pipeline is enabled",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = input("Prompt> ")

    cfg = Config(
        keep_ratio=args.keep_ratio,
        strict_mode=args.strict_mode,
        domain=args.domain,
        use_output_normalizer=not args.disable_output_normalizer,
    )
    if args.full_pipeline:
        def _mock_model(_compressed_input: str) -> str:
            if args.mock_model_output:
                return args.mock_model_output
            return _compressed_input

        result = run_inference_pipeline(text, _mock_model, cfg)
        print(json.dumps(asdict(result), indent=2))
        return

    result = compress_prompt(text, cfg)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
