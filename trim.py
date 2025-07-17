#!/usr/bin/env python3
"""Standalone script to trim a tracking CSV file."""
import argparse
from tracking_analysis.reader import preprocess_csv


def main():
    parser = argparse.ArgumentParser(description="Trim a tracking CSV file")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("output", nargs="?", default="./results/trimmed.csv",
                        help="Path for trimmed CSV")
    parser.add_argument(
        "--summary",
        default="./results/summary.txt",
        help="Summary file for removed lines and stats",
    )
    args = parser.parse_args()
    preprocess_csv(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
