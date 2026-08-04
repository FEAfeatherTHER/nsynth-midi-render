"""Command-line entry point for rendering one MIDI file."""

import argparse
import sys
from typing import Optional, Sequence

from nsynth_midi_render import NoteSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one single-instrument MIDI with one NSynth timbre parquet."
    )
    parser.add_argument("--timbre", required=True, help="Path to one timbre parquet")
    parser.add_argument("--midi", required=True, help="Path to one MIDI file")
    parser.add_argument("--output", required=True, help="Complete output .wav path")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--attack-ms", type=float, default=0.0)
    parser.add_argument("--release-ms", type=float, default=500.0)
    parser.add_argument("--random-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sustain-crossfade-ms", type=float, default=50.0)
    parser.add_argument("--release-crossfade-ms", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        synthesizer = NoteSynthesizer(
            args.timbre,
            sample_rate=args.sample_rate,
            attack_ms=args.attack_ms,
            release_ms=args.release_ms,
            random_ratio=args.random_ratio,
            seed=args.seed,
            normalize=args.normalize,
            sustain_crossfade_ms=args.sustain_crossfade_ms,
            release_crossfade_ms=args.release_crossfade_ms,
        )
        output_path = synthesizer.render(
            args.midi,
            args.output,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if output_path is None:
        print("Skipped: the timbre does not contain every required pitch.", file=sys.stderr)
        return 2
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
