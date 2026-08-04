"""Render single-instrument MIDI files with NSynth note samples."""

from pathlib import Path
from typing import Optional, Union
import warnings

import mido
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf


PathLike = Union[str, Path]


class TimbreValidationError(ValueError):
    """Raised when an NSynth timbre parquet is invalid."""


class MidiValidationError(ValueError):
    """Raised when a MIDI file cannot be rendered safely."""


class NoteSynthesizer:
    """Render MIDI notes from one NSynth timbre."""

    _SUSTAIN_LOOP_START_SECONDS = 2.0
    _NSYNTH_NOTE_OFF_SECONDS = 3.0

    def __init__(
        self,
        timbre_path: PathLike,
        *,
        sample_rate: Optional[int] = None,
        attack_ms: float = 0.0,
        release_ms: float = 500.0,
        random_ratio: float = 0.1,
        seed: Optional[int] = 42,
        normalize: bool = False,
        sustain_crossfade_ms: float = 50.0,
        release_crossfade_ms: float = 20.0,
    ) -> None:
        self._validate_parameters(
            sample_rate=sample_rate,
            attack_ms=attack_ms,
            release_ms=release_ms,
            random_ratio=random_ratio,
            seed=seed,
            normalize=normalize,
            sustain_crossfade_ms=sustain_crossfade_ms,
            release_crossfade_ms=release_crossfade_ms,
        )

        path = Path(timbre_path)
        if not path.is_file():
            raise FileNotFoundError(f"Timbre parquet does not exist: {path}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise TimbreValidationError(f"Cannot read timbre parquet: {path}") from exc

        required_columns = {"pitch", "velocity", "sample_rate", "audio"}
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise TimbreValidationError(
                f"Timbre parquet is missing required columns: {missing_columns}"
            )
        if frame.empty:
            raise TimbreValidationError("Timbre parquet is empty")

        row_sample_rates = set()
        nested_sample_rates = set()
        raw_rows = []
        seen_keys = set()

        for row_number, row in frame.iterrows():
            pitch = row["pitch"]
            velocity = row["velocity"]
            if not isinstance(pitch, (int, np.integer)) or not 0 <= int(pitch) <= 127:
                raise TimbreValidationError(f"Invalid pitch at row {row_number}: {pitch}")
            if not isinstance(velocity, (int, np.integer)) or not 1 <= int(velocity) <= 127:
                raise TimbreValidationError(
                    f"Invalid velocity at row {row_number}: {velocity}"
                )

            key = (int(pitch), int(velocity))
            if key in seen_keys:
                raise TimbreValidationError(
                    f"Found duplicate pitch/velocity pair: {key}"
                )
            seen_keys.add(key)

            try:
                row_sample_rate = int(row["sample_rate"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise TimbreValidationError(
                    f"Invalid sample rate at row {row_number}"
                ) from exc
            if row_sample_rate <= 0:
                raise TimbreValidationError(
                    f"Invalid sample rate at row {row_number}: {row_sample_rate}"
                )

            audio = row["audio"]
            if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
                raise TimbreValidationError(
                    f"Invalid audio record at row {row_number}; expected array and sampling_rate"
                )
            try:
                nested_sample_rate = int(audio["sampling_rate"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise TimbreValidationError(
                    f"Invalid audio sample rate at row {row_number}"
                ) from exc
            if nested_sample_rate != row_sample_rate:
                raise TimbreValidationError(
                    f"Audio sample rate does not match sample_rate at row {row_number}"
                )

            audio_array = np.asarray(audio["array"], dtype=np.float32)
            if audio_array.ndim != 1 or audio_array.size == 0:
                raise TimbreValidationError(
                    f"Audio array at row {row_number} must be non-empty mono audio"
                )
            if not np.all(np.isfinite(audio_array)):
                raise TimbreValidationError(
                    f"Audio array at row {row_number} must contain only finite values"
                )

            row_sample_rates.add(row_sample_rate)
            nested_sample_rates.add(nested_sample_rate)
            raw_rows.append((key, audio_array))

        if len(row_sample_rates) != 1 or len(nested_sample_rates) != 1:
            raise TimbreValidationError("All timbre samples must use one sample rate")
        parquet_sample_rate = next(iter(row_sample_rates))
        if sample_rate is not None and sample_rate != parquet_sample_rate:
            raise TimbreValidationError(
                f"Requested sample rate {sample_rate} does not match parquet sample rate "
                f"{parquet_sample_rate}"
            )

        note_off_sample = int(round(self._NSYNTH_NOTE_OFF_SECONDS * parquet_sample_rate))
        for key, audio_array in raw_rows:
            if audio_array.size < note_off_sample:
                raise TimbreValidationError(
                    f"Audio sample {key} must be at least 3 seconds long"
                )

        self.timbre_path = path
        self.timbre_name = path.stem
        self.sample_rate = parquet_sample_rate
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.random_ratio = float(random_ratio)
        self.seed = seed
        self.normalize = normalize
        self.sustain_crossfade_ms = float(sustain_crossfade_ms)
        self.release_crossfade_ms = float(release_crossfade_ms)
        self._rng = np.random.default_rng(seed)
        self._sample_index = dict(raw_rows)
        self._velocities_by_pitch = {}
        for pitch, velocity in self._sample_index:
            self._velocities_by_pitch.setdefault(pitch, []).append(velocity)
        for pitch in self._velocities_by_pitch:
            self._velocities_by_pitch[pitch].sort()

    @staticmethod
    def _validate_parameters(
        *,
        sample_rate: Optional[int],
        attack_ms: float,
        release_ms: float,
        random_ratio: float,
        seed: Optional[int],
        normalize: bool,
        sustain_crossfade_ms: float,
        release_crossfade_ms: float,
    ) -> None:
        if sample_rate is not None and (
            not isinstance(sample_rate, int) or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer or None")
        for name, value in (
            ("attack_ms", attack_ms),
            ("release_ms", release_ms),
            ("random_ratio", random_ratio),
            ("sustain_crossfade_ms", sustain_crossfade_ms),
            ("release_crossfade_ms", release_crossfade_ms),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if sustain_crossfade_ms >= 1000.0:
            raise ValueError("sustain_crossfade_ms must be shorter than 1000 ms")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None")
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a bool")

    def render_audio(self, midi_path: PathLike) -> Optional[np.ndarray]:
        """Validate and render one MIDI file, or return None for missing pitches."""
        notes, midi_end_time = self._load_and_validate_midi(midi_path)
        missing_pitches = sorted(
            {note.pitch for note in notes if note.pitch not in self._velocities_by_pitch}
        )
        if missing_pitches:
            warnings.warn(
                f"Skipping MIDI because the timbre is missing pitches: {missing_pitches}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        resolved_notes = []
        for note in notes:
            available_velocities = self._velocities_by_pitch[note.pitch]
            velocity = min(
                available_velocities,
                key=lambda candidate: (abs(candidate - note.velocity), candidate),
            )
            resolved_notes.append((note, velocity))
        return self._render_notes(resolved_notes, midi_end_time)

    def render(
        self,
        midi_path: PathLike,
        output_path: PathLike,
        *,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """Render one MIDI file to a FLOAT WAV at the complete output path."""
        path = Path(output_path)
        if path.suffix.lower() != ".wav":
            raise ValueError(f"output_path must end with .wav: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {path}")

        audio = self.render_audio(midi_path)
        if audio is None:
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(path),
            audio,
            self.sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        return path

    def _load_and_validate_midi(self, midi_path: PathLike):
        path = Path(midi_path)
        if not path.is_file():
            raise FileNotFoundError(f"MIDI file does not exist: {path}")

        try:
            midi_file = mido.MidiFile(path)
        except Exception as exc:
            raise MidiValidationError(f"Cannot parse MIDI file: {path}") from exc
        if midi_file.type == 2:
            raise MidiValidationError("MIDI format 2 is not supported")

        try:
            midi_data = pretty_midi.PrettyMIDI(str(path))
        except Exception as exc:
            raise MidiValidationError(f"Cannot parse MIDI file: {path}") from exc

        note_instruments = [instrument for instrument in midi_data.instruments if instrument.notes]
        if len(note_instruments) != 1:
            raise MidiValidationError(
                "MIDI must contain exactly one instrument with notes; "
                f"found {len(note_instruments)}"
            )
        instrument = note_instruments[0]
        if instrument.is_drum:
            raise MidiValidationError("drum MIDI is not supported")

        end_time = float(midi_data.get_end_time())
        if not np.isfinite(end_time) or not 0.0 < end_time < 10000.0:
            raise MidiValidationError(
                f"MIDI duration must be finite and between 0 and 10000 seconds: {end_time}"
            )

        if instrument.control_changes:
            warnings.warn(
                "MIDI control changes are ignored, including sustain pedal CC64",
                RuntimeWarning,
                stacklevel=3,
            )
        if instrument.pitch_bends:
            warnings.warn(
                "MIDI pitch bends are ignored",
                RuntimeWarning,
                stacklevel=3,
            )

        notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
        for note in notes:
            if (
                not np.isfinite(note.start)
                or not np.isfinite(note.end)
                or note.start < 0.0
                or note.end <= note.start
            ):
                raise MidiValidationError(
                    f"Invalid note time for pitch {note.pitch}: {note.start}..{note.end}"
                )
            if not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127:
                raise MidiValidationError(
                    f"Invalid MIDI note pitch/velocity: {note.pitch}/{note.velocity}"
                )
        return notes, end_time

    def _render_notes(self, resolved_notes, midi_end_time: float) -> np.ndarray:
        events = []
        output_length = int(round(midi_end_time * self.sample_rate))

        for note, velocity in resolved_notes:
            start_sample = int(round(note.start * self.sample_rate))
            end_sample = int(round(note.end * self.sample_rate))
            duration = max(1, end_sample - start_sample)
            source = self._sample_index[(note.pitch, velocity)]
            attack_samples = min(duration, self._randomized_length(self.attack_ms))
            requested_release = self._randomized_length(self.release_ms)
            native_release_length = max(
                0,
                source.size
                - int(round(self._NSYNTH_NOTE_OFF_SECONDS * self.sample_rate)),
            )
            release_samples = min(requested_release, native_release_length)
            note_length = duration + release_samples
            output_length = max(output_length, start_sample + note_length)
            events.append(
                (
                    start_sample,
                    source,
                    duration,
                    attack_samples,
                    release_samples,
                )
            )

        output = np.zeros(output_length, dtype=np.float32)
        for start_sample, source, duration, attack_samples, release_samples in events:
            note_audio = self._render_note_audio(
                source,
                duration=duration,
                attack_samples=attack_samples,
                release_samples=release_samples,
            )
            output[start_sample : start_sample + note_audio.size] += note_audio

        if self.normalize and output.size:
            peak = float(np.max(np.abs(output)))
            if peak > 0.0:
                output /= peak
        return output

    def _randomized_length(self, milliseconds: float) -> int:
        if milliseconds <= 0.0:
            return 0
        factor = 1.0
        if self.random_ratio > 0.0:
            factor += float(self._rng.uniform(0.0, self.random_ratio))
        return max(0, int(round(milliseconds * 0.001 * self.sample_rate * factor)))

    def _render_note_audio(
        self,
        source: np.ndarray,
        *,
        duration: int,
        attack_samples: int,
        release_samples: int,
    ) -> np.ndarray:
        note_off_sample = int(round(self._NSYNTH_NOTE_OFF_SECONDS * self.sample_rate))
        release_crossfade = min(
            int(round(self.release_crossfade_ms * 0.001 * self.sample_rate)),
            release_samples,
        )

        if duration <= note_off_sample:
            body = source[:duration].copy()
            continuation = source[
                duration : min(source.size, duration + release_crossfade)
            ]
        else:
            sustained = self._render_sustain(source, duration + release_crossfade)
            body = sustained[:duration].copy()
            continuation = sustained[duration:]

        if attack_samples > 0:
            attack_envelope = np.linspace(
                0.0, 1.0, attack_samples, endpoint=True, dtype=np.float32
            )
            body[:attack_samples] *= attack_envelope

        if release_samples <= 0:
            return body

        native_tail = source[
            note_off_sample : note_off_sample + release_samples
        ].copy()
        transition_length = min(
            release_crossfade, continuation.size, native_tail.size
        )
        if transition_length > 1:
            fade_out = np.linspace(
                1.0, 0.0, transition_length, endpoint=True, dtype=np.float32
            )
            fade_in = 1.0 - fade_out
            native_tail[:transition_length] = (
                continuation[:transition_length] * fade_out
                + native_tail[:transition_length] * fade_in
            )
        elif transition_length == 1:
            native_tail[0] = continuation[0]

        final_fade_length = min(release_crossfade, native_tail.size)
        if final_fade_length > 1:
            final_fade = np.linspace(
                1.0, 0.0, final_fade_length, endpoint=True, dtype=np.float32
            )
            native_tail[-final_fade_length:] *= final_fade
        native_tail[-1] = 0.0
        return np.concatenate((body, native_tail)).astype(np.float32, copy=False)

    def _render_sustain(self, source: np.ndarray, length: int) -> np.ndarray:
        note_off_sample = int(round(self._NSYNTH_NOTE_OFF_SECONDS * self.sample_rate))
        if length <= note_off_sample:
            return source[:length].copy()

        loop_start = int(round(self._SUSTAIN_LOOP_START_SECONDS * self.sample_rate))
        loop = source[loop_start:note_off_sample]
        crossfade = min(
            int(round(self.sustain_crossfade_ms * 0.001 * self.sample_rate)),
            max(0, loop.size - 1),
        )
        output = np.empty(length, dtype=np.float32)
        output[:note_off_sample] = source[:note_off_sample]
        filled = note_off_sample

        if crossfade == 0:
            while filled < length:
                count = min(loop.size, length - filled)
                output[filled : filled + count] = loop[:count]
                filled += count
            return output

        fade_out = np.linspace(
            1.0, 0.0, crossfade, endpoint=True, dtype=np.float32
        )
        fade_in = 1.0 - fade_out
        while filled < length:
            append_start = filled - crossfade
            write_count = min(loop.size, length - append_start)
            overlap_count = min(crossfade, write_count)
            output[append_start : append_start + overlap_count] = (
                output[append_start : append_start + overlap_count]
                * fade_out[:overlap_count]
                + loop[:overlap_count] * fade_in[:overlap_count]
            )
            remainder = write_count - overlap_count
            if remainder > 0:
                output[
                    append_start + overlap_count : append_start + write_count
                ] = loop[overlap_count : overlap_count + remainder]
            new_filled = append_start + write_count
            if new_filled <= filled:
                raise RuntimeError("Sustain loop did not advance")
            filled = new_filled
        return output
