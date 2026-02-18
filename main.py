#!/usr/bin/env python3
"""CLI tool to transcribe a video file to SRT format using OpenAI Whisper large-v3."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from a video file and convert to 16kHz mono WAV using ffmpeg.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Path to write the extracted WAV audio.
    """
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

    # Check that ffmpeg is available
    if shutil.which("ffmpeg") is None:
        logger.error(
            "ffmpeg is required to extract audio from video files. "
            "Install it with: sudo apt install ffmpeg  (or brew install ffmpeg on macOS)"
        )
        sys.exit(1)

    logger.info(f"Extracting audio from video: {video_path}")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        "16000",  # 16 kHz sample rate
        "-ac",
        "1",  # mono
        "-y",  # overwrite output
        str(output_audio_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        logger.info(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed:\n{e.stderr.decode()}")
        sys.exit(1)


def format_srt(chunks):
    """
    Format a list of chunk dicts into SRT content.

    Each chunk has the shape:
        {"timestamp": (start_seconds, end_seconds), "text": "..."}
    """
    lines = []
    index = 0

    for chunk in chunks:
        start_sec, end_sec = chunk["timestamp"]

        # Whisper may return None for the last end timestamp; fall back to start + 5s
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = start_sec + 5.0

        text = chunk.get("text", "").strip()
        if not text:
            continue

        index += 1
        start_ts = _seconds_to_srt_timestamp(start_sec)
        end_ts = _seconds_to_srt_timestamp(end_sec)

        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def _seconds_to_srt_timestamp(seconds):
    """Convert a float seconds value to an SRT timestamp string HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path, language=None, task="transcribe"):
    """
    Transcribe a video file to SRT format using OpenAI Whisper large-v3.

    Uses model.generate() directly so that Whisper's own long-form
    sequential chunking mechanism handles audio of arbitrary length
    (see Whisper paper §3.8).

    Steps:
    1. Extract audio from video as 16kHz mono WAV via ffmpeg
    2. Load Whisper large-v3 model from Hugging Face
    3. Build full mel-spectrogram features (no truncation)
    4. Call model.generate() with return_timestamps / return_segments
    5. Decode segments and generate SRT file
    6. Clean up temporary audio files
    """
    video_path = Path(video_path)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem

    # Create directories if needed
    os.makedirs("processing", exist_ok=True)
    os.makedirs("captions", exist_ok=True)

    extracted_audio_path = Path(f"processing/{video_stem}_audio.wav")
    output_srt_path = Path(f"captions/{video_stem}.srt")

    try:
        # -----------------------------------------------------------------
        # Step 1: Extract audio from video
        # -----------------------------------------------------------------
        extract_audio_from_video(video_path, extracted_audio_path)

        # -----------------------------------------------------------------
        # Step 2: Load Whisper large-v3 model
        # -----------------------------------------------------------------
        logger.info("Loading Whisper large-v3 model (this may take a moment)...")

        # Enable memory-efficient CUDA allocator to reduce fragmentation
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Using device: {device}  dtype: {torch_dtype}")

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        # -----------------------------------------------------------------
        # Step 3: Load audio and build full mel-spectrogram features
        # -----------------------------------------------------------------
        logger.info("Loading audio and computing mel-spectrogram features...")

        import soundfile as sf

        audio_array, sr = sf.read(str(extracted_audio_path), dtype="float32")

        # Convert stereo to mono if needed (ffmpeg should already output mono,
        # but guard against unexpected formats)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        duration_sec = len(audio_array) / sr
        logger.info(f"Audio duration: {duration_sec:.1f}s  sample rate: {sr}Hz")

        # Build features WITHOUT truncation so the full spectrogram is kept.
        # Whisper's generate() detects inputs > 30 s and automatically
        # enters long-form sequential decoding mode.
        inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(device, dtype=torch_dtype)
        attention_mask = inputs.attention_mask.to(device)

        # -----------------------------------------------------------------
        # Step 4: Transcribe with model.generate()
        # -----------------------------------------------------------------
        logger.info("Transcribing audio (this may take a while for long videos)...")

        generate_kwargs = {
            # Whisper long-form heuristics (paper §4.5)
            "condition_on_prev_tokens": True,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            # Timestamps & segments
            "return_timestamps": True,
            "return_segments": True,
            "num_beams": 1,
            # Do NOT set max_new_tokens — Whisper's long-form sequential
            # decoding adjusts it dynamically per segment to account for
            # the variable-length decoder_input_ids (start tokens +
            # previous-segment prompt tokens).
        }

        if language is not None:
            generate_kwargs["language"] = language
            logger.info(f"Forcing source language: {language}")

        if task == "translate":
            generate_kwargs["task"] = "translate"
            logger.info("Task: translate (target language is English)")
        else:
            generate_kwargs["task"] = "transcribe"

        outputs = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # -----------------------------------------------------------------
        # Step 5: Extract segments and build SRT
        # -----------------------------------------------------------------
        chunks = []

        if isinstance(outputs, dict) and "segments" in outputs:
            # return_segments=True  →  outputs["segments"] is a list
            # (one entry per batch item) of lists of segment dicts.
            all_segments = outputs["segments"][0]  # single-item batch
            logger.info(f"Received {len(all_segments)} segments from Whisper")

            for seg in all_segments:
                text = processor.decode(seg["tokens"], skip_special_tokens=True).strip()
                if text:
                    chunks.append(
                        {
                            "timestamp": (float(seg["start"]), float(seg["end"])),
                            "text": text,
                        }
                    )
        else:
            # Fallback: no structured segments, decode the full sequence
            logger.warning("No structured segments returned; decoding full output")
            token_ids = (
                outputs if not isinstance(outputs, dict) else outputs["sequences"]
            )
            full_text = processor.batch_decode(token_ids, skip_special_tokens=True)[
                0
            ].strip()
            if full_text:
                chunks.append(
                    {
                        "timestamp": (0.0, duration_sec),
                        "text": full_text,
                    }
                )

        logger.info(f"Generating SRT file from {len(chunks)} subtitle entries...")

        if not chunks:
            logger.warning("No speech detected in the audio")
            srt_content = ""
        else:
            srt_content = format_srt(chunks)

        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT saved to {output_srt_path}")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    finally:
        # -----------------------------------------------------------------
        # Step 6: Clean up temporary audio
        # -----------------------------------------------------------------
        if extracted_audio_path.exists():
            logger.info("Cleaning up extracted audio file")
            extracted_audio_path.unlink()

    return str(output_srt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a video file to SRT format using OpenAI Whisper large-v3.",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the video file to transcribe.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Source audio language (e.g. 'english', 'french'). "
            "If omitted, Whisper auto-detects the language."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help=(
            "'transcribe' keeps the original language; "
            "'translate' translates everything into English. "
            "Default: transcribe."
        ),
    )
    args = parser.parse_args()

    result = transcribe_video(
        args.file_path,
        language=args.language,
        task=args.task,
    )
    if result:
        print(f"Transcription complete: {result}")


if __name__ == "__main__":
    main()
