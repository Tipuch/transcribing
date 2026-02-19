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

CHUNK_DURATION_SEC = 120  # 2 minutes per chunk


def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from a video file and convert to 16kHz mono WAV using ffmpeg.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Path to write the extracted WAV audio.
    """
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

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
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
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


def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return float(result.stdout.decode().strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"ffprobe failed: {e}")
        sys.exit(1)


def split_audio_into_chunks(audio_path, chunk_duration_sec, output_dir):
    """
    Split an audio file into fixed-length chunks using ffmpeg.

    Returns a list of (chunk_path, offset_seconds) tuples.
    """
    total_duration = get_audio_duration(audio_path)
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    chunk_paths = []

    start = 0.0
    chunk_index = 0

    while start < total_duration:
        chunk_path = output_dir / f"{audio_path.stem}_chunk{chunk_index:04d}.wav"
        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(chunk_duration_sec),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(chunk_path),
        ]
        try:
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg chunk split failed:\n{e.stderr.decode()}")
            sys.exit(1)

        chunk_paths.append((chunk_path, start))
        logger.info(
            f"  Chunk {chunk_index}: {start:.1f}s – {min(start + chunk_duration_sec, total_duration):.1f}s"
        )
        start += chunk_duration_sec
        chunk_index += 1

    logger.info(
        f"Split audio into {len(chunk_paths)} chunk(s) of up to {chunk_duration_sec}s each"
    )
    return chunk_paths


def transcribe_chunk(audio_path, processor, model, device, torch_dtype, language, task):
    """
    Transcribe a single audio chunk and return a list of segment dicts
    with keys 'start', 'end', 'text' (timestamps relative to chunk start).
    """
    import soundfile as sf

    audio_array, sr = sf.read(str(audio_path), dtype="float32")

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

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

    generate_kwargs = {
        "return_timestamps": True,
        "return_segments": True,
    }

    if language is not None:
        generate_kwargs["language"] = language
    generate_kwargs["task"] = task

    outputs = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    segments = []
    chunk_duration = len(audio_array) / sr

    if isinstance(outputs, dict) and "segments" in outputs:
        all_segments = outputs["segments"][0]  # single-item batch
        for seg in all_segments:
            text = processor.decode(seg["tokens"], skip_special_tokens=True).strip()
            if text:
                segments.append(
                    {
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": text,
                    }
                )
    else:
        # Fallback: no structured segments
        token_ids = outputs if not isinstance(outputs, dict) else outputs["sequences"]
        full_text = processor.batch_decode(token_ids, skip_special_tokens=True)[
            0
        ].strip()
        if full_text:
            segments.append(
                {
                    "start": 0.0,
                    "end": chunk_duration,
                    "text": full_text,
                }
            )

    return segments


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

    Steps:
    1. Extract audio from video as 16kHz mono WAV via ffmpeg
    2. Split audio into 2-minute chunks
    3. Load Whisper large-v3 model from Hugging Face
    4. Transcribe each chunk, offsetting timestamps
    5. Combine segments and generate SRT file
    6. Clean up temporary files
    """
    video_path = Path(video_path)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem

    os.makedirs("processing", exist_ok=True)
    os.makedirs("captions", exist_ok=True)

    extracted_audio_path = Path(f"processing/{video_stem}_audio.wav")
    output_srt_path = Path(f"captions/{video_stem}.srt")
    chunk_paths = []

    try:
        # Step 1 ─ Extract audio
        extract_audio_from_video(video_path, extracted_audio_path)

        # Step 2 ─ Split into 2-minute chunks
        logger.info(f"Splitting audio into {CHUNK_DURATION_SEC}s chunks...")
        chunk_paths = split_audio_into_chunks(
            extracted_audio_path, CHUNK_DURATION_SEC, "processing"
        )

        # Step 3 ─ Load model
        logger.info("Loading Whisper large-v3 model (this may take a moment)...")

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

        # Step 4 ─ Transcribe each chunk
        all_chunks = []

        for i, (chunk_path, offset) in enumerate(chunk_paths):
            logger.info(
                f"Transcribing chunk {i + 1}/{len(chunk_paths)} "
                f"(offset {offset:.1f}s)..."
            )

            segments = transcribe_chunk(
                chunk_path, processor, model, device, torch_dtype, language, task
            )

            logger.info(f"  → {len(segments)} segment(s) from chunk {i + 1}")

            for seg in segments:
                all_chunks.append(
                    {
                        "timestamp": (seg["start"] + offset, seg["end"] + offset),
                        "text": seg["text"],
                    }
                )

        # Step 5 ─ Build SRT
        logger.info(f"Generating SRT file from {len(all_chunks)} subtitle entries...")

        if not all_chunks:
            logger.warning("No speech detected in the audio")
            srt_content = ""
        else:
            srt_content = format_srt(all_chunks)

        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT saved to {output_srt_path}")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    finally:
        # Step 6 ─ Clean up
        for chunk_path, _ in chunk_paths:
            if Path(chunk_path).exists():
                Path(chunk_path).unlink()
        if extracted_audio_path.exists():
            logger.info("Cleaning up temporary audio files")
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
