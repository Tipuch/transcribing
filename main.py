#!/usr/bin/env python3
"""CLI tool to transcribe an audio file to SRT format using NeMo ASR."""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_audio_into_chunks(audio_path, chunk_duration_sec=300):
    """
    Split an audio file into chunks of the specified duration using soundfile.

    Returns a list of Path objects pointing to the chunk files.
    """
    import soundfile as sf
    import numpy as np

    audio_path = Path(audio_path)
    chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(audio_path))
    sample_rate = info.samplerate
    total_frames = info.frames
    chunk_frames = int(chunk_duration_sec * sample_rate)

    chunk_files = []
    offset = 0
    chunk_idx = 0

    while offset < total_frames:
        frames_to_read = min(chunk_frames, total_frames - offset)
        data, sr = sf.read(str(audio_path), start=offset, frames=frames_to_read, dtype="int16")

        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.wav"
        sf.write(str(chunk_path), data, sr, subtype="PCM_16")
        chunk_files.append(chunk_path)

        offset += frames_to_read
        chunk_idx += 1

    logger.info(f"Split audio into {len(chunk_files)} chunks of ~{chunk_duration_sec}s each")
    return chunk_files


def format_srt(segments):
    """Format a list of segment dicts (with 'start', 'end', 'segment') into SRT content."""
    lines = []

    for i, seg in enumerate(segments, start=1):
        start_ts = _seconds_to_srt_timestamp(seg["start"])
        end_ts = _seconds_to_srt_timestamp(seg["end"])
        text = seg.get("segment", "").strip()
        lines.append(str(i))
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


def transcribe_audio(audio_path):
    """
    Transcribe an audio file to SRT format using NeMo ASR.

    Steps:
    1. Ensure audio is 16kHz mono WAV (convert if needed)
    2. Split audio into manageable chunks (for 8GB VRAM)
    3. Load NeMo model with long-form audio configuration
    4. Transcribe each chunk with segment timestamps
    5. Merge results and generate SRT file
    6. Clean up temporary audio files
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    audio_stem = audio_path.stem

    # Create directories if needed
    os.makedirs("processing", exist_ok=True)
    os.makedirs("captions", exist_ok=True)

    # We may need a resampled/converted copy if the input isn't 16kHz mono WAV
    converted_audio_path = Path(f"processing/{audio_stem}_converted.wav")
    output_srt_path = Path(f"captions/{audio_stem}.srt")
    chunk_dir = None
    needs_conversion = False

    try:
        # Check if we need to convert the audio to 16kHz mono PCM WAV
        import soundfile as sf

        working_audio_path = audio_path

        try:
            info = sf.info(str(audio_path))
            if info.samplerate != 16000 or info.channels != 1:
                needs_conversion = True
                logger.info(
                    f"Audio is {info.samplerate}Hz, {info.channels}ch — converting to 16kHz mono WAV"
                )
        except Exception:
            # If soundfile can't read it directly, try converting with librosa
            needs_conversion = True
            logger.info("Audio format not directly readable by soundfile — converting to 16kHz mono WAV")

        if needs_conversion:
            _convert_audio_to_16khz_mono(audio_path, converted_audio_path)
            working_audio_path = converted_audio_path

        # Step 2: Split audio into chunks (5 minutes each for 8GB VRAM)
        logger.info("Splitting audio into chunks for memory-efficient processing...")
        chunk_files = split_audio_into_chunks(working_audio_path, chunk_duration_sec=300)

        if not chunk_files:
            logger.error("Failed to create audio chunks")
            sys.exit(1)

        chunk_dir = chunk_files[0].parent

        # Step 3: Load NeMo model
        logger.info("Loading NeMo ASR model (this may take a moment)...")

        # Disable strict protobuf version checking to work around version mismatch
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        # Enable memory-efficient CUDA allocator to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        import nemo.collections.asr as nemo_asr

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )

        # Configure for long-form audio with memory optimization (8GB VRAM target: ~6-7GB usage)
        asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )

        # Enable auto-chunking of subsampling for processing long audio files piece-by-piece
        asr_model.change_subsampling_conv_chunking_factor(1)

        # Disable CUDA graphs to avoid CUDA 12.8 API incompatibility (GitHub issue #15340)
        # Must configure BEFORE transcription since transcribe() reinitializes the decoder
        from omegaconf import open_dict
        with open_dict(asr_model.cfg.decoding):
            asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        logger.info("Configured decoder to disable CUDA graphs")

        # Step 4: Transcribe each chunk with timestamps
        logger.info(f"Transcribing {len(chunk_files)} audio chunks (this may take a while)...")
        all_segments = []
        cumulative_time_offset = 0.0

        for i, chunk_path in enumerate(chunk_files):
            logger.info(f"Processing chunk {i + 1}/{len(chunk_files)}: {chunk_path.name}")

            output = asr_model.transcribe([str(chunk_path)], timestamps=True, batch_size=1)
            chunk_segments = output[0].timestamp["segment"]

            # Adjust timestamps to account for chunk position in original audio
            for segment in chunk_segments:
                segment["start"] += cumulative_time_offset
                segment["end"] += cumulative_time_offset
                all_segments.append(segment)

            # Update offset for next chunk (5 minutes = 300 seconds)
            cumulative_time_offset += 300.0

        # Step 5: Generate SRT
        logger.info(f"Generating SRT file from {len(all_segments)} segments...")
        srt_content = format_srt(all_segments)

        # Write to file
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT saved to {output_srt_path}")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    finally:
        # Step 6: Clean up temporary files
        if chunk_dir and chunk_dir.exists():
            logger.info("Cleaning up audio chunks")
            shutil.rmtree(chunk_dir)

        if converted_audio_path.exists():
            logger.info("Cleaning up converted audio file")
            converted_audio_path.unlink()

    return str(output_srt_path)


def _convert_audio_to_16khz_mono(input_path, output_path):
    """Convert an audio file to 16kHz mono WAV using librosa and soundfile."""
    import numpy as np

    try:
        import librosa

        logger.info(f"Loading audio with librosa: {input_path}")
        # librosa.load automatically resamples to the target sr and converts to mono
        audio_data, sr = librosa.load(str(input_path), sr=16000, mono=True)

        import soundfile as sf

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, 16000, subtype="PCM_16")
        logger.info(f"Converted audio saved to {output_path}")

    except ImportError:
        logger.error(
            "librosa is required for audio format conversion. "
            "Install it with: pip install librosa"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file to SRT format using NeMo ASR.",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the audio file to transcribe.",
    )
    args = parser.parse_args()

    result = transcribe_audio(args.file_path)
    if result:
        print(f"Transcription complete: {result}")


if __name__ == "__main__":
    main()
