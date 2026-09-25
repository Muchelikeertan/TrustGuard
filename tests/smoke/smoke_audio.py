import os
import sys
import numpy as np
from pathlib import Path

# Ensure ffmpeg binary directory is included in PATH if installed via WinGet
winget_ffmpeg_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
for p in winget_ffmpeg_dir.glob("Gyan.FFmpeg*/**/bin"):
    if (p / "ffmpeg.exe").exists():
        if str(p) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        break


def run_smoke_audio(
    ref_path: str = "sample_data/audio/reference.wav",
    test_path: str = "sample_data/audio/test.wav",
    diff_path: str = "sample_data/audio/different_speaker.wav",
) -> dict:
    """Computes Resemblyzer speaker similarity and Whisper speech-to-text transcription.

    Returns:
        dict: {'score': float or None, 'label': str, 'reason': str}
    """
    p_ref = Path(ref_path)
    p_test = Path(test_path)
    p_diff = Path(diff_path)

    missing = []
    for p in [p_ref, p_test, p_diff]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("[WAITING FOR DATA] Missing required sample audio file(s):")
        for m in missing:
            print(f"  -> Please add: {m}")
        return {
            "score": None,
            "label": "WAITING FOR DATA",
            "reason": f"Missing audio files: {', '.join(missing)}",
        }

    # Imports
    from resemblyzer import VoiceEncoder, preprocess_wav
    import whisper

    print("--- 1. Running Resemblyzer Speaker Similarity ---")
    encoder = VoiceEncoder()

    wav_ref = preprocess_wav(str(p_ref))
    wav_test = preprocess_wav(str(p_test))
    wav_diff = preprocess_wav(str(p_diff))

    emb_ref = encoder.embed_utterance(wav_ref)
    emb_test = encoder.embed_utterance(wav_test)
    emb_diff = encoder.embed_utterance(wav_diff)

    # Cosine similarity
    sim_same = float(np.inner(emb_ref, emb_test) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_test)))
    sim_diff = float(np.inner(emb_ref, emb_diff) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_diff)))

    print(f"  Similarity (reference vs test):            {sim_same:.4f}")
    print(f"  Similarity (reference vs different_speaker): {sim_diff:.4f}")

    print("\n--- 2. Running Whisper Speech-to-Text ---")
    whisper_model = whisper.load_model("base")
    transcription = whisper_model.transcribe(str(p_test))
    transcribed_text = transcription.get("text", "").strip()
    print(f"  Transcribed text: \"{transcribed_text}\"")

    # Evaluation
    # Threshold for Resemblyzer speaker match is typically ~0.70-0.75
    if sim_same > 0.70 and sim_diff < sim_same:
        label = "VERIFIED_SPEAKER"
        score = sim_same
        reason = f"Speaker matched reference (sim={sim_same:.3f}) and differed from imposter (sim={sim_diff:.3f})."
    else:
        label = "SUSPICIOUS_SPEAKER"
        score = sim_same
        reason = f"Voice similarity below threshold or imposter mismatch: sim_same={sim_same:.3f}, sim_diff={sim_diff:.3f}."

    return {
        "score": score,
        "label": label,
        "reason": reason,
    }


if __name__ == "__main__":
    result = run_smoke_audio()
    print("\nResult:", result)
