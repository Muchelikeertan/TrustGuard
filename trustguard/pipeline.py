"""TrustGuard End-to-End Multimodal Analysis Pipeline.

Executes real biometric and NLP models conditionally and lazily:
1. DeepFace (Face verification with enforce_detection=True; per-photo face checks)
2. Resemblyzer (Speaker acoustic verification only if both audio clips provided)
3. Whisper (Speech-to-text transcription only if test audio clip provided)
4. BART Large MNLI (Zero-shot text classification only if chat or transcript provided)
5. Multimodal Consistency Engine (Signal fusion and decision-support risk assessment)
"""

import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 1. Windows UTF-8 console output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Legacy Keras configuration for TensorFlow 2.16+ / DeepFace compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Ensure ffmpeg is discovered on Windows
if not shutil.which("ffmpeg"):
    winget_ffmpeg_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for p in winget_ffmpeg_dir.glob("Gyan.FFmpeg*/**/bin"):
        if (p / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            break

import numpy as np

from trustguard.consistency_engine import (
    MultimodalConsistencyResult,
    evaluate_consistency,
    load_thresholds,
)


# Global singleton cache for models
_CACHED_MODELS = {
    "voice_encoder": None,
    "whisper": None,
}


def get_voice_encoder():
    """Returns cached Resemblyzer voice encoder (loaded lazily on demand)."""
    if _CACHED_MODELS["voice_encoder"] is None:
        from resemblyzer import VoiceEncoder
        _CACHED_MODELS["voice_encoder"] = VoiceEncoder()
    return _CACHED_MODELS["voice_encoder"]


def get_whisper_model(model_name: str = "base"):
    """Returns cached OpenAI Whisper model (loaded lazily on demand)."""
    if _CACHED_MODELS["whisper"] is None:
        import whisper
        _CACHED_MODELS["whisper"] = whisper.load_model(model_name)
    return _CACHED_MODELS["whisper"]


def run_multimodal_analysis(
    claimed_identity: str,
    ref_photo_path: Optional[Union[str, Path]] = None,
    test_photo_path: Optional[Union[str, Path]] = None,
    ref_audio_path: Optional[Union[str, Path]] = None,
    test_audio_path: Optional[Union[str, Path]] = None,
    chat_text: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    thresholds_path: Optional[Union[str, Path]] = None,
) -> Tuple[MultimodalConsistencyResult, List[str]]:
    """Runs real multimodal pipeline end-to-end with lazy loading and input-conditional skipping.

    Args:
        claimed_identity: Name or persona being claimed.
        ref_photo_path: Path to reference image.
        test_photo_path: Path to test image.
        ref_audio_path: Path to reference audio clip.
        test_audio_path: Path to test audio clip.
        chat_text: Raw incoming text message.
        progress_callback: Optional callback(step_name, progress_ratio).
        thresholds_path: Custom YAML config path.

    Returns:
        Tuple[MultimodalConsistencyResult, List[str]]: Assessment result and list of model warning notices.
    """
    start_time = time.perf_counter()
    warnings: List[str] = []
    thresholds = load_thresholds(thresholds_path)

    p_ref_img = Path(ref_photo_path) if ref_photo_path else None
    p_test_img = Path(test_photo_path) if test_photo_path else None
    p_ref_aud = Path(ref_audio_path) if ref_audio_path else None
    p_test_aud = Path(test_audio_path) if test_audio_path else None

    has_both_photos = bool(p_ref_img and p_test_img and p_ref_img.exists() and p_test_img.exists())
    has_both_audios = bool(p_ref_aud and p_test_aud and p_ref_aud.exists() and p_test_aud.exists())
    has_test_audio = bool(p_test_aud and p_test_aud.exists())
    has_chat = bool(chat_text and chat_text.strip())

    image_result: Optional[Dict[str, Any]] = None
    audio_result: Optional[Dict[str, Any]] = None
    transcribed_text: Optional[str] = None

    # -----------------------------------------------------------------------
    # Step 1: Image - DeepFace Facial Biometrics (Only if BOTH photos provided)
    # -----------------------------------------------------------------------
    if has_both_photos:
        if progress_callback:
            progress_callback("Step 1/5: Image — Facial biometric comparison (DeepFace)", 0.20)

        try:
            from deepface import DeepFace

            # Phase 1: Separate face presence checks with enforce_detection=True
            ref_has_face = True
            test_has_face = True

            try:
                DeepFace.extract_faces(
                    img_path=str(p_ref_img),
                    detector_backend="opencv",
                    enforce_detection=True,
                )
            except ValueError:
                ref_has_face = False

            try:
                DeepFace.extract_faces(
                    img_path=str(p_test_img),
                    detector_backend="opencv",
                    enforce_detection=True,
                )
            except ValueError:
                test_has_face = False

            # If no face in one or both photos, return CAN'T TELL without distance
            if not ref_has_face and not test_has_face:
                image_result = {
                    "status": "CAN'T TELL",
                    "distance": None,
                    "reason": "No clear face was found in either the reference photo or test photo. Please upload a clear, front-facing photo.",
                }
            elif not ref_has_face:
                image_result = {
                    "status": "CAN'T TELL",
                    "distance": None,
                    "reason": "No clear face was found in the reference photo. Please upload a clear, front-facing photo.",
                }
            elif not test_has_face:
                image_result = {
                    "status": "CAN'T TELL",
                    "distance": None,
                    "reason": "No clear face was found in the test photo. Please upload a clear, front-facing photo.",
                }
            else:
                # Both photos contain clear faces: perform verification with enforce_detection=True
                res_face = DeepFace.verify(
                    img1_path=str(p_ref_img),
                    img2_path=str(p_test_img),
                    enforce_detection=True,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                )
                dist = float(res_face.get("distance", 1.0))
                thresh = float(res_face.get("threshold", thresholds["image"]["pass_max_distance"]))
                verified = bool(res_face.get("verified", False))

                pass_max = thresholds["image"]["pass_max_distance"]
                fail_min = thresholds["image"]["fail_min_distance"]

                if dist <= pass_max:
                    img_stat = "PASS"
                    reason_img = f"Facial geometry is consistent with the reference photo (distance: {dist:.3f})."
                elif dist >= fail_min:
                    img_stat = "FAIL"
                    reason_img = f"Facial geometry diverges from the reference photo (distance: {dist:.3f})."
                else:
                    img_stat = "WARNING"
                    reason_img = f"Borderline facial consistency with reference photo (distance: {dist:.3f})."

                image_result = {
                    "status": img_stat,
                    "distance": dist,
                    "threshold": thresh,
                    "verified": verified,
                    "reason": reason_img,
                }

        except ValueError as val_err:
            # Face detection failure caught directly from verify
            err_msg = str(val_err)
            if "Face could not be detected" in err_msg:
                image_result = {
                    "status": "CAN'T TELL",
                    "distance": None,
                    "reason": "No clear face was found in one or both photos. Please upload a clear, front-facing photo.",
                }
            else:
                print("\n[DeepFace Real ValueError]:")
                traceback.print_exc()
                warnings.append(f"DeepFace error: {str(val_err)}")
                image_result = {
                    "status": "CAN'T TELL",
                    "distance": None,
                    "reason": f"Image processing error: {str(val_err)}",
                }
        except Exception as exc:
            # Real errors (corrupted file, model failure) logged with traceback
            print("\n[DeepFace Real Exception]:")
            traceback.print_exc()
            warnings.append(f"DeepFace processing error: {str(exc)}")
            image_result = {
                "status": "CAN'T TELL",
                "distance": None,
                "reason": f"Image processing error: {str(exc)}",
            }
    else:
        if progress_callback:
            progress_callback("Step 1/5: Image — Skipped (no input)", 0.20)
        image_result = {
            "status": "CAN'T TELL",
            "distance": None,
            "reason": "Both photos are needed.",
        }

    # -----------------------------------------------------------------------
    # Step 2: Voice - Resemblyzer Acoustics (Only if BOTH voice clips provided)
    # -----------------------------------------------------------------------
    similarity: Optional[float] = None
    v_stat = "CAN'T TELL"
    reason_v = "Both voice clips are needed."

    if has_both_audios:
        if progress_callback:
            progress_callback("Step 2/5: Voice — Acoustic speaker comparison (Resemblyzer)", 0.40)

        try:
            from resemblyzer import preprocess_wav

            encoder = get_voice_encoder()
            wav_ref = preprocess_wav(str(p_ref_aud))
            wav_test = preprocess_wav(str(p_test_aud))

            emb_ref = encoder.embed_utterance(wav_ref)
            emb_test = encoder.embed_utterance(wav_test)

            similarity = float(np.inner(emb_ref, emb_test) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_test)))

            pass_sim = thresholds["voice"]["pass_min_similarity"]
            fail_sim = thresholds["voice"]["fail_max_similarity"]

            if similarity >= pass_sim:
                v_stat = "PASS"
                reason_v = f"Voice acoustics are consistent with the reference speaker (similarity: {similarity:.3f})."
            elif similarity < fail_sim:
                v_stat = "FAIL"
                reason_v = f"Voice acoustics diverge from the reference speaker (similarity: {similarity:.3f})."
            else:
                v_stat = "WARNING"
                reason_v = f"Borderline voice acoustic similarity with reference speaker (similarity: {similarity:.3f})."

        except Exception as exc:
            print("\n[Resemblyzer Real Exception]:")
            traceback.print_exc()
            warnings.append(f"Resemblyzer error: {str(exc)}")
            v_stat = "CAN'T TELL"
            reason_v = f"Voice processing error: {str(exc)}"
    else:
        if progress_callback:
            progress_callback("Step 2/5: Voice — Skipped (no input)", 0.40)

    # -----------------------------------------------------------------------
    # Step 3: Transcribing - Whisper (Only if TEST voice clip provided)
    # -----------------------------------------------------------------------
    if has_test_audio:
        if progress_callback:
            progress_callback("Step 3/5: Transcribing — Speech-to-text extraction (Whisper)", 0.60)

        try:
            whisper_mod = get_whisper_model("base")
            trans_res = whisper_mod.transcribe(str(p_test_aud))
            transcribed_text = trans_res.get("text", "").strip()
            if not transcribed_text:
                transcribed_text = None
        except Exception as exc:
            print("\n[Whisper Real Exception]:")
            traceback.print_exc()
            warnings.append(f"Whisper error: {str(exc)}")
            transcribed_text = None
    else:
        if progress_callback:
            progress_callback("Step 3/5: Transcribing — Skipped (no input)", 0.60)

    audio_result = {
        "status": v_stat,
        "similarity": similarity,
        "transcript": transcribed_text,
        "reason": reason_v,
    }

    # -----------------------------------------------------------------------
    # Step 4: Chat - Threat & Coercion Analysis (Only if chat or transcript present)
    # -----------------------------------------------------------------------
    if has_chat or transcribed_text:
        if progress_callback:
            progress_callback("Step 4/5: Chat — Threat & coercion analysis (BART Zero-Shot)", 0.80)
    else:
        if progress_callback:
            progress_callback("Step 4/5: Chat — Skipped (no input)", 0.80)

    # -----------------------------------------------------------------------
    # Step 5: Consistency Engine - Cross-modal evaluation
    # -----------------------------------------------------------------------
    if progress_callback:
        progress_callback("Step 5/5: Consistency Engine — Multimodal contradiction evaluation", 0.95)

    consistency_result = evaluate_consistency(
        claimed_identity=claimed_identity,
        image_result=image_result,
        audio_result=audio_result,
        chat_text=chat_text if has_chat else None,
        thresholds_path=thresholds_path,
    )

    elapsed_time = round(time.perf_counter() - start_time, 2)
    consistency_result.details["elapsed_time_seconds"] = elapsed_time

    if progress_callback:
        progress_callback(f"Complete — Analysis finished in {elapsed_time}s", 1.0)

    return consistency_result, warnings
