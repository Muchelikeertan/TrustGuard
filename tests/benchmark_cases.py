"""Benchmarks execution times and results for cases (a), (b), (c), (d)."""

import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trustguard.pipeline import run_multimodal_analysis

SAMPLE_IMAGES = root_dir / "sample_data" / "images"
SAMPLE_AUDIO = root_dir / "sample_data" / "audio"
NO_FACE_IMG = SAMPLE_IMAGES / "no_face.jpg"


def benchmark_all():
    print("=" * 80)
    print("RUNNING BENCHMARKS FOR CASES (a), (b), (c), (d)")
    print("=" * 80)

    # Case (a): No-face photo + normal photo
    print("\n>>> Case (a): No-face photo + normal photo")
    t0 = time.perf_counter()
    res_a, warn_a = run_multimodal_analysis(
        claimed_identity="TestUser",
        ref_photo_path=NO_FACE_IMG,
        test_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
    )
    dt_a = time.perf_counter() - t0
    print(f"Time: {dt_a:.2f}s")
    print(f"Risk: {res_a.risk_level}")
    print(f"Image status: {res_a.modality_statuses['image']}")
    print(f"Image distance: {res_a.details['image']['distance']}")
    print(f"Image reason: {res_a.details['image']['source_reason']}")

    # Case (b): Only a chat message
    print("\n>>> Case (b): Only a chat message")
    t0 = time.perf_counter()
    res_b, warn_b = run_multimodal_analysis(
        claimed_identity="TestUser",
        chat_text="Urgent! Send Rs 25,000 right now to this UPI, don't call me or verify!",
    )
    dt_b = time.perf_counter() - t0
    print(f"Time: {dt_b:.2f}s")
    print(f"Risk: {res_b.risk_level}")
    print(f"Chat status: {res_b.modality_statuses['chat']}")
    print(f"Chat reason: {res_b.details.get('chat_signal', {}).get('top_scam_label', 'N/A')}")
    print(f"Image status: {res_b.modality_statuses['image']}")
    print(f"Voice status: {res_b.modality_statuses['voice']}")

    # Case (c): Only photos
    print("\n>>> Case (c): Only photos (same person)")
    t0 = time.perf_counter()
    res_c, warn_c = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=SAMPLE_IMAGES / "same_person_b.jpg",
    )
    dt_c = time.perf_counter() - t0
    print(f"Time: {dt_c:.2f}s")
    print(f"Risk: {res_c.risk_level}")
    print(f"Image status: {res_c.modality_statuses['image']}")
    print(f"Image distance: {res_c.details['image']['distance']}")
    print(f"Image reason: {res_c.details['image']['source_reason']}")
    print(f"Voice status: {res_c.modality_statuses['voice']}")

    # Case (d): Everything
    print("\n>>> Case (d): Everything (Photos + Audio + Chat)")
    t0 = time.perf_counter()
    res_d, warn_d = run_multimodal_analysis(
        claimed_identity="Alex",
        ref_photo_path=SAMPLE_IMAGES / "same_person_a.jpg",
        test_photo_path=SAMPLE_IMAGES / "same_person_b.jpg",
        ref_audio_path=SAMPLE_AUDIO / "reference.wav",
        test_audio_path=SAMPLE_AUDIO / "test.wav",
        chat_text="Hi Mom, just landed safely at the airport. Taking an Uber home now.",
    )
    dt_d = time.perf_counter() - t0
    print(f"Time: {dt_d:.2f}s")
    print(f"Risk: {res_d.risk_level}")
    print(f"Statuses: {res_d.modality_statuses}")
    print(f"Image distance: {res_d.details['image']['distance']}")
    print(f"Voice similarity: {res_d.details['voice']['similarity']}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print(f"Case (a) No-face photo:   {dt_a:.2f}s -> {res_a.risk_level} (Image: {res_a.modality_statuses['image']})")
    print(f"Case (b) Chat-only:        {dt_b:.2f}s -> {res_b.risk_level} (Chat: {res_b.modality_statuses['chat']})")
    print(f"Case (c) Photo-only:       {dt_c:.2f}s -> {res_c.risk_level} (Image: {res_c.modality_statuses['image']})")
    print(f"Case (d) Everything:       {dt_d:.2f}s -> {res_d.risk_level} (All: {res_d.modality_statuses})")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_all()
