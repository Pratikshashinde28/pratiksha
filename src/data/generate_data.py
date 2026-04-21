from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


POST_TYPES = ["image", "carousel", "reel", "video"]


def _bounded_normal(center: float, spread: float, low: float, high: float) -> float:
    value = random.gauss(center, spread)
    return float(max(low, min(high, value)))


def build_dataset(rows: int = 1400, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    post_boost = {
        "image": {"likes": 0.95, "comments": 0.9},
        "carousel": {"likes": 1.15, "comments": 1.2},
        "reel": {"likes": 1.35, "comments": 1.05},
        "video": {"likes": 1.0, "comments": 1.1},
    }

    records: list[dict[str, object]] = []
    for _ in range(rows):
        followers = int(np.random.randint(1500, 500000))
        post_type = random.choice(POST_TYPES)
        caption_length = int(np.random.randint(20, 350))
        hashtags_count = int(np.random.randint(0, 16))
        hour_posted = int(np.random.randint(6, 24))
        is_weekend = bool(np.random.randint(0, 2))
        sponsorship = bool(np.random.randint(0, 2))

        type_factor = post_boost[post_type]
        timing_boost = 1.12 if hour_posted in range(18, 23) else 0.97
        weekend_boost = 1.08 if is_weekend else 1.0
        hashtag_boost = 1 + min(hashtags_count, 10) * 0.018
        caption_boost = 1 + min(caption_length, 220) * 0.0009
        sponsor_penalty = 0.9 if sponsorship else 1.0

        follower_signal = followers ** 0.62
        likes_noise = _bounded_normal(1.0, 0.12, 0.7, 1.35)
        comments_noise = _bounded_normal(1.0, 0.15, 0.65, 1.4)

        likes = (
            follower_signal
            * 13
            * type_factor["likes"]
            * timing_boost
            * weekend_boost
            * hashtag_boost
            * caption_boost
            * sponsor_penalty
            * likes_noise
        )
        comments = (
            follower_signal
            * 0.9
            * type_factor["comments"]
            * (1.03 if caption_length > 80 else 0.95)
            * (1 + min(hashtags_count, 8) * 0.01)
            * (1.04 if hour_posted in range(19, 22) else 0.98)
            * sponsor_penalty
            * comments_noise
        )

        records.append(
            {
                "followers": followers,
                "post_type": post_type,
                "caption_length": caption_length,
                "hashtags_count": hashtags_count,
                "hour_posted": hour_posted,
                "is_weekend": int(is_weekend),
                "sponsorship": int(sponsorship),
                "likes": int(max(50, likes)),
                "comments": int(max(5, comments)),
            }
        )

    return pd.DataFrame(records)


def save_dataset(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    target = Path("artifacts") / "instagram_posts.csv"
    save_dataset(target)
    print(f"Saved dataset to {target}")
