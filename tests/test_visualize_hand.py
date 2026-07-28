import numpy as np
import pytest

from grounded.data import visualize_hand


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        (None, None),
        ("", ""),
        ("   ", "   "),
        ("roll the dough", "Roll the dough"),
        ('  "wipe the table"', '  "Wipe the table"'),
        ("use the SDK API", "Use the SDK API"),
        ("RGB feed from API", "RGB feed from API"),
        ("3D scan with SLAM", "3D scan with SLAM"),
    ],
)
def test_format_caption_for_display(caption, expected):
    assert visualize_hand.format_caption_for_display(caption) == expected


def test_caption_bar_uses_display_format_without_mutating_source(monkeypatch):
    rendered_lines = []

    def capture_text(image, text, *args, **kwargs):
        rendered_lines.append(text)
        return image

    monkeypatch.setattr(visualize_hand.cv2, "putText", capture_text)
    source_caption = "use the SDK API"

    visualize_hand._append_caption_bar(np.zeros((32, 1600, 3), dtype=np.uint8), source_caption)

    assert rendered_lines == ["Use the SDK API"]
    assert source_caption == "use the SDK API"
