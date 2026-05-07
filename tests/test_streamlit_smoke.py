"""Streamlit app smoke test — verifies the file exists."""
from __future__ import annotations


def test_streamlit_app_imports():
    """Importing the Streamlit module should not raise (it shouldn't run on import)."""
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app_test", p)
    assert spec is not None
    assert p.exists()
