import pytest

from src.data.preprocess import clean_text


def test_clean_text_removes_urls() -> None:
    result = clean_text("Check this out http://example.com now")
    assert "http" not in result
    assert "example" not in result


def test_clean_text_removes_mentions() -> None:
    result = clean_text("hello @worlduser how are you")
    assert "@" not in result


def test_clean_text_handles_empty_string() -> None:
    assert clean_text("") == ""


def test_clean_text_lowercases() -> None:
    result = clean_text("HELLO WORLD")
    assert result == result.lower()
