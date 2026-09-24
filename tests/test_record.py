import pytest
from pathlib import Path
from record import get_unrecorded_prompts, append_metadata

def test_resume_logic(tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    prompts = [
        ("id1", "Hello world."),
        ("id2", "Second line."),
        ("id3", "Third line.")
    ]
    
    # Empty metadata -> all unrecorded
    unrecorded = get_unrecorded_prompts(prompts, metadata_csv)
    assert len(unrecorded) == 3
    assert unrecorded[0][0] == "id1"
    
    # Write one metadata line
    append_metadata(metadata_csv, "id1", "Hello world.")
    
    # Resume logic should skip id1
    unrecorded = get_unrecorded_prompts(prompts, metadata_csv)
    assert len(unrecorded) == 2
    assert unrecorded[0][0] == "id2"
    assert unrecorded[1][0] == "id3"
    
    # Write another line
    append_metadata(metadata_csv, "id2", "Second line.")
    
    # Resume logic should skip id1 and id2
    unrecorded = get_unrecorded_prompts(prompts, metadata_csv)
    assert len(unrecorded) == 1
    assert unrecorded[0][0] == "id3"

def test_metadata_writing(tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    append_metadata(metadata_csv, "id123", "Some text.")
    append_metadata(metadata_csv, "id456", "More text here.")
    
    content = metadata_csv.read_text(encoding="utf-8")
    assert content == "id123|Some text.\nid456|More text here.\n"

