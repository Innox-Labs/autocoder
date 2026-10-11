from autocoder.pipeline.sections import section_for_offset, segment

NOTE = """CHIEF COMPLAINT
Follow-up of diabetes.

PAST MEDICAL HISTORY
Type 2 diabetes mellitus.

Assessment and Plan:
Continue metformin.
"""


def test_segment_covers_whole_document():
    sections = segment(NOTE)
    assert sections[0].start == 0
    assert sections[-1].end == len(NOTE)
    for a, b in zip(sections, sections[1:]):
        assert a.end == b.start


def test_headers_detected():
    sections = segment(NOTE)
    headers = [s.raw_header for s in sections if s.raw_header]
    assert "CHIEF COMPLAINT" in headers
    assert "PAST MEDICAL HISTORY" in headers
    assert "Assessment and Plan" in headers


def test_section_for_offset():
    sections = segment(NOTE)
    offset = NOTE.index("metformin")
    section = section_for_offset(sections, offset)
    assert section is not None
    assert section.raw_header == "Assessment and Plan"
