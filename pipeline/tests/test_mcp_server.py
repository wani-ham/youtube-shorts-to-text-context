import json
import stat
import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client

from pipeline import mcp_server as app

HEX = "a" * 64


def fake_pipeline(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_pipeline.py"
    script.write_text(
        "import sys, time\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        f"HEX = '{HEX}'\n" + body,
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return [sys.executable, str(script)]


def wait(job, seconds=10):
    deadline = time.time() + seconds
    while job.status in app.ACTIVE and time.time() < deadline:
        time.sleep(0.02)
    assert job.status not in app.ACTIVE
    return job


@pytest.mark.parametrize("code, status", [(0, "complete"), (1, "partial")])
def test_job_reports_finished_record(tmp_path, code, status):
    body = f"print('noise'); print(out + '/' + HEX + '/record.json'); sys.exit({code})\n"
    manager = app.JobManager(tmp_path, fake_pipeline(tmp_path, body))
    job = wait(manager.submit(["--file", "x"], ("x", False)))
    assert job.status == status and job.media_id == f"sha256:{HEX}"


def test_job_reports_pipeline_rejection(tmp_path):
    body = (
        "print('{\"error\": \"too_long\", \"message\": \"Video is 600s.\"}', file=sys.stderr)\n"
        "sys.exit(2)\n"
    )
    job = wait(app.JobManager(tmp_path, fake_pipeline(tmp_path, body)).submit([], ("x", False)))
    assert (job.status, job.error_code, job.error_message) == (
        "failed", "too_long", "Video is 600s.")


def test_crash_does_not_expose_stderr_or_stop_the_worker(tmp_path):
    body = "print('Traceback with /private/path', file=sys.stderr); sys.exit(1)\n"
    manager = app.JobManager(tmp_path, fake_pipeline(tmp_path, body))
    first = wait(manager.submit([], ("a", False)))
    second = wait(manager.submit([], ("b", False)))
    assert first.error_code == "pipeline_crashed" and "private" not in first.error_message
    assert second.status == "failed"


def test_output_path_outside_store_is_not_trusted(tmp_path):
    body = "print('/elsewhere/' + HEX + '/record.json')\n"
    job = wait(app.JobManager(tmp_path, fake_pipeline(tmp_path, body)).submit([], ("x", False)))
    assert job.status == "failed" and job.media_id is None


def test_job_timeout_stops_process(tmp_path):
    manager = app.JobManager(tmp_path, fake_pipeline(tmp_path, "time.sleep(30)\n"), timeout_s=0.2)
    job = wait(manager.submit([], ("x", False)))
    assert job.error_code == "job_timeout"


def test_same_active_video_is_not_queued_twice(tmp_path):
    manager = app.JobManager(tmp_path, fake_pipeline(tmp_path, "time.sleep(1)\n"), timeout_s=0.3)
    first = manager.submit(["--url", "u"], ("u", False))
    second = manager.submit(["--url", "u"], ("u", False))
    assert first is second
    wait(first)


def test_local_files_are_disabled_by_default(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.delenv("VIDEO_TEXT_ALLOWED_DIRS", raising=False)
    with pytest.raises(ValueError, match="disabled"):
        app.resolve_allowed_file(str(video))


def test_allowed_folder_blocks_escapes_and_symlinks(tmp_path, monkeypatch):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir(), private.mkdir()
    (shared / "ok.mp4").write_bytes(b"x")
    (private / "secret.mp4").write_bytes(b"x")
    (shared / "link.mp4").symlink_to(private / "secret.mp4")
    monkeypatch.setenv("VIDEO_TEXT_ALLOWED_DIRS", str(shared))
    assert app.resolve_allowed_file(str(shared / "ok.mp4")) == (shared / "ok.mp4").resolve()
    for attempt in (shared / ".." / "private" / "secret.mp4", shared / "link.mp4"):
        with pytest.raises(ValueError, match="outside"):
            app.resolve_allowed_file(str(attempt))


@pytest.mark.parametrize("bad", ["../../etc/passwd", "sha256:xyz", "A" * 64, HEX + "/../x", ""])
def test_media_id_cannot_traverse_paths(bad):
    with pytest.raises(ValueError):
        app.media_hex(bad)


def write_record(store: Path) -> None:
    directory = store / HEX
    directory.mkdir(parents=True)
    (directory / "record.md").write_text("# Video text record\n", encoding="utf-8")
    (directory / "record.json").write_text(json.dumps({
        "media_id": f"sha256:{HEX}", "llm_document": "duplicate",
        "source": {"url": "https://youtu.be/x", "platform": "youtube", "uploader": "u"},
        "media": {"duration_s": 12.0}, "transcript": {"language": "en"},
        "visual_context": {"footage_type": "mixed"}, "pipeline": {"status": "complete"},
    }), encoding="utf-8")


def test_tools_over_mcp_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_TEXT_STORE", str(tmp_path))
    monkeypatch.delenv("VIDEO_TEXT_ALLOWED_DIRS", raising=False)
    monkeypatch.setattr(app, "_manager", None)
    write_record(tmp_path)

    async def scenario():
        async with Client(app.server) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == {"start_video_analysis", "get_analysis_status", "get_video_text",
                             "list_analyzed_videos", "check_setup"}

            markdown = await client.call_tool("get_video_text", {"media_id": f"sha256:{HEX}"})
            assert "# Video text record" in markdown.content[0].text

            raw = await client.call_tool("get_video_text", {"media_id": HEX, "format": "json"})
            assert "llm_document" not in json.loads(raw.content[0].text)

            listed = await client.call_tool("list_analyzed_videos", {})
            assert f"sha256:{HEX}" in listed.content[0].text

            for name, arguments in (
                ("start_video_analysis", {"url": "https://example.com/video"}),
                ("start_video_analysis", {"file_path": "/etc/passwd"}),
                ("start_video_analysis", {}),
                ("get_analysis_status", {"job_id": "missing"}),
            ):
                assert (await client.call_tool(name, arguments)).is_error

    anyio.run(scenario)
