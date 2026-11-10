from __future__ import annotations

import pytest

from agent.errors import ToolError
from agent.registry import ExecutionContext, ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import dvc_tools
from agent.tools.safe_exec import run_command
from tests import fakes

CTX = ExecutionContext("t")


def reg(settings, runner=None):
    r = ToolRegistry()
    r.register_all(dvc_tools.build_tools(settings, runner=runner or fakes.make_runner()))
    return r


def call(r, tool, **args):
    return r.execute(ToolCall(tool=tool, args=args), CTX)


def test_list_datasets(settings):
    res = call(reg(settings), "dvc_list_datasets")
    assert res.ok and res.data["datasets"][0]["md5"] == fakes.MD5


def test_current_version_reads_commit_and_hash(settings):
    res = call(reg(settings), "dvc_current_version")
    assert res.data["git_commit"] == fakes.COMMIT and res.data["git_dirty"] is False


def test_detect_changes_clean_and_modified(settings):
    assert call(reg(settings), "dvc_detect_changes").data["changed"] is False
    dirty = fakes.make_runner({("dvc", "status"): (0, '{"data/train.csv": "modified"}', "")})
    assert call(reg(settings, dirty), "dvc_detect_changes").data["changed"] is True


def test_compare_versions(settings):
    shows = iter([f"outs:\n- md5: {'a'*32}\n  size: 10\n  path: train.csv\n",
                  f"outs:\n- md5: {'b'*32}\n  size: 25\n  path: train.csv\n"])

    def runner(args, **kw):
        import subprocess
        if tuple(args[:2]) == ("git", "show"):
            return subprocess.CompletedProcess(args, 0, next(shows), "")
        return subprocess.CompletedProcess(args, 0, fakes.COMMIT + "\n", "")

    res = call(reg(settings, runner), "dvc_compare_versions", rev_a="HEAD~1", rev_b="HEAD")
    assert res.data["changed"] is True and res.data["size_delta_bytes"] == 15


def test_find_commit_for_hash(settings):
    res = call(reg(settings), "dvc_find_commit_for_hash", md5=fakes.MD5)
    assert res.data["found"] and res.data["introducing_commit"] == fakes.COMMIT


def test_option_injection_in_revision_is_rejected(settings):
    res = call(reg(settings), "dvc_reproduce_steps", rev="--upload-pack=evil")
    assert res.status == ToolStatus.INVALID


def test_commands_outside_allow_list_are_rejected(tmp_path):
    with pytest.raises(ToolError) as e:
        run_command(["git", "push"], cwd=tmp_path)
    assert e.value.code == "command_not_allowed"


def test_reproduce_steps_contains_pinned_commit(settings):
    shows = f"outs:\n- md5: {'a'*32}\n  size: 10\n  path: train.csv\n"
    runner = fakes.make_runner({("git", "show"): (0, shows, ""), ("git", "rev-parse"): (0, fakes.COMMIT + "\n", "")})
    res = call(reg(settings, runner), "dvc_reproduce_steps", rev="HEAD")
    assert f"git switch --detach {fakes.COMMIT}" in res.data["steps"][0]
