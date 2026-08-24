#!/usr/bin/env python3
"""Two-hour GitHub monitor for the Windows/ROCm/gfx1201 FreeToken port.

The script is intentionally dependency-free so it can run in GitHub Actions with
only the repository GITHUB_TOKEN. It stores its baseline in one edited PR
comment and posts a new PR comment only for meaningful changes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
STATE_MARKER = "<!-- freetoken-rocm-watch-state-v1 -->"
STATE_END = "<!-- /freetoken-rocm-watch-state-v1 -->"
STATE_VERSION = 1

UPSTREAM_REPO = "FlashML-org/FreeToken"
TRACKED_PRS = (27, 112, 118, 125, 129, 131, 132, 133, 134, 135, 136, 137)
P0_PRS = (132, 133, 134, 135, 136, 137)
TRACKED_ISSUES = (79, 82, 110, 111, 120, 122, 123, 124)
P0_ISSUES = (79, 82, 120, 122)

BRANCHES = {
    "FlashML-org/FreeToken:main": ("FlashML-org/FreeToken", "main"),
    "PialGhosh2233/FreeToken-rocm-gfx1200:gfx1200-moe-gguf": (
        "PialGhosh2233/FreeToken-rocm-gfx1200",
        "gfx1200-moe-gguf",
    ),
    "Maxritz/FreeToken-rocm-test:main": ("Maxritz/FreeToken-rocm-test", "main"),
}

RELEASE_REPOS = (
    "FlashML-org/FreeToken",
    "ROCm/TheRock",
    "ROCm/ROCm",
    "triton-lang/triton-windows",
    "apache/tvm-ffi",
    "ggml-org/llama.cpp",
)

DEPENDENCY_SEARCHES = {
    "ROCm/TheRock": (
        "gfx1201",
        "RDNA4 Windows",
        "hipErrorInvalidKernelFile",
        "PyTorch Windows ROCm",
        "hipHostGetDevicePointer",
    ),
    "triton-lang/triton-windows": (
        "ROCm",
        "HIP AMD",
        "launch_pdl",
        "clang-cl",
        "gfx1201",
    ),
    "apache/tvm-ffi": (
        "Windows HIP",
        "ROCm",
        "offload-arch",
        "clang-cl",
        "gfx1201",
    ),
    "pytorch/pytorch": (
        "gfx1201",
        "Windows ROCm",
        "hipErrorInvalidKernelFile",
    ),
    "ggml-org/llama.cpp": (
        "gfx1201",
        "Windows HIP",
        "NVFP4",
        "Qwen3.6",
        "MUL_MAT",
        "ROCm invalid argument",
    ),
}

HARD_TERMS = (
    "rocm",
    "gfx1200",
    "gfx1201",
    "rdna3",
    "rdna4",
    "rx 9060",
    "rx 9070",
    "radeon",
    "hiphostregister",
    "hiphostgetdevicepointer",
    "hiperrorinvalidkernelfile",
    "amdhip64",
    "rccl",
    "therock",
)

WINDOWS_ROCM_TERMS = (
    "triton",
    "tvm-ffi",
    "tvm ffi",
    "pinned",
    "mapped",
    "host memory",
    "host pointer",
    "offload",
    "moe",
    "kernel",
    "jit",
    "launch_pdl",
    "clang-cl",
    "qwen3.6",
    "qwen3_5",
    "nvfp4",
    "hybrid_radix",
)

MODEL_STABILITY_TERMS = (
    "hang",
    "stuck",
    "crash",
    "tdr",
    "driver reset",
    "prefill",
    "decode",
    "offload",
    "nvfp4",
    "hybrid_radix",
    "kv cache",
    "kvcache",
    "pinned",
    "pageable",
    "first token",
)

RELEVANT_PATH_PARTS = (
    "python/freetoken/kernel/pinned.py",
    "python/freetoken/kernel/triton/",
    "python/freetoken/kernel/csrc/jit/",
    "python/freetoken/kernel/backend.py",
    "python/freetoken/kernel/fast_index_copy.py",
    "python/freetoken/kernel/csrc/fast_index_copy",
    "python/freetoken/moe/",
    "python/freetoken/models/qwen3_5_moe/",
    "python/freetoken/engine/",
    "python/freetoken/kvcache/",
    "python/freetoken/scheduler/",
    "setup.py",
    "pyproject.toml",
    "docs/install.md",
)

RELEVANT_COMMIT_TERMS = HARD_TERMS + WINDOWS_ROCM_TERMS + (
    "qwen3.6",
    "nvfp4",
    "host bank",
    "pin budget",
    "safe h2d",
    "partial pin",
    "pageable",
)

FAIL_CONCLUSIONS = {
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
    "startup_failure",
    "stale",
}


class ApiError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, message: str):
        super().__init__(f"{method} {url}: HTTP {status}: {message}")
        self.method = method
        self.url = url
        self.status = status
        self.message = message


@dataclass
class GitHubClient:
    token: str

    def request(
        self,
        method: str,
        path: str,
        data: Any | None = None,
        *,
        accept: str = "application/vnd.github+json",
        allow_404: bool = False,
    ) -> Any | None:
        url = path if path.startswith("https://") else f"{API_ROOT}{path}"
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "freetoken-rocm-watch/1.0",
        }
        body = None
        if data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            if allow_404 and exc.code == 404:
                return None
            try:
                detail = json.loads(payload).get("message", payload)
            except json.JSONDecodeError:
                detail = payload
            raise ApiError(method, url, exc.code, str(detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {url}: {exc}") from exc

    def get(self, path: str, **kwargs: Any) -> Any | None:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, data: Any) -> Any | None:
        return self.request("POST", path, data)

    def patch(self, path: str, data: Any) -> Any | None:
        return self.request("PATCH", path, data)


class Collector:
    def __init__(self, client: GitHubClient, old_state: dict[str, Any]):
        self.client = client
        self.old_state = old_state
        self.errors: list[str] = []

    def guarded(self, label: str, fn: Any) -> Any | None:
        try:
            return fn()
        except Exception as exc:
            self.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"warning: {self.errors[-1]}", file=sys.stderr)
            return None

    def collect(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "tracked_prs": {},
            "tracked_issues": {},
            "discovered": {},
            "branches": {},
            "releases": {},
            "last_checked": now_iso(),
            "errors": [],
        }

        for number in TRACKED_PRS:
            value = self.guarded(
                f"{UPSTREAM_REPO} PR #{number}",
                lambda n=number: self.collect_pr(UPSTREAM_REPO, n),
            )
            if value is not None:
                state["tracked_prs"][str(number)] = value

        for number in TRACKED_ISSUES:
            value = self.guarded(
                f"{UPSTREAM_REPO} issue #{number}",
                lambda n=number: self.collect_issue(UPSTREAM_REPO, n),
            )
            if value is not None:
                state["tracked_issues"][str(number)] = value

        upstream_items = self.guarded(
            "upstream discovery",
            self.collect_upstream_discovery,
        )
        if upstream_items is not None:
            state["discovered"].update(upstream_items)

        dependency_items = self.guarded(
            "dependency discovery",
            self.collect_dependency_discovery,
        )
        if dependency_items is not None:
            state["discovered"].update(dependency_items)

        for key, (repo, branch) in BRANCHES.items():
            value = self.guarded(
                f"branch {key}",
                lambda r=repo, b=branch: self.collect_branch(r, b),
            )
            if value is not None:
                state["branches"][key] = value

        for repo in RELEASE_REPOS:
            value = self.guarded(
                f"release {repo}",
                lambda r=repo: self.collect_release(r),
            )
            if value is not None:
                state["releases"][repo] = value

        state["errors"] = sorted(set(self.errors))
        return merge_missing_from_old(state, self.old_state)

    def collect_pr(self, repo: str, number: int) -> dict[str, Any]:
        pr = self.client.get(f"/repos/{repo}/pulls/{number}")
        if not isinstance(pr, dict):
            raise RuntimeError("unexpected PR response")
        head_sha = str(pr.get("head", {}).get("sha") or "")
        checks = self.collect_checks(repo, head_sha) if head_sha else {"state": "unknown"}
        return {
            "number": number,
            "title": clean_text(pr.get("title")),
            "url": pr.get("html_url"),
            "state": pr.get("state"),
            "draft": bool(pr.get("draft")),
            "merged": bool(pr.get("merged")),
            "head_sha": head_sha,
            "base_sha": str(pr.get("base", {}).get("sha") or ""),
            "updated_at": pr.get("updated_at"),
            "comments": int(pr.get("comments") or 0),
            "review_comments": int(pr.get("review_comments") or 0),
            "commits": int(pr.get("commits") or 0),
            "body_hash": digest(pr.get("body")),
            "checks": checks,
        }

    def collect_checks(self, repo: str, sha: str) -> dict[str, Any]:
        combined = None
        runs = None
        try:
            combined = self.client.get(f"/repos/{repo}/commits/{sha}/status")
        except Exception as exc:
            self.errors.append(f"status {repo}@{sha[:8]}: {type(exc).__name__}: {exc}")
        try:
            runs = self.client.get(
                f"/repos/{repo}/commits/{sha}/check-runs?per_page=100",
                accept="application/vnd.github+json",
            )
        except Exception as exc:
            self.errors.append(f"checks {repo}@{sha[:8]}: {type(exc).__name__}: {exc}")

        status_state = combined.get("state") if isinstance(combined, dict) else None
        check_runs = runs.get("check_runs", []) if isinstance(runs, dict) else []
        pending: list[str] = []
        failed: list[str] = []
        passed = 0
        for run in check_runs:
            name = clean_text(run.get("name")) or "unnamed"
            status = run.get("status")
            conclusion = run.get("conclusion")
            if status != "completed":
                pending.append(name)
            elif conclusion in FAIL_CONCLUSIONS:
                failed.append(name)
            elif conclusion in {"success", "neutral", "skipped"}:
                passed += 1

        if failed:
            state = "failure"
        elif pending or status_state == "pending":
            state = "pending"
        elif check_runs and passed == len(check_runs):
            state = "success"
        elif status_state in {"success", "failure", "pending", "error"}:
            state = status_state
        else:
            state = "unknown"

        return {
            "state": state,
            "total": len(check_runs),
            "failed": sorted(failed)[:12],
            "pending": sorted(pending)[:12],
        }

    def collect_issue(self, repo: str, number: int) -> dict[str, Any]:
        issue = self.client.get(f"/repos/{repo}/issues/{number}")
        if not isinstance(issue, dict):
            raise RuntimeError("unexpected issue response")
        return normalize_issue_item(repo, issue)

    def collect_upstream_discovery(self) -> dict[str, Any]:
        items = self.client.get(
            f"/repos/{UPSTREAM_REPO}/issues?state=all&sort=updated&direction=desc&per_page=100"
        )
        out: dict[str, Any] = {}
        if not isinstance(items, list):
            return out
        for item in items:
            number = int(item.get("number") or 0)
            if number in TRACKED_PRS or number in TRACKED_ISSUES:
                continue
            if is_upstream_relevant(item):
                key = f"{UPSTREAM_REPO}#{number}"
                out[key] = normalize_issue_item(UPSTREAM_REPO, item)
        ordered = sorted(
            out.items(),
            key=lambda kv: str(kv[1].get("updated_at") or ""),
            reverse=True,
        )
        return dict(ordered[:60])

    def collect_dependency_discovery(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for repo, terms in DEPENDENCY_SEARCHES.items():
            for term in terms:
                query = f"repo:{repo} {term}"
                encoded = urllib.parse.quote(query)
                payload = self.client.get(
                    f"/search/issues?q={encoded}&sort=updated&order=desc&per_page=20"
                )
                if not isinstance(payload, dict):
                    continue
                for item in payload.get("items", []):
                    if not dependency_item_relevant(repo, item, term):
                        continue
                    number = int(item.get("number") or 0)
                    key = f"{repo}#{number}"
                    out[key] = normalize_issue_item(repo, item)
        ordered = sorted(
            out.items(),
            key=lambda kv: str(kv[1].get("updated_at") or ""),
            reverse=True,
        )
        return dict(ordered[:80])

    def collect_branch(self, repo: str, branch: str) -> dict[str, Any]:
        branch_q = urllib.parse.quote(branch, safe="")
        payload = self.client.get(f"/repos/{repo}/branches/{branch_q}")
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected branch response")
        commit = payload.get("commit", {})
        nested = commit.get("commit", {}) if isinstance(commit, dict) else {}
        return {
            "repo": repo,
            "branch": branch,
            "sha": str(commit.get("sha") or ""),
            "message": first_line(nested.get("message")),
            "url": commit.get("html_url"),
            "date": nested.get("committer", {}).get("date"),
        }

    def collect_release(self, repo: str) -> dict[str, Any]:
        payload = self.client.get(f"/repos/{repo}/releases/latest", allow_404=True)
        if payload is None:
            return {"tag": None, "published_at": None, "url": None, "name": None}
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected release response")
        return {
            "tag": payload.get("tag_name"),
            "published_at": payload.get("published_at"),
            "url": payload.get("html_url"),
            "name": clean_text(payload.get("name")),
            "draft": bool(payload.get("draft")),
            "prerelease": bool(payload.get("prerelease")),
        }


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def first_line(value: Any) -> str:
    text = str(value or "").strip()
    return text.splitlines()[0].strip() if text else ""


def digest(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def normalize_issue_item(repo: str, item: dict[str, Any]) -> dict[str, Any]:
    is_pr = "pull_request" in item
    return {
        "repo": repo,
        "number": int(item.get("number") or 0),
        "kind": "pr" if is_pr else "issue",
        "title": clean_text(item.get("title")),
        "url": item.get("html_url"),
        "state": item.get("state"),
        "updated_at": item.get("updated_at"),
        "comments": int(item.get("comments") or 0),
        "body_hash": digest(item.get("body")),
    }


def is_upstream_relevant(item: dict[str, Any]) -> bool:
    text = f"{item.get('title', '')}\n{item.get('body', '')}".lower()
    if any(term in text for term in HARD_TERMS):
        return True
    if "windows" in text and any(term in text for term in WINDOWS_ROCM_TERMS):
        return True
    if ("qwen3.6" in text or "qwen3_5" in text or "qwen3.5" in text) and any(
        term in text for term in MODEL_STABILITY_TERMS
    ):
        return True
    if "nvfp4" in text and any(
        term in text for term in ("offload", "download", "index", "hang", "cache", "windows")
    ):
        return True
    return False


def dependency_item_relevant(repo: str, item: dict[str, Any], term: str) -> bool:
    text = f"{item.get('title', '')}\n{item.get('body', '')}".lower()
    needle = term.lower().strip('"')
    if needle in text:
        return True
    if repo == "pytorch/pytorch":
        return "gfx1201" in text or ("windows" in text and "rocm" in text)
    if repo == "ROCm/TheRock":
        return any(x in text for x in ("gfx1201", "rdna4", "windows", "invalid kernel file"))
    if repo == "triton-lang/triton-windows":
        return any(x in text for x in ("rocm", "hip", "amd", "gfx1201", "launch_pdl", "clang-cl"))
    if repo == "apache/tvm-ffi":
        return ("windows" in text and any(x in text for x in ("hip", "rocm", "clang"))) or "gfx1201" in text
    return False


def merge_missing_from_old(current: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """Do not interpret transient API failures as tracked-item deletions."""
    if not old:
        return current
    for section in ("tracked_prs", "tracked_issues", "branches", "releases"):
        old_section = old.get(section, {})
        cur_section = current.setdefault(section, {})
        if isinstance(old_section, dict) and isinstance(cur_section, dict):
            for key, value in old_section.items():
                cur_section.setdefault(key, value)
    if not current.get("discovered") and old.get("discovered"):
        current["discovered"] = old["discovered"]
    return current


def parse_state_comment(comments: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], int | None]:
    for comment in comments:
        body = str(comment.get("body") or "")
        if STATE_MARKER not in body or STATE_END not in body:
            continue
        try:
            raw = body.split(STATE_MARKER, 1)[1].split(STATE_END, 1)[0].strip()
            state = json.loads(raw)
            if isinstance(state, dict):
                return state, int(comment.get("id"))
        except Exception:
            continue
    return {}, None


def list_all_comments(client: GitHubClient, repo: str, issue_number: int) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = client.get(
            f"/repos/{repo}/issues/{issue_number}/comments?per_page=100&page={page}"
        )
        if not isinstance(batch, list):
            break
        comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return comments


def state_comment_body(state: dict[str, Any]) -> str:
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        f"{STATE_MARKER}\n{payload}\n{STATE_END}\n\n"
        f"_Technical baseline updated automatically: `{state.get('last_checked')}`. "
        "This comment is edited in place and does not represent a notification._"
    )


def short_sha(value: Any) -> str:
    text = str(value or "")
    return text[:8] if text else "—"


def item_link(item: dict[str, Any], fallback: str) -> str:
    title = clean_text(item.get("title")) or fallback
    url = item.get("url")
    return f"[{title}]({url})" if url else title


def checks_label(checks: Any) -> str:
    if not isinstance(checks, dict):
        return "unknown"
    state = checks.get("state") or "unknown"
    failed = checks.get("failed") or []
    if failed:
        return f"{state} ({', '.join(failed[:3])})"
    return str(state)


def diff_states(
    client: GitHubClient,
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[str]:
    changes: list[str] = []

    old_prs = old.get("tracked_prs", {})
    new_prs = new.get("tracked_prs", {})
    for key in sorted(new_prs, key=lambda x: int(x)):
        cur = new_prs[key]
        prev = old_prs.get(key)
        prefix = f"FreeToken PR #{key}"
        link = item_link(cur, prefix)
        if prev is None:
            changes.append(f"**Новый отслеживаемый PR:** {link}.")
            continue
        if prev.get("head_sha") != cur.get("head_sha"):
            changes.append(
                f"{link}: новый commit `{short_sha(prev.get('head_sha'))}` → "
                f"`{short_sha(cur.get('head_sha'))}`."
            )
        for field, label in (("state", "state"), ("draft", "draft"), ("merged", "merged")):
            if prev.get(field) != cur.get(field):
                changes.append(f"{link}: **{label}** `{prev.get(field)}` → `{cur.get(field)}`.")
        if prev.get("checks") != cur.get("checks"):
            changes.append(
                f"{link}: CI/checks `{checks_label(prev.get('checks'))}` → "
                f"`{checks_label(cur.get('checks'))}`."
            )
        discussion_changed = (
            prev.get("comments") != cur.get("comments")
            or prev.get("review_comments") != cur.get("review_comments")
            or prev.get("body_hash") != cur.get("body_hash")
        )
        if discussion_changed and prev.get("head_sha") == cur.get("head_sha"):
            changes.append(
                f"{link}: обновлены описание/review/discussion "
                f"(comments {prev.get('comments', 0)}→{cur.get('comments', 0)}, "
                f"review {prev.get('review_comments', 0)}→{cur.get('review_comments', 0)})."
            )

    old_issues = old.get("tracked_issues", {})
    new_issues = new.get("tracked_issues", {})
    for key in sorted(new_issues, key=lambda x: int(x)):
        cur = new_issues[key]
        prev = old_issues.get(key)
        link = item_link(cur, f"FreeToken issue #{key}")
        if prev is None:
            changes.append(f"**Новый отслеживаемый issue:** {link}.")
            continue
        fields_changed = []
        if prev.get("state") != cur.get("state"):
            fields_changed.append(f"state {prev.get('state')}→{cur.get('state')}")
        if prev.get("comments") != cur.get("comments"):
            fields_changed.append(f"comments {prev.get('comments', 0)}→{cur.get('comments', 0)}")
        if prev.get("body_hash") != cur.get("body_hash"):
            fields_changed.append("body edited")
        if fields_changed:
            changes.append(f"{link}: " + ", ".join(fields_changed) + ".")

    old_items = old.get("discovered", {})
    new_items = new.get("discovered", {})
    for key, cur in sorted(
        new_items.items(),
        key=lambda kv: str(kv[1].get("updated_at") or ""),
        reverse=True,
    ):
        prev = old_items.get(key)
        link = item_link(cur, key)
        if prev is None:
            changes.append(f"**Новый релевантный {cur.get('kind', 'item')}:** {link}.")
            continue
        if (
            prev.get("state") != cur.get("state")
            or prev.get("comments") != cur.get("comments")
            or prev.get("body_hash") != cur.get("body_hash")
            or prev.get("title") != cur.get("title")
        ):
            changes.append(
                f"{link}: обновлён "
                f"(state {prev.get('state')}→{cur.get('state')}, "
                f"comments {prev.get('comments', 0)}→{cur.get('comments', 0)})."
            )

    old_branches = old.get("branches", {})
    new_branches = new.get("branches", {})
    for key, cur in sorted(new_branches.items()):
        prev = old_branches.get(key)
        if prev is None:
            changes.append(
                f"Начато отслеживание `{key}`: `{short_sha(cur.get('sha'))}` "
                f"— {cur.get('message') or 'без сообщения'}."
            )
            continue
        if prev.get("sha") != cur.get("sha"):
            url = cur.get("url")
            label = f"[`{key}`]({url})" if url else f"`{key}`"
            changes.append(
                f"{label}: `{short_sha(prev.get('sha'))}` → `{short_sha(cur.get('sha'))}` "
                f"— {cur.get('message') or 'новый commit'}."
            )

    old_releases = old.get("releases", {})
    new_releases = new.get("releases", {})
    for repo, cur in sorted(new_releases.items()):
        prev = old_releases.get(repo)
        if prev is None:
            continue
        if prev.get("tag") != cur.get("tag") and cur.get("tag"):
            url = cur.get("url")
            label = f"[{repo} {cur.get('tag')}]({url})" if url else f"{repo} {cur.get('tag')}"
            changes.append(f"**Новый релиз:** {label}.")

    old_main = old_branches.get("FlashML-org/FreeToken:main", {}).get("sha")
    new_main = new_branches.get("FlashML-org/FreeToken:main", {}).get("sha")
    if old_main and new_main and old_main != new_main:
        relevant = relevant_upstream_main_compare(client, old_main, new_main)
        if relevant:
            changes.append(relevant)

    old_errors = set(old.get("errors", []))
    for error in new.get("errors", []):
        if error not in old_errors:
            changes.append(f"⚠️ Новый сбой части мониторинга: `{truncate(error, 350)}`")

    return dedupe(changes)


def relevant_upstream_main_compare(client: GitHubClient, old_sha: str, new_sha: str) -> str | None:
    try:
        payload = client.get(
            f"/repos/{UPSTREAM_REPO}/compare/{urllib.parse.quote(old_sha)}...{urllib.parse.quote(new_sha)}"
        )
    except Exception as exc:
        return f"Upstream `main` изменился `{short_sha(old_sha)}`→`{short_sha(new_sha)}`, compare недоступен: `{truncate(exc, 180)}`."
    if not isinstance(payload, dict):
        return None
    commits = payload.get("commits", [])
    files = payload.get("files", [])
    messages = [first_line(c.get("commit", {}).get("message")) for c in commits]
    relevant_messages = [
        m for m in messages if any(term in m.lower() for term in RELEVANT_COMMIT_TERMS)
    ]
    relevant_files = [
        str(f.get("filename") or "")
        for f in files
        if any(part in str(f.get("filename") or "") for part in RELEVANT_PATH_PARTS)
    ]
    if not relevant_messages and not relevant_files:
        return None
    url = payload.get("html_url")
    label = f"[Upstream `main`]({url})" if url else "Upstream `main`"
    details = []
    if relevant_messages:
        details.append("commits: " + "; ".join(relevant_messages[:4]))
    if relevant_files:
        details.append("files: " + ", ".join(relevant_files[:8]))
    return (
        f"{label}: релевантные изменения `{short_sha(old_sha)}` → `{short_sha(new_sha)}` — "
        + " | ".join(details)
        + "."
    )


def truncate(value: Any, limit: int) -> str:
    text = clean_text(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def baseline_notification(state: dict[str, Any], mention: str) -> str:
    lines = [
        f"@{mention} **FreeToken ROCm Watch запущен.** Проверка выполняется каждые 2 часа; при отсутствии существенных изменений комментариев не будет.",
        "",
        "Текущий P0 baseline:",
    ]
    prs = state.get("tracked_prs", {})
    for number in P0_PRS:
        item = prs.get(str(number))
        if not item:
            continue
        lines.append(
            f"- PR #{number}: `{item.get('state')}`"
            f"{' / draft' if item.get('draft') else ''}, head `{short_sha(item.get('head_sha'))}`, "
            f"checks `{checks_label(item.get('checks'))}` — {item_link(item, item.get('title') or '')}"
        )
    issues = state.get("tracked_issues", {})
    issue_labels = []
    for number in P0_ISSUES:
        item = issues.get(str(number))
        if item:
            issue_labels.append(item_link(item, f"#{number}"))
    if issue_labels:
        lines.extend(["", "Ключевые issues: " + ", ".join(issue_labels) + "."])
    lines.extend(
        [
            "",
            "Дополнительно отслеживаются fork commits, ROCm/TheRock, Triton Windows, TVM-FFI, PyTorch `gfx1201`, Qwen3.6/NVFP4, mapped host pointers, pin-budget, long-prefill и offload stability.",
        ]
    )
    if state.get("errors"):
        lines.extend(["", "⚠️ На baseline были частичные ошибки API: " + "; ".join(state["errors"][:3])])
    return "\n".join(lines)


def change_notification(changes: list[str], mention: str, checked_at: str) -> str:
    max_lines = 35
    shown = changes[:max_lines]
    remaining = len(changes) - len(shown)
    lines = [
        f"@{mention} **FreeToken ROCm Watch: обнаружены существенные изменения** (`{checked_at}`).",
        "",
    ]
    lines.extend(f"- {change}" for change in shown)
    if remaining:
        lines.append(f"- …ещё {remaining} изменений скрыто, чтобы комментарий не превращался в свалку.")
    lines.extend(
        [
            "",
            "Перед следующим Codex-таском стоит сначала сверить эти изменения с текущим blocker/патчем.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "Castoff995/FreeToken-rocm-test").strip()
    pr_number = int(os.environ.get("WATCH_PR_NUMBER", "1"))
    mention = os.environ.get("WATCH_MENTION", "Castoff995").strip().lstrip("@")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    client = GitHubClient(token)
    comments = list_all_comments(client, repo, pr_number)
    old_state, state_comment_id = parse_state_comment(comments)

    collector = Collector(client, old_state)
    new_state = collector.collect()

    if old_state:
        changes = diff_states(client, old_state, new_state)
        if changes:
            client.post(
                f"/repos/{repo}/issues/{pr_number}/comments",
                {"body": change_notification(changes, mention, new_state["last_checked"])},
            )
            print(f"posted notification with {len(changes)} change(s)")
        else:
            print("no meaningful changes")
    else:
        client.post(
            f"/repos/{repo}/issues/{pr_number}/comments",
            {"body": baseline_notification(new_state, mention)},
        )
        print("posted initial baseline")

    body = state_comment_body(new_state)
    if state_comment_id is None:
        client.post(
            f"/repos/{repo}/issues/{pr_number}/comments",
            {"body": body},
        )
        print("created state comment")
    else:
        client.patch(
            f"/repos/{repo}/issues/comments/{state_comment_id}",
            {"body": body},
        )
        print("updated state comment")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
