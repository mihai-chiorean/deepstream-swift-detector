# Reliability audit — Python vs Swift detector, 2026-04-15 → 2026-04-16

**Status:** research notes. Produced to pressure-test the claim "Python detector crashed silently 4× this session; Swift hasn't crashed once" after codex flagged it as the strongest practical but weakest evidentiary point.

**Window audited:** 2026-04-15 00:00 UTC → 2026-04-16 21:14 UTC (~45.2 h).

---

## Numeric summary

| Metric | Python (`detector` / `deepstream-vision`) | Swift (`detector-swift`) |
|---|---|---|
| Current incarnation uptime | 0.75 h (PID 12454, started 2026-04-16 20:30:33 UTC) | **5.78 h** (since 2026-04-16 15:27:21 UTC, post-port) |
| Distinct `Creating container` launches in window | 7 (some `alreadyExists` races, not true relaunches) | 78 |
| OOM kills of the detector binary | **0** | **11** — all 2026-04-15, all **pre-port** software-decode Swift, oom_score_adj=300, RSS 4–7 GB |
| wendy-agent auto-restart cycles | 0 | **2799** — all 2026-04-15 06:27:54 → 18:34:58 UTC (~12 h). A single initial OOM-triggered crash, with the wendy-agent `ContainerMonitor` then rebuilding the pipeline every ~15s for 12 hours. Count as ONE incident. |
| Median uptime across launches | (insufficient observable launches to compute) | 214 s (n=78, min 3 s, p95 4210 s, max 20803 s) |
| Longest post-port uninterrupted run | Current 0.75 h, metrics healthy (20 fps, 56k frames) | Current 5.78 h; second-longest 3.48 h |
| Core dumps captured | 0 | 0 |

All 11 OOM kills and the 2799-cycle auto-restart storm happened on the **pre-port Swift detector** — software-decode `avdec_h264 → videoconvert → appsink`, the pipeline that the blog's "65 MB/s nvmap leak" narrative is about. The DeepStream-native Swift (first launch 2026-04-15 19:18:29) has had **0 OOMs and 0 auto-restart cycles** in ~25.9 h of observable operation.

---

## Measurement limits

1. **Python has no persistent log.** `/var/lib/wendy-agent/storage/detector-python-cache/` is empty. Pre-current-incarnation output only lived in containerd's task log buffer and is unrecoverable. **All prior Python crashes are unattributable from data** — we have Mihai's stop markers but no cause. Swift, by contrast, keeps `/var/lib/wendy-agent/storage/detector-cache/detector.log` at 1.73 M lines going back to 2026-04-11.
2. **Containerd auto-restart confound.** HANDOFF.md §10 #5 says "we avoided `--restart-unless-stopped`." This is **partially wrong.** `ctr containers info detector` confirms containerd's restart policy is `"no"`. But the **wendy-agent `ContainerMonitor` demonstrably auto-restarted detector-swift 2799 times** independently of containerd's policy — the agent layer has its own restart logic that fires on abnormal exit. So "manual restart needed vs self-healed" is not cleanly decidable from journal signals alone; the wendy-agent monitor heals by default at the Go/Swift layer. HANDOFF §10 #5 should be rewritten to reflect this.
3. **Pre-port vs post-port Swift must be distinguished everywhere.** The 11 OOMs and 2799-cycle loop are pre-port. The current detector is post-port. Conflating these produces false negatives.
4. **Session anchor is fuzzy.** Wendy-agent service restarted 9 times between 2026-04-14 20:25 and 2026-04-16 07:30; continuity of internal counters across those boundaries is not assumed.

---

## Does the "4× silent Python crashes" claim survive?

**Weakened, but directionally defensible.** The source is `docs/HANDOFF.md:621`. There is no logged counter that says "4." What the evidence supports:

- Python definitely crashed more than zero times. `benchmark-python-vs-swift.md` cites a silent crash during Run 2. Early-morning 2026-04-16 redeploys at 05:58, 05:59, 06:01, 06:11, 06:31 UTC are consistent with "start → die → redeploy" cycles — 5 container-recreate events in 33 minutes. That's plausibly 3–5 silent crashes.
- Swift crashed MORE in the same window, but **all pre-port**. 11 kernel-logged OOMs + one restart storm.
- The post-port Swift has 0 abnormal exits in ~26 h of observable operation.

**Recommended blog language:** drop the specific "4×" and replace with something like —

> The Python detector wedged silently several times during the benchmark session — restart counts in the low single digits over ~1 h of deploy attempts, no crash trace recoverable. The DeepStream-native Swift rewrite, once shipped, has run a 5.8 h uninterrupted incarnation through the current window with no abnormal exits.

If you want a number: **"3–5 silent Python crashes, ~1 h of deploy wrangling"** is what the data supports. "4" sits inside that range but isn't pinned. Anything more confident than "3–5 with no trace recoverable" is unsupported.

---

## Citable incidents

### Python silent-crash fingerprint (2026-04-16 morning benchmark window)

```
Apr 16 05:58:49 wendy-agent: info Creating container app-name=detector
Apr 16 05:59:03 wendy-agent: error alreadyExists: container "detector": already exists
Apr 16 05:59:03 wendy-agent: info ContainerMonitor: container=detector Marked explicitly stopped
Apr 16 05:59:28 wendy-agent: error alreadyExists: container "detector": already exists
... [5 recreate cycles in 33 minutes, container record never cleanly torn down]
```

Interpretation: the process exited (crashed) but the containerd container record wasn't cleaned up. Classic silent-segfault footprint: no exit status propagated, no trace.

### Swift pre-port OOM

```
Apr 15 00:15:18 kernel: Out of memory: Killed process 18812 (Detector)
  total-vm:19272904kB, anon-rss:4053280kB, file-rss:0kB, shmem-rss:0kB,
  UID:0 pgtables:16896kB oom_score_adj:300
```

Citable if and only if it's explicitly framed as "pre-port Swift, before the NVMM fix." This is the thing the port went on to fix.

---

## Impact on the blog post

1. The post's leak narrative (pre-port Swift leaked 65 MB/s nvmap) is fully supported by the 11 OOM kills on 2026-04-15.
2. The "post-port Swift is stable" claim is fully supported by 25.9 h of observable DeepStream-native operation with zero OOMs and zero auto-restarts.
3. The "Python crashes more than Swift" claim should be narrowed to "Python has silent startup failures the post-port Swift doesn't." Absolute crash counts favor post-port Swift; pre-port numbers are worse than Python's and should not be cited against Python.
4. The implicit arc of the port — "broken → fixed" — is cleanly supported: pre-port Swift's pathology (the OOM storm) is the entire reason the port happened.

---

## Flagged corrections

- `docs/HANDOFF.md:621` — "Python detector died four times" is unpinnable. Replace with "3–5 silent Python crashes over ~1 h of deploy wrangling; no cause recoverable because Python has no persistent log" or similar.
- `docs/HANDOFF.md:564` (§10 #5) — the "we avoided `--restart-unless-stopped`" claim is misleading. Containerd's policy is `"no"` but the wendy-agent `ContainerMonitor` auto-restarts independently. Both layers' restart behavior must be stated together.

These are in Mihai's cold-start orientation doc; not touched here. Apply them in his voice when convenient.
