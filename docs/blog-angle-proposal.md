# Blog post angle proposal — synthesis of everything we know as of 2026-04-16

**Status:** proposal for Mihai. Not a rewrite of `docs/blog-swift-detector-port.md` — a structured recommendation he can react to.

**Why this exists:** through Run 3 + the codex pushback + the CPU sample + the reliability audit + the tracker/transport research, the angle has shifted twice. This doc pins the current best framing and the receipts that support it, so Mihai isn't reconstructing the synthesis from chat history.

---

## Recommended thesis (one sentence)

**Swift is a viable production language for embedded NVIDIA Jetson ML pipelines, validated by porting a working Python DeepStream detector and measuring the result against the original.**

Why this framing and not the others:

- **Not "Swift vs Python."** Codex flagged that comparison as hand-wavy because the heavy work is NVIDIA's. The honest claim is "Swift hits the same target," not "Swift wins."
- **Not "throughput is NVIDIA's, not ours."** Too absolute — generalizes from a source-limited test.
- **Not "Python crashes, Swift doesn't."** The reliability audit weakens the specific crash count and shows pre-port Swift was worse than Python ever was. Survives only if narrowed to the post-port detector.
- **This framing matches the original Wendy meetup-talk thesis** ("Swift, compiled like C, easier to write than C++, can do production embedded work") and lets the data carry it.

Python is the reference / strawman that proves the target exists; the post is about what it took to hit that target in Swift.

---

## Receipts we now hold

Numeric, cite-able, all in the repo:

1. **Same throughput.** Run 3, 300s concurrent: Swift 21.17 fps, Python 21.16 fps. 6,350 vs 6,349 frames in the window. (`docs/benchmark-python-vs-swift.md` Run 3.)
2. **~1.96× less CPU.** Swift 26.6%, Python 52.1% of one core, mean over 60 samples. Both at ~21 fps doing identical work. (`docs/benchmark-data/cpu-sample-300s.csv`.)
3. **63-120 MB less RSS.** Range depending on Python's interpreter GC state during sampling.
4. **nvmap shared = 421,520 kB bit-exact across two independent concurrent runs days apart.** The strongest single fact in the report.
5. **Post-port Swift uptime: 25.9 h observable with 0 OOMs and 0 auto-restart cycles.** Current incarnation 5.8 h. (`docs/reliability-audit.md`.)
6. **Pre-port Swift validates the WHY of the port.** 11 kernel-logged OOMs in ~6 hours on 2026-04-15, pre-NVMM-fix. This is the leak narrative the existing blog already tells.
7. **Stage 2 redesign (drop MJPEG, adopt WebRTC) shipped in one afternoon.** AsyncStream + typed DetectionFrame made this contained; the same change in pyds would be a callback rewrite touching the GLib mainloop integration.
8. **Hardware path proven end-to-end:** NVDEC H.264 decode, TensorRT FP16 inference (YOLO26n), NvDCF tracking, custom C++ bbox parser plugin, CDI bind-mount of NVIDIA libs, Yocto SDK with DeepStream headers — all working under Swift orchestration.
9. **Coexistence with Python detector** via mediamtx relay, no VIC NVDEC contention. Proves the architecture is composable, not exclusive.

---

## Honest caveats to keep in the post

- "Robust detection pipeline" should read as "robust **orchestration** of NVIDIA's detection pipeline." Swift didn't implement detection; NVIDIA did. Swift's contribution is the host process: zero buffer-map, no leak, 25.9 h uptime.
- The 21 fps ceiling is the camera, not the model. Both detectors are source-limited. Throughput equality is consistent with that.
- Per-component latency histograms are inter-frame interval, not glass-to-glass. Don't report them as latency SLOs.
- Python crashes are real but unpinnable to a specific count (Python has no persistent log on this device). Use "3-5 silent Python crashes over ~1 h of deploy attempts, no trace recoverable" instead of "4×."
- The CPU 1.96× ratio is single-stream. Whether it widens or collapses under multi-stream load is unmeasured. Worth flagging as future work, not claiming as a generalization.

---

## Recommended structure

The existing draft at `docs/blog-swift-detector-port.md` is structurally sound. Suggested edits, not a rewrite:

1. **Add an opening hook** that calls back to the Wendy meetup talk: "I made a bet 3 years ago at a Swift meetup that Swift could do this. Here's the receipt." Link the recording.
2. **Frame the leak section** (the existing "65 MB/s nvmap" story) as the *motivation* for the port, not just an anecdote. The reliability audit confirms 11 OOMs in 6 h — the leak was real and severe.
3. **Add the receipts section** with the numbers above. The CPU 1.96× delta is the freshest finding and probably belongs near the top of the side-by-side.
4. **Add a "what Swift contributed" paragraph** with the orchestration framing — this preempts the "but NVIDIA does the heavy lifting" reader objection.
5. **Close** with the talk recording, the bet, and a short list of the things that aren't done yet (multi-stream test, DataChannel transport, NvSORT tracker swap) — frames the post as a milestone, not a finale.

---

## LinkedIn version (separate post)

Different shape — lead with the personal arc:

> Three years ago at a Swift meetup I bet that Swift could do production-grade embedded ML on NVIDIA edge hardware. Most of the room was skeptical. I never published the recording.
>
> Here it is. And here's what I shipped since to back it up: Python DeepStream detector → Swift port, same camera, same model, same 21 fps. Half the CPU. 25 hours stable. Hardware H.264 decode, TensorRT FP16, NvDCF tracking — all driven from Swift, on a Jetson Orin Nano.
>
> Full write-up: [link to blog]
> Talk recording: [link]

Short, narrative, no numbers in the lead — let curiosity drive the click.

---

## Open editorial decisions for Mihai

These each strengthen specific parts of the thesis. None are blocking.

| Decision | Effort | Strengthens |
|---|---|---|
| Run cold-start measurement (5 cold deploys per detector, time-to-first-detection) | ~1 h | Operator UX angle. Codex's #1 ask. |
| Run multi-stream K=1..8 test | ~2 h infra + Swift code change | Could promote the post to "Swift scales further" headline if Python's GIL caps it; or refute the asymmetric-scaling claim |
| Apply tracker class filter (`operate-on-class-ids=0;1;2;3;5;7`) | 1-line config + redeploy | Tracker efficiency angle, real CPU win, demo-worthy |
| Switch NvDCF → NvSORT | hours of tuning + revalidation | Bigger CPU win, but more risk of ID-switch regression |
| Ship DataChannel + RVFC `rtpTimestamp` | 2-3 days | Closes the bbox-lead UX question entirely; adds a clean technical-narrative chapter |
| HANDOFF.md corrections per reliability audit (§10 #5 on auto-restart, §10b #21 on "4×") | 10 min, your voice | Consistency for future sessions |

My ranking if you want a single recommendation: **cold-start measurement first** (cheapest, clean number), **then class filter** (one-line config, low risk, real win, demo-worthy), **then publish**. Defer multi-stream and DataChannel as "future work" call-outs in the post.

---

## What NOT to claim

- "Swift beats Python at perf." We matched fps; the wins are CPU and RSS at the host-process layer.
- "Swift's robust detection pipeline." Swift's robust **orchestration**. The detection robustness is NVIDIA's.
- "Python crashed exactly 4 times." Use 3-5 with no trace recoverable.
- "Throughput is NVIDIA's, not ours." Too absolute. Say "in this source-limited workload, accelerator stages dominate, so the port did not change steady-state fps."
- "Single-stream CPU savings prove Swift scales further." That requires the multi-stream curve.
