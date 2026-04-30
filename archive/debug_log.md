# Debug Log

Append-only. Format: YYYY-MM-DD | phase | issue | resolution

2026-04-30 | Phase 2 | datasets 3.x requires trust_remote_code=True for PolyAI/banking77 (custom loading script) | added trust_remote_code=True to all load_dataset calls in src/data.py and scripts/data_stats.py
