from __future__ import annotations

import os

import psutil


def rss_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss
