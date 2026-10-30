"""Writing replay data through the in-process service the way the platform does: concurrent Adds, and retries with
the service's Retry-After on 429/5xx (the platform retries Adds up to 32 times)."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


def add_all(client, headers: dict, bodies: list, concurrency: int = 0, attempts: int = 32) -> None:
    """POST every Add body, `concurrency` at a time (default EVAL_ADD_CONCURRENCY or 16). Order across bodies is
    not guaranteed, as on the platform; the service orders by timestamp and request ordinal."""
    concurrency = concurrency or int(os.environ.get("EVAL_ADD_CONCURRENCY", "16"))

    def one(body):
        for _ in range(attempts):
            r = client.post("/v1/memories/add", headers=headers, json=body)
            if r.status_code == 200:
                return
            if r.status_code not in RETRYABLE:
                raise AssertionError(f"Add {body['request_id']} -> {r.status_code} {r.text[:200]}")
            time.sleep(min(float(r.headers.get("retry-after", "5")), 10.0))
        raise AssertionError(f"Add {body['request_id']} still failing after {attempts} attempts")

    if concurrency <= 1:
        for b in bodies:
            one(b)
        return
    with ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(one, bodies))
