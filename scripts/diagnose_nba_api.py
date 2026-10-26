"""
Pinpoint where requests to stats.nba.com fail, one layer at a time.

Makes at most 4 HTTP requests with pauses between them, so it is safe to run
without tripping rate limits. Usage: python scripts/diagnose_nba_api.py
"""
import socket
import ssl
import sys
import time
from datetime import datetime, timedelta

import requests

sys.path.insert(0, "scripts")

HOST = "stats.nba.com"
PAUSE = 4  # seconds between HTTP requests
TIMEOUT = 20


def step(name, fn):
    print(f"\n[{name}]")
    start = time.time()
    try:
        detail = fn()
        print(f"  OK in {time.time() - start:.1f}s {detail or ''}")
        return True
    except Exception as e:
        print(f"  FAILED after {time.time() - start:.1f}s: {type(e).__name__}: {e}")
        return False


def dns():
    return f"-> {sorted({a[4][0] for a in socket.getaddrinfo(HOST, 443)})}"


def tls_handshake():
    # Connects and completes TLS but sends no HTTP request
    ctx = ssl.create_default_context()
    with socket.create_connection((HOST, 443), timeout=TIMEOUT) as sock:
        with ctx.wrap_socket(sock, server_hostname=HOST) as tls:
            return f"-> {tls.version()}, cert for {dict(x[0] for x in tls.getpeercert()['subject'])['commonName']}"


def nba_homepage():
    r = requests.get("https://www.nba.com/", timeout=TIMEOUT)
    return f"-> HTTP {r.status_code}, {len(r.content)} bytes"


def cdn_live_scoreboard():
    from nba_api.live.nba.endpoints import scoreboard
    games = scoreboard.ScoreBoard(timeout=TIMEOUT).get_dict()["scoreboard"]["games"]
    return f"-> {len(games)} games today (cdn.nba.com, library default headers)"


def stats_library_defaults():
    from nba_api.stats.endpoints import scoreboardv2
    day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    df = scoreboardv2.ScoreboardV2(game_date=day, timeout=TIMEOUT).get_data_frames()[0]
    return f"-> {len(df)} games on {day} (stats.nba.com, library default headers)"


def stats_pipeline_config():
    from nba_api.stats.endpoints import scoreboardv2
    from update_frontend_data import CUSTOM_HEADERS
    day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    df = scoreboardv2.ScoreboardV2(game_date=day, headers=CUSTOM_HEADERS, timeout=TIMEOUT).get_data_frames()[0]
    return f"-> {len(df)} games on {day} (stats.nba.com, pipeline headers)"


def main():
    import nba_api
    print(f"nba_api {getattr(nba_api, '__version__', '?')}, requests {requests.__version__}, Python {sys.version.split()[0]}")

    results = {}
    results["dns"] = step("1. DNS lookup for stats.nba.com", dns)
    results["tls"] = step("2. TCP + TLS handshake with stats.nba.com (no HTTP request)", tls_handshake)
    results["home"] = step("3. GET www.nba.com", nba_homepage)
    time.sleep(PAUSE)
    results["cdn"] = step("4. Live scoreboard from cdn.nba.com", cdn_live_scoreboard)
    time.sleep(PAUSE)
    results["stats"] = step("5. One stats.nba.com call (library defaults)", stats_library_defaults)
    if not results["stats"]:
        time.sleep(PAUSE)
        results["stats_pipeline"] = step("6. Same call with the pipeline's headers", stats_pipeline_config)

    print("\n--- Diagnosis ---")
    if not results["dns"] or not results["tls"]:
        print("Can't reach stats.nba.com at the network level: check internet, DNS, firewall or VPN.")
    elif results["stats"] or results.get("stats_pipeline"):
        print("stats.nba.com is answering. The pipeline should work; earlier failures were temporary.")
    else:
        print("The connection and TLS handshake succeed, but stats.nba.com never answers the HTTP request.")
        print("That is a server-side silent drop, not a network outage or a code bug. It matches")
        print("nba_api issues #652/#728 (from March 2026): NBA's bot protection now drops requests")
        print("from scripts. Retrying or changing timeouts won't fix it.")
        if results["cdn"]:
            print("cdn.nba.com still works, so live/scoreboard data is reachable from this machine.")


if __name__ == "__main__":
    main()
