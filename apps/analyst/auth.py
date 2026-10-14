"""Password hashing (scrypt), signed session cookies, login rate limiting.

    python -m apps.analyst.auth set-password USER [--users users.json]

Users live in a JSON file {"user": "scrypt$n$r$p$salt$hash"}; the session
secret is a random 32-byte file created on first start (mode 600).
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

N, R, P = 2 ** 14, 8, 1


def hash_password(pw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=N, r=R, p=P, dklen=32)
    return f"scrypt${N}${R}${P}${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got, base64.b64decode(h))
    except Exception:
        return False


class Users:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def check(self, user: str, pw: str) -> bool:
        stored = self.load().get(user)
        if stored is None:
            hash_password(pw)  # same cost as a real check, so timing does not reveal user names
            return False
        return verify_password(pw, stored)

    def set(self, user: str, pw: str) -> None:
        d = self.load()
        d[user] = hash_password(pw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)


class Sessions:
    """Stateless signed tokens: user|expiry|nonce|hmac."""

    def __init__(self, secret_path: Path, ttl_s: int = 7 * 24 * 3600):
        p = Path(secret_path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(secrets.token_bytes(32))
            os.chmod(p, 0o600)
        self.key = p.read_bytes()
        self.ttl = ttl_s

    def _sig(self, msg: str) -> str:
        return hmac.new(self.key, msg.encode(), hashlib.sha256).hexdigest()

    def issue(self, user: str) -> str:
        msg = f"{user}|{int(time.time()) + self.ttl}|{secrets.token_hex(8)}"
        return msg + "|" + self._sig(msg)

    def verify(self, token: str | None) -> str | None:
        if not token or token.count("|") != 3:
            return None
        msg, sig = token.rsplit("|", 1)
        if not hmac.compare_digest(sig, self._sig(msg)):
            return None
        user, exp, _ = msg.split("|")
        return user if int(exp) > time.time() else None


class RateLimiter:
    """At most `limit` failed logins per `window_s` per client address."""

    def __init__(self, limit: int = 5, window_s: int = 600):
        self.limit, self.window = limit, window_s
        self.fail: dict[str, deque] = defaultdict(deque)

    def allowed(self, key: str) -> bool:
        q = self.fail[key]
        while q and q[0] < time.time() - self.window:
            q.popleft()
        return len(q) < self.limit

    def failed(self, key: str) -> None:
        self.fail[key].append(time.time())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set-password")
    s.add_argument("user")
    s.add_argument("--users", default=str(Path.home() / ".config/edgeinfer-analyst/users.json"))
    s.add_argument("--random", action="store_true", help="generate a password and write it next to users.json")
    a = ap.parse_args()
    users = Users(Path(a.users))
    Path(a.users).parent.mkdir(parents=True, exist_ok=True)
    os.chmod(Path(a.users).parent, 0o700)
    if a.random:
        pw = secrets.token_urlsafe(12)
        out = Path(a.users).with_name(f"initial-password-{a.user}.txt")
        out.write_text(pw + "\n")
        os.chmod(out, 0o600)
        print(f"generated a password for {a.user}; it is in {out} (mode 600)")
    else:
        pw = getpass.getpass("new password: ")
        if pw != getpass.getpass("again: "):
            raise SystemExit("passwords differ")
        if len(pw) < 10:
            raise SystemExit("use at least 10 characters")
    users.set(a.user, pw)
    print(f"password set for {a.user} in {a.users}")


if __name__ == "__main__":
    main()
