"""Download Derm7pt inside a Kaggle notebook.

Derm7pt is behind a username and password that the dataset owners send after registration.
Save them as the Kaggle secrets DERM7PT_USER and DERM7PT_PASS; this downloads and unzips the
release into /kaggle/working/data and points DERM7PT_DIR at it. Nothing happens without them.
"""
import os
import zipfile
from pathlib import Path

URL = "http://derm.cs.sfu.ca/restricted/release_v0.zip"


def derm7pt(dest="/kaggle/working/data/derm7pt"):
    dest = Path(dest)
    meta = list(dest.rglob("meta.csv"))
    if not meta:
        try:
            from kaggle_secrets import UserSecretsClient
            s = UserSecretsClient()
            user, pw = s.get_secret("DERM7PT_USER"), s.get_secret("DERM7PT_PASS")
        except Exception:
            print("Derm7pt secrets not set; running without Derm7pt")
            return None
        import requests
        from requests.auth import HTTPBasicAuth, HTTPDigestAuth
        user, pw = user.strip(), pw.strip()
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest / "release_v0.zip"
        ok = False
        for url in (URL.replace("http://", "https://"), URL):
            for auth in (HTTPBasicAuth(user, pw), HTTPDigestAuth(user, pw)):
                with requests.get(url, auth=auth, stream=True, timeout=60) as r:
                    if r.status_code == 401:
                        print(f"{url} {type(auth).__name__}: refused "
                              f"(server asks for: {r.headers.get('WWW-Authenticate', '?').split(' ')[0]})")
                        continue
                    r.raise_for_status()
                    with open(zpath, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    ok = True
                    break
            if ok:
                break
        if not ok:
            print("Derm7pt login was refused; check DERM7PT_USER and DERM7PT_PASS "
                  f"(username has {len(user)} characters, password has {len(pw)})")
            return None
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        zpath.unlink()
        meta = list(dest.rglob("meta.csv"))
    meta = [m for m in meta if m.parent.name == "meta"]
    if not meta:
        print("Derm7pt download did not contain meta/meta.csv")
        return None
    root = meta[0].parent.parent
    os.environ["DERM7PT_DIR"] = str(root)
    print("Derm7pt ready at", root)
    return root


def setup_derm7pt(work="/kaggle/working/work"):
    """Download Derm7pt if the secrets are set. If an earlier prepare step ran without it,
    remove that sample so the next prepare includes Derm7pt. Model answers already saved are kept."""
    import shutil
    import pandas as pd
    root = derm7pt()
    man = Path(work) / "manifest.csv"
    if root and man.exists() and "derm7pt" not in set(pd.read_csv(man).dataset):
        man.unlink()
        for d in ("images", "masks"):
            shutil.rmtree(Path(work) / d, ignore_errors=True)
        print("earlier sample had no Derm7pt; it will be rebuilt")
    return root
