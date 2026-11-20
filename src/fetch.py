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
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest / "release_v0.zip"
        with requests.get(URL, auth=(user, pw), stream=True, timeout=60) as r:
            if r.status_code == 401:
                print("Derm7pt login was refused; check DERM7PT_USER and DERM7PT_PASS")
                return None
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
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
