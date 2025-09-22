import json
import os
import re
import time
import shutil
import subprocess
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import httpx  
except ImportError:
    httpx = None


def build_headers(url: str, referer: str | None = None, include_range: bool = False) -> dict:
    parsed = urlparse(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "DNT": "1",
    }

    if referer:
        headers["Referer"] = referer
    else:
        # Sensible default referer guess
        if "osha.gov" in parsed.netloc.lower():
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/publications"
        else:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

    if include_range:
        headers["Range"] = "bytes=0-"

    return headers


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def stream_to_file(resp: requests.Response, file_path: str) -> None:
    with open(file_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def parent_dir_referer(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path and "/" in parsed.path:
        parent_path = parsed.path.rsplit("/", 1)[0]
        return f"{parsed.scheme}://{parsed.netloc}{parent_path}/"
    return f"{parsed.scheme}://{parsed.netloc}/"


def uppercase_osha_pdf_variant(url: str) -> str | None:
    # If last segment matches osha####.pdf (any case), try OSHA####.pdf
    parsed = urlparse(url)
    last = parsed.path.rsplit("/", 1)[-1]
    m = re.match(r"(?i)(osha)(\d{3,5})\.pdf$", last or "")
    if not m:
        return None
    new_last = f"OSHA{m.group(2)}.pdf"
    new_path = parsed.path.rsplit("/", 1)[0] + "/" + new_last
    return urlunparse(parsed._replace(path=new_path))


def warm_up_osha(session: requests.Session, url: str) -> None:
    parsed = urlparse(url)
    if "osha.gov" not in parsed.netloc.lower():
        return
    base = f"{parsed.scheme}://{parsed.netloc}"
    warm_headers = build_headers(url, referer=f"{base}/")
    for wu in [f"{base}/", f"{base}/robots.txt", f"{base}/publications"]:
        try:
            session.get(wu, headers=warm_headers, timeout=15, allow_redirects=True)
        except requests.RequestException:
            pass
        time.sleep(0.2)


def try_requests(session: requests.Session, url: str, headers: dict, file_path: str) -> int:
    resp = session.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
    status = resp.status_code
    if status == 200:
        stream_to_file(resp, file_path)
    resp.close()
    return status


def try_httpx_http2(url: str, headers: dict, cookies: requests.cookies.RequestsCookieJar, file_path: str) -> int:
    if httpx is None:
        return -1
    try:
        # copy cookies
        cookie_dict = requests.utils.dict_from_cookiejar(cookies)
        with httpx.Client(http2=True, headers=headers, cookies=cookie_dict, follow_redirects=True, timeout=30.0) as client:
            r = client.get(url)
            status = r.status_code
            if status == 200:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            return status
    except Exception:
        return -1


def try_curl(url: str, headers: dict, file_path: str) -> int:
    curl_path = shutil.which("curl")
    if not curl_path:
        return -1

    cmd = [curl_path, "-fL", "-o", file_path, url]
    # map headers
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    # be explicit with user-agent and referer if present
    if "User-Agent" in headers:
        cmd.extend(["-A", headers["User-Agent"]])
    if "Referer" in headers:
        cmd.extend(["-e", headers["Referer"]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return 200
        # Map common curl exit codes (22 often maps to HTTP error)
        return 403 if result.returncode == 22 else -1
    except Exception:
        return -1


def download_with_strategies(session: requests.Session, url: str, file_path: str) -> None:
    # 1) Warm-up for OSHA
    warm_up_osha(session, url)

    # 2) Try a sequence of realistic header combos and URL variants
    referers = [
        parent_dir_referer(url),
        f"{urlparse(url).scheme}://{urlparse(url).netloc}/publications",
        f"{urlparse(url).scheme}://{urlparse(url).netloc}/",
    ]

    url_variants = [url]
    upper = uppercase_osha_pdf_variant(url)
    if upper and upper not in url_variants:
        url_variants.append(upper)

    # Try: each url variant with (no-range) then (range)
    for u in url_variants:
        for include_range in (False, True):
            for ref in referers:
                headers = build_headers(u, referer=ref, include_range=include_range)
                status = try_requests(session, u, headers, file_path)
                if status == 200:
                    return
                if status not in (403, 429):  # if it's another error (404 etc.), move on quickly
                    continue

    # 3) HTTP/2 fallback with httpx (if installed)
    for u in url_variants:
        headers = build_headers(u, referer=parent_dir_referer(u), include_range=True)
        status = try_httpx_http2(u, headers, session.cookies, file_path)
        if status == 200:
            return

    # 4) Final fallback to curl (if installed)
    for u in url_variants:
        headers = build_headers(u, referer=parent_dir_referer(u), include_range=True)
        status = try_curl(u, headers, file_path)
        if status == 200:
            return

    raise requests.HTTPError("403 Forbidden after all fallback attempts")


def get_filename_from_url_or_title(url: str, title: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name or "." not in name:
        safe_title = "".join(c for c in (title or "document") if c.isalnum() or c in (" ", "_")).rstrip()
        name = safe_title.lower().replace(" ", "_") + ".pdf"
    return name


def ensure_pdf_extension(filename: str) -> str:
    return filename if "." in filename else filename + ".pdf"


def download_and_restructure(sources_path: str, output_folder: str, updated_sources_path: str):
    """
    Downloads files from a sources.json file and creates a new, updated JSON file
    with the structure: { "filename": ..., "title": ..., "url": ... }.
    """
    if not os.path.exists(sources_path):
        print(f"Error: Input file not found at '{sources_path}'")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Files will be saved to the '{output_folder}/' directory.")

    with open(sources_path, 'r', encoding='utf-8') as f:
        initial_sources = json.load(f)

    restructured_sources = []
    session = create_session()

    for i, item in enumerate(initial_sources):
        title = item.get('title', 'No Title Provided')
        url = item.get('url')

        if not url:
            print(f"\n[!] Skipping '{title}' because it has no URL.")
            continue

        print(f"\nProcessing [{i+1}/{len(initial_sources)}]: '{title}'")
        print(f"  -> Downloading from: {url}")

        try:
            filename = get_filename_from_url_or_title(url, title)
            filename = ensure_pdf_extension(filename)
            file_path = os.path.join(output_folder, filename)

            download_with_strategies(session, url, file_path)

            print(f"  -> Saved as: '{filename}'")

            new_entry = {
                "filename": filename,
                "title": title,
                "url": url
            }
            restructured_sources.append(new_entry)

        except requests.exceptions.HTTPError as e:
            print(f"  [!] FAILED to download. HTTP {getattr(getattr(e, 'response', None), 'status_code', 403)} - {e}")
        except requests.exceptions.RequestException as e:
            print(f"  [!] FAILED to download. Error: {e}")
        except Exception as e:
            print(f"  [!] An unexpected error occurred: {e}")

    if restructured_sources:
        print(f"\nWriting updated data to '{updated_sources_path}'...")
        with open(updated_sources_path, 'w', encoding='utf-8') as f:
            json.dump(restructured_sources, f, indent=2, ensure_ascii=False)
        print("✅ Process complete.")
    else:
        print("\nNo files were downloaded. The output file was not created.")


if __name__ == '__main__':
    input_file = "sources.json"
    download_folder = "industrial-safety-pdfs"
    output_file = "source_updated.json"
    download_and_restructure(input_file, download_folder, output_file)