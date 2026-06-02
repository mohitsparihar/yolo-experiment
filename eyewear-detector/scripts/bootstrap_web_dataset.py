"""Collect web images with Playwright, auto-label them with Grounding DINO, and optionally train YOLO.

This script is intentionally conservative:
- It downloads a small, query-diverse image set instead of bulk scraping.
- It pseudo-labels with the existing Grounding DINO product finder.
- It filters detections aggressively to reduce noisy labels.
- It stores the final YOLO dataset directly and deletes staging files by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.product_finder import load_product_finder, find_products

RANDOM_SEED = 42
VAL_SPLIT = 0.2
MAX_INFERENCE_SIZE = 1024
DEFAULT_MIN_CONF = 0.45
DEFAULT_MIN_SIDE = 256
DEFAULT_MIN_BOX_AREA = 0.01
DEFAULT_MAX_BOX_AREA = 0.85
DEFAULT_MAX_BOXES = 8
DEFAULT_QUERY_LIMIT = 40
MIN_TRAIN_IMAGES = 20

DEFAULT_QUERIES = [
    "eyeglasses product photo",
    "sunglasses product photo",
    "multiple eyeglasses on table",
    "glasses flat lay",
    "people wearing eyeglasses",
    "group wearing sunglasses",
]


def resize_for_inference(image: Image.Image) -> Image.Image:
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_INFERENCE_SIZE:
        return image
    scale = MAX_INFERENCE_SIZE / max_dim
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def bbox_xyxy_to_cxcywh(bbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


async def collect_bing_image_urls(query: str, limit: int, headed: bool = False) -> list[str]:
    from playwright.async_api import async_playwright

    search_url = "https://www.bing.com/images/search?q=" + urllib.parse.quote_plus(query)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headed)
        page = await browser.new_page()
        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)

        # Scroll a few times to let image results hydrate.
        for _ in range(4):
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(1200)

        urls = await page.evaluate(
            """(limit) => {
                const seen = new Set();
                const out = [];
                for (const node of document.querySelectorAll('a.iusc')) {
                    const raw = node.getAttribute('m');
                    if (!raw) continue;
                    try {
                        const parsed = JSON.parse(raw);
                        const url = parsed.murl || parsed.turl;
                        if (!url || seen.has(url)) continue;
                        seen.add(url);
                        out.push(url);
                        if (out.length >= limit) break;
                    } catch (_) {}
                }
                return out;
            }""",
            limit,
        )
        await browser.close()
        return [url for url in urls if isinstance(url, str)]


async def collect_bing_image_urls_with_page(page, query: str, limit: int) -> list[str]:
    search_url = "https://www.bing.com/images/search?q=" + urllib.parse.quote_plus(query)
    await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)

    for _ in range(4):
        await page.mouse.wheel(0, 2500)
        await page.wait_for_timeout(1200)

    urls = await page.evaluate(
        """(limit) => {
            const seen = new Set();
            const out = [];
            for (const node of document.querySelectorAll('a.iusc')) {
                const raw = node.getAttribute('m');
                if (!raw) continue;
                try {
                    const parsed = JSON.parse(raw);
                    const url = parsed.murl || parsed.turl;
                    if (!url || seen.has(url)) continue;
                    seen.add(url);
                    out.push(url);
                    if (out.length >= limit) break;
                } catch (_) {}
            }
            return out;
        }""",
        limit,
    )
    return [url for url in urls if isinstance(url, str)]


async def collect_google_image_urls(
    query: str,
    limit: int,
    headed: bool = False,
    pause_for_manual: bool = False,
) -> list[str]:
    from playwright.async_api import async_playwright

    search_url = "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote_plus(query)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headed)
        page = await browser.new_page()
        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)

        if headed and pause_for_manual:
            print("[INFO] Google page opened. Complete any verify-human or consent step in the browser.")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: input("[ACTION] Press Enter here after the page is ready with visible image results... "),
            )

        # Consent pages appear inconsistently by region/account state.
        for selector in [
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Reject all")',
            'button:has-text("No thanks")',
        ]:
            try:
                await page.locator(selector).first.click(timeout=2000)
                await page.wait_for_timeout(1500)
                break
            except Exception:
                pass

        for _ in range(5):
            await page.mouse.wheel(0, 2800)
            await page.wait_for_timeout(1200)

        async def extract_urls() -> list[str]:
            return await page.evaluate(
            """(limit) => {
                const out = [];
                const seen = new Set();

                const pushUrl = (value) => {
                    if (!value || seen.has(value)) return;
                    if (!/^https?:\\/\\//i.test(value)) return;
                    if (/google\\.com\\/images/i.test(value)) return;
                    seen.add(value);
                    out.push(value);
                };

                for (const anchor of document.querySelectorAll('a[href*="imgurl="], a[href*="/imgres?"], a[href^="http"]')) {
                    try {
                        const parsed = new URL(anchor.href, window.location.origin);
                        pushUrl(parsed.searchParams.get('imgurl'));
                        pushUrl(parsed.searchParams.get('imgrefurl'));
                        pushUrl(anchor.href);
                        if (out.length >= limit) return out;
                    } catch (_) {}
                }

                for (const img of document.querySelectorAll('img')) {
                    pushUrl(img.getAttribute('data-iurl'));
                    pushUrl(img.getAttribute('data-src'));
                    pushUrl(img.getAttribute('data-src'));
                    pushUrl(img.getAttribute('src'));
                    if (out.length >= limit) return out;
                }

                return out;
            }""",
            limit,
        )

        async def extract_preview_image_urls(extra_limit: int) -> list[str]:
            return await page.evaluate(
                """(limit) => {
                    const out = [];
                    const seen = new Set();

                    const pushUrl = (value) => {
                        if (!value || seen.has(value)) return;
                        if (!/^https?:\\/\\//i.test(value)) return;
                        if (/google\\./i.test(value)) return;
                        seen.add(value);
                        out.push(value);
                    };

                    const imgs = Array.from(document.querySelectorAll('img'));
                    for (const img of imgs) {
                        const width = img.naturalWidth || 0;
                        const height = img.naturalHeight || 0;
                        if (Math.max(width, height) < 200) continue;
                        pushUrl(img.currentSrc || img.src);
                        pushUrl(img.getAttribute('data-src'));
                        pushUrl(img.getAttribute('data-iurl'));
                        if (out.length >= limit) break;
                    }

                    return out;
                }""",
                extra_limit,
            )
        urls = await extract_urls()

        if len(urls) < limit:
            thumb_selectors = [
                'img[jsname]',
                'a[href*="/imgres?"] img',
                'div[role="button"] img',
            ]
            clicked = 0
            preview_urls: list[str] = []
            for selector in thumb_selectors:
                thumbs = page.locator(selector)
                count = await thumbs.count()
                for idx in range(min(count, max(limit * 2, 8))):
                    try:
                        await thumbs.nth(idx).click(timeout=1500)
                        await page.wait_for_timeout(800)
                        preview_urls.extend(await extract_preview_image_urls(limit * 3))
                        clicked += 1
                    except Exception:
                        continue
                    if len(preview_urls) >= limit or clicked >= max(limit, 8):
                        break
                if len(preview_urls) >= limit or clicked >= max(limit, 8):
                    break

            merged: list[str] = []
            seen = set()
            for candidate in preview_urls + urls + await extract_urls():
                if candidate in seen:
                    continue
                seen.add(candidate)
                merged.append(candidate)
                if len(merged) >= limit:
                    break
            urls = merged
        await browser.close()
        return [url for url in urls if isinstance(url, str)]


async def collect_google_image_urls_with_page(
    page,
    query: str,
    limit: int,
    pause_for_manual: bool = False,
) -> list[str]:
    search_url = "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote_plus(query)
    await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)

    if pause_for_manual:
        print("[INFO] Google page opened. Complete any verify-human or consent step in the browser.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: input("[ACTION] Press Enter here after the page is ready with visible image results... "),
        )

    for selector in [
        'button:has-text("Accept all")',
        'button:has-text("I agree")',
        'button:has-text("Reject all")',
        'button:has-text("No thanks")',
    ]:
        try:
            await page.locator(selector).first.click(timeout=2000)
            await page.wait_for_timeout(1500)
            break
        except Exception:
            pass

    for _ in range(5):
        await page.mouse.wheel(0, 2800)
        await page.wait_for_timeout(1200)

    async def extract_urls() -> list[str]:
        return await page.evaluate(
        """(limit) => {
            const out = [];
            const seen = new Set();

            const pushUrl = (value) => {
                if (!value || seen.has(value)) return;
                if (!/^https?:\\/\\//i.test(value)) return;
                if (/google\\.com\\/images/i.test(value)) return;
                seen.add(value);
                out.push(value);
            };

            for (const anchor of document.querySelectorAll('a[href*="imgurl="], a[href*="/imgres?"], a[href^="http"]')) {
                try {
                    const parsed = new URL(anchor.href, window.location.origin);
                    pushUrl(parsed.searchParams.get('imgurl'));
                    pushUrl(parsed.searchParams.get('imgrefurl'));
                    pushUrl(anchor.href);
                    if (out.length >= limit) return out;
                } catch (_) {}
            }

            for (const img of document.querySelectorAll('img')) {
                pushUrl(img.getAttribute('data-iurl'));
                pushUrl(img.getAttribute('data-src'));
                pushUrl(img.getAttribute('data-src'));
                pushUrl(img.getAttribute('src'));
                if (out.length >= limit) return out;
            }

            return out;
        }""",
        limit,
    )

    async def extract_preview_image_urls(extra_limit: int) -> list[str]:
        return await page.evaluate(
            """(limit) => {
                const out = [];
                const seen = new Set();

                const pushUrl = (value) => {
                    if (!value || seen.has(value)) return;
                    if (!/^https?:\\/\\//i.test(value)) return;
                    if (/google\\./i.test(value)) return;
                    seen.add(value);
                    out.push(value);
                };

                const imgs = Array.from(document.querySelectorAll('img'));
                for (const img of imgs) {
                    const width = img.naturalWidth || 0;
                    const height = img.naturalHeight || 0;
                    if (Math.max(width, height) < 200) continue;
                    pushUrl(img.currentSrc || img.src);
                    pushUrl(img.getAttribute('data-src'));
                    pushUrl(img.getAttribute('data-iurl'));
                    if (out.length >= limit) break;
                }

                return out;
            }""",
            extra_limit,
        )

    urls = await extract_urls()

    if len(urls) < limit:
        thumb_selectors = [
            'img[jsname]',
            'a[href*="/imgres?"] img',
            'div[role="button"] img',
        ]
        clicked = 0
        preview_urls: list[str] = []
        for selector in thumb_selectors:
            thumbs = page.locator(selector)
            count = await thumbs.count()
            for idx in range(min(count, max(limit * 2, 8))):
                try:
                    await thumbs.nth(idx).click(timeout=1500)
                    await page.wait_for_timeout(800)
                    preview_urls.extend(await extract_preview_image_urls(limit * 3))
                    clicked += 1
                except Exception:
                    continue
                if len(preview_urls) >= limit or clicked >= max(limit, 8):
                    break
            if len(preview_urls) >= limit or clicked >= max(limit, 8):
                break

        merged: list[str] = []
        seen = set()
        for candidate in preview_urls + urls + await extract_urls():
            if candidate in seen:
                continue
            seen.add(candidate)
            merged.append(candidate)
            if len(merged) >= limit:
                break
        urls = merged

    return [url for url in urls if isinstance(url, str)]


async def collect_image_urls(
    engine: str,
    query: str,
    limit: int,
    headed: bool = False,
    pause_for_manual: bool = False,
) -> list[str]:
    if engine == "google":
        return await collect_google_image_urls(
            query,
            limit,
            headed=headed,
            pause_for_manual=pause_for_manual,
        )
    return await collect_bing_image_urls(query, limit, headed=headed)


async def collect_image_urls_with_page(
    page,
    engine: str,
    query: str,
    limit: int,
    pause_for_manual: bool = False,
) -> list[str]:
    if engine == "google":
        return await collect_google_image_urls_with_page(
            page,
            query,
            limit,
            pause_for_manual=pause_for_manual,
        )
    return await collect_bing_image_urls_with_page(page, query, limit)


def download_image(url: str, destination: Path, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return "not_image"
            destination.write_bytes(response.read())
        return "ok"
    except Exception:
        return "download_error"


def verify_and_normalize_image(path: Path, min_side: int) -> bool:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            if min(w, h) < min_side:
                return False
            image.save(path, format="JPEG", quality=92)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def clear_dataset_dir(dataset_dir: Path) -> None:
    for name in ["images", "labels"]:
        target = dataset_dir / name
        if target.exists():
            shutil.rmtree(target)


def write_dataset_yaml(dataset_dir: Path, class_name: str) -> Path:
    yaml_path = dataset_dir / "product_finder.yaml"
    yaml_content = f"""# Auto-generated web dataset
path: {dataset_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: {class_name}
"""
    yaml_path.write_text(yaml_content)
    return yaml_path


def should_keep_box(bbox: list[float], min_area: float, max_area: float) -> bool:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return min_area <= area <= max_area


def build_dataset(
    images: list[tuple[Path, str, str]],
    dataset_dir: Path,
    min_conf: float,
    min_box_area: float,
    max_box_area: float,
    max_boxes: int,
) -> tuple[Path, int, int]:
    random.seed(RANDOM_SEED)
    load_product_finder(backend="dino")

    labeled: list[tuple[Path, list[str]]] = []
    rejected = 0

    for img_path, query, source_url in images:
        try:
            image = Image.open(img_path).convert("RGB")
            inference_image = resize_for_inference(image)
            products = find_products(inference_image)
        except Exception:
            rejected += 1
            continue

        labels: list[str] = []
        for product in products:
            if product["confidence"] < min_conf:
                continue
            if not should_keep_box(product["bbox"], min_box_area, max_box_area):
                continue
            cx, cy, w, h = bbox_xyxy_to_cxcywh(product["bbox"])
            labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not labels or len(labels) > max_boxes:
            rejected += 1
            continue

        labeled.append((img_path, labels))

    random.shuffle(labeled)
    split_idx = int(len(labeled) * (1 - VAL_SPLIT))
    train_data = labeled[:split_idx]
    val_data = labeled[split_idx:]

    clear_dataset_dir(dataset_dir)
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    image_lookup = {path: {"query": query, "source_url": source_url} for path, query, source_url in images}
    for split, data in [("train", train_data), ("val", val_data)]:
        for img_path, label_lines in data:
            dst_img = dataset_dir / "images" / split / f"{img_path.stem}.jpg"
            dst_label = dataset_dir / "labels" / split / f"{img_path.stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_label.write_text("\n".join(label_lines))
            meta = image_lookup[img_path]
            manifest.append(
                {
                    "split": split,
                    "image": str(dst_img.relative_to(dataset_dir)),
                    "labels": str(dst_label.relative_to(dataset_dir)),
                    "query": meta["query"],
                    "source_url": meta["source_url"],
                    "boxes": len(label_lines),
                }
            )

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    yaml_path = write_dataset_yaml(dataset_dir, "eyewear_product")
    return yaml_path, len(labeled), rejected


def maybe_run_training(
    train: bool,
    yaml_path: Path,
    labeled_count: int,
    epochs: int,
    imgsz: int,
    device: str,
) -> bool:
    if not train:
        return False
    if labeled_count < MIN_TRAIN_IMAGES:
        print(
            f"[WARN] Skipping training: only {labeled_count} labeled images were generated; "
            f"need at least {MIN_TRAIN_IMAGES}."
        )
        print("[WARN] Try increasing --per-query, adding more queries, or lowering --min-conf slightly.")
        return False
    try:
        cmd = [
            sys.executable,
            "training/train_product_finder.py",
            "--data",
            str(yaml_path.resolve()),
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--device",
            device,
        ]
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Training failed with exit code {exc.returncode}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from web images using Playwright + Grounding DINO"
    )
    parser.add_argument("--query", action="append", dest="queries", help="Search query. Repeat for multiple queries.")
    parser.add_argument("--query-file", type=Path, help="Text file with one search query per line.")
    parser.add_argument(
        "--engine",
        choices=["bing", "google"],
        default="bing",
        help="Image search engine to use for URL collection.",
    )
    parser.add_argument("--per-query", type=int, default=DEFAULT_QUERY_LIMIT, help="Images to attempt per query.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("training/datasets/product_finder_web"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--staging-dir", type=Path, help="Optional directory to keep downloaded raw images.")
    parser.add_argument("--keep-staging", action="store_true", help="Keep downloaded raw images after dataset creation.")
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE)
    parser.add_argument("--min-box-area", type=float, default=DEFAULT_MIN_BOX_AREA)
    parser.add_argument("--max-box-area", type=float, default=DEFAULT_MAX_BOX_AREA)
    parser.add_argument("--max-boxes", type=int, default=DEFAULT_MAX_BOXES)
    parser.add_argument("--train", action="store_true", help="Train YOLO immediately after dataset creation.")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Playwright with a visible browser window.",
    )
    parser.add_argument(
        "--pause-for-manual",
        action="store_true",
        help="In headed mode, wait for manual verification before collecting URLs.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def resolve_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.queries or [])
    if args.query_file:
        queries.extend(
            line.strip() for line in args.query_file.read_text().splitlines() if line.strip()
        )
    return queries or DEFAULT_QUERIES


async def gather_images(
    engine: str,
    queries: list[str],
    per_query: int,
    staging_dir: Path,
    min_side: int,
    headed: bool,
    pause_for_manual: bool,
) -> list[tuple[Path, str, str]]:
    from playwright.async_api import async_playwright

    downloaded: list[tuple[Path, str, str]] = []
    seen_hashes: set[str] = set()
    reject_counts = {
        "download_error": 0,
        "not_image": 0,
        "too_small_or_invalid": 0,
        "duplicate": 0,
    }

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headed)
        page = await browser.new_page()
        manual_pause_pending = pause_for_manual and headed and engine == "google"

        try:
            for query in queries:
                print(f"[INFO] Search ({engine}): {query}")
                urls = await collect_image_urls_with_page(
                    page,
                    engine,
                    query,
                    per_query,
                    pause_for_manual=manual_pause_pending,
                )
                manual_pause_pending = False
                print(f"[INFO] Found {len(urls)} candidate URLs")

                kept_for_query = 0
                for idx, url in enumerate(urls):
                    filename = f"{hashlib.sha1(f'{query}-{idx}-{url}'.encode()).hexdigest()[:16]}.jpg"
                    destination = staging_dir / filename
                    download_status = download_image(url, destination)
                    if download_status != "ok":
                        reject_counts[download_status] += 1
                        continue
                    if not verify_and_normalize_image(destination, min_side=min_side):
                        reject_counts["too_small_or_invalid"] += 1
                        destination.unlink(missing_ok=True)
                        continue

                    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                    if digest in seen_hashes:
                        reject_counts["duplicate"] += 1
                        destination.unlink(missing_ok=True)
                        continue

                    seen_hashes.add(digest)
                    downloaded.append((destination, query, url))
                    kept_for_query += 1

                print(f"[INFO] Kept {kept_for_query} images for query")
        finally:
            await browser.close()

    print(
        "[INFO] Download filter summary: "
        f"download_error={reject_counts['download_error']}, "
        f"not_image={reject_counts['not_image']}, "
        f"too_small_or_invalid={reject_counts['too_small_or_invalid']}, "
        f"duplicate={reject_counts['duplicate']}"
    )

    return downloaded


def main() -> int:
    args = parse_args()
    queries = resolve_queries(args)
    dataset_dir = args.dataset_dir.resolve()

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.staging_dir:
        staging_dir = args.staging_dir.resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_staging:
        staging_dir = dataset_dir / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="eyewear_web_")
        staging_dir = Path(temp_dir_obj.name)

    try:
        images = asyncio.run(
            gather_images(
                args.engine,
                queries,
                args.per_query,
                staging_dir,
                args.min_side,
                args.headed,
                args.pause_for_manual,
            )
        )
        print(f"[INFO] Downloaded {len(images)} usable images")
        if not images:
            print("[ERROR] No usable images downloaded")
            return 1

        yaml_path, labeled_count, rejected_count = build_dataset(
            images=images,
            dataset_dir=dataset_dir,
            min_conf=args.min_conf,
            min_box_area=args.min_box_area,
            max_box_area=args.max_box_area,
            max_boxes=args.max_boxes,
        )

        print(f"[INFO] Pseudo-labeled {labeled_count} images")
        print(f"[INFO] Rejected {rejected_count} images during labeling/filtering")
        print(f"[INFO] Dataset written to {dataset_dir}")
        print(f"[INFO] Training config: {yaml_path}")

        trained = maybe_run_training(
            train=args.train,
            yaml_path=yaml_path,
            labeled_count=labeled_count,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
        )
        if args.train and not trained:
            print("[INFO] Dataset was created successfully; training was not started.")
            print(
                "[INFO] Suggested retry: python scripts/bootstrap_web_dataset.py "
                "--query-file scripts/web_queries.txt --per-query 80 "
                "--min-conf 0.35 --dataset-dir training/datasets/product_finder_web --train"
            )
    finally:
        if temp_dir_obj is not None and not args.keep_staging:
            temp_dir_obj.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
