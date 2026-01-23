from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class OliveYoungBestItem:
    rank: Optional[int]
    goods_no: Optional[str]
    disp_cat_no: Optional[str]
    brand: Optional[str]
    name: Optional[str]
    url: Optional[str]
    image_url: Optional[str]
    org_price: Optional[int]
    cur_price: Optional[int]
    flags: list[str]


def parse_int_kr_price(text: str) -> Optional[int]:
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def extract_goods_no_from_url(url: str) -> tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    qs = parse_qs(urlparse(url).query)
    goods_no = (qs.get("goodsNo") or [None])[0]
    disp_cat_no = (qs.get("dispCatNo") or [None])[0]
    return goods_no, disp_cat_no


def parse_oliveyoung_best_html(
    html: str,
    base_url: str = "https://www.oliveyoung.co.kr",
) -> list[OliveYoungBestItem]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    product_lis = soup.select("ul.cate_prd_list > li")
    items: list[OliveYoungBestItem] = []

    for li in product_lis:
        prd_info = li.select_one("div.prd_info")
        if not prd_info:
            continue

        rank_text = None
        rank_el = prd_info.select_one("a.prd_thumb span.thumb_flag.best")
        if rank_el:
            rank_text = rank_el.get_text(strip=True)
        rank = parse_int_kr_price(rank_text or "")

        thumb_a = prd_info.select_one("a.prd_thumb[href]")
        url = urljoin(base_url, thumb_a["href"]) if thumb_a and thumb_a.get("href") else None

        img_el = prd_info.select_one("a.prd_thumb img")
        image_url = img_el["src"].strip() if img_el and img_el.get("src") else None

        brand_el = prd_info.select_one("div.prd_name span.tx_brand")
        name_el = prd_info.select_one("div.prd_name p.tx_name")
        brand = brand_el.get_text(strip=True) if brand_el else None
        name = name_el.get_text(strip=True) if name_el else None

        goods_no, disp_cat_no = extract_goods_no_from_url(url or "")

        org_price = None
        cur_price = None
        org_num = prd_info.select_one("p.prd_price span.tx_org span.tx_num")
        cur_num = prd_info.select_one("p.prd_price span.tx_cur span.tx_num")

        if org_num:
            org_price = parse_int_kr_price(org_num.get_text(strip=True))
        if cur_num:
            cur_price = parse_int_kr_price(cur_num.get_text(strip=True))
        if cur_price is None:
            first_num = prd_info.select_one("p.prd_price span.tx_num")
            cur_price = parse_int_kr_price(first_num.get_text(strip=True) if first_num else "")

        flags = []
        for flag in prd_info.select("p.prd_flag span.icon_flag"):
            t = flag.get_text(strip=True)
            if t:
                flags.append(t)

        cart_btn = prd_info.select_one("button.cartBtn")
        if cart_btn:
            goods_no = cart_btn.get("data-ref-goodsNo") or goods_no
            disp_cat_no = cart_btn.get("data-ref-dispCatNo") or disp_cat_no
            brand = cart_btn.get("data-ref-goodsbrand") or brand
            name = cart_btn.get("data-ref-goodsnm") or name

        if not (name and (url or goods_no)):
            continue

        items.append(
            OliveYoungBestItem(
                rank=rank,
                goods_no=goods_no,
                disp_cat_no=disp_cat_no,
                brand=brand,
                name=name,
                url=url,
                image_url=image_url,
                org_price=org_price,
                cur_price=cur_price,
                flags=flags,
            )
        )

    dedup: dict[str, OliveYoungBestItem] = {}
    for it in items:
        key = it.goods_no or it.url or f"{it.brand}|{it.name}"
        dedup[key] = it

    return sorted(
        dedup.values(),
        key=lambda x: (x.rank is None, x.rank if x.rank is not None else 10**9),
    )
