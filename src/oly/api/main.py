from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from typing import List, Optional
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from curl_cffi import requests
from oly.storage.db.database import SessionLocal, engine
from oly.storage.models.models import Base, RankingItem
from oly.ingestion.parsers.parser import parse_oliveyoung_best_html

RANKING_URL = "https://www.oliveyoung.co.kr/store/main/getBestList.do"
_stop_event = threading.Event()
_last_sync_at: Optional[datetime] = None
_last_attempt_at: Optional[datetime] = None
_last_error: Optional[str] = None


class RankingItemOut(BaseModel):
    rank: Optional[int]
    goods_no: Optional[str]
    disp_cat_no: Optional[str]
    brand: Optional[str]
    name: str
    url: Optional[str]
    image_url: Optional[str]
    org_price: Optional[int]
    cur_price: Optional[int]
    flags: List[str]
    captured_at: datetime


class SyncResult(BaseModel):
    count: int
    inserted: int
    updated: int


class SchedulerStatus(BaseModel):
    last_sync_at: Optional[datetime]
    last_attempt_at: Optional[datetime]
    last_error: Optional[str]
    next_sync_at: datetime


def _fetch_html() -> str:
    resp = requests.get(
        RANKING_URL,
        impersonate="chrome110",
        headers={
            "referer": "https://www.google.com/",
            "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    return resp.text


def sync_best_items() -> SyncResult:
    global _last_sync_at, _last_error
    html = _fetch_html()
    items = parse_oliveyoung_best_html(html)
    inserted = 0
    updated = 0
    captured_at = datetime.utcnow()

    with SessionLocal() as db:
        for it in items:
            existing = None
            if it.goods_no:
                existing = (
                    db.query(RankingItem)
                    .filter(RankingItem.goods_no == it.goods_no)
                    .one_or_none()
                )
            elif it.url:
                existing = (
                    db.query(RankingItem)
                    .filter(RankingItem.url == it.url)
                    .one_or_none()
                )

            flags_text = ",".join(it.flags) if it.flags else None

            if existing:
                existing.rank = it.rank
                existing.disp_cat_no = it.disp_cat_no
                existing.brand = it.brand
                existing.name = it.name or existing.name
                existing.url = it.url
                existing.image_url = it.image_url
                existing.org_price = it.org_price
                existing.cur_price = it.cur_price
                existing.flags = flags_text
                existing.captured_at = captured_at
                updated += 1
            else:
                db.add(
                    RankingItem(
                        rank=it.rank,
                        goods_no=it.goods_no,
                        disp_cat_no=it.disp_cat_no,
                        brand=it.brand,
                        name=it.name or "",
                        url=it.url,
                        image_url=it.image_url,
                        org_price=it.org_price,
                        cur_price=it.cur_price,
                        flags=flags_text,
                        captured_at=captured_at,
                    )
                )
                inserted += 1
        db.commit()

    _last_sync_at = captured_at
    _last_error = None
    return SyncResult(count=len(items), inserted=inserted, updated=updated)


def _seconds_until_next_hour() -> float:
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return max((next_hour - now).total_seconds(), 0.0)


def _scheduler_loop() -> None:
    global _last_attempt_at, _last_error
    while not _stop_event.is_set():
        _stop_event.wait(_seconds_until_next_hour())
        if _stop_event.is_set():
            break
        try:
            _last_attempt_at = datetime.utcnow()
            sync_best_items()
        except Exception:
            _last_error = traceback.format_exc().strip()

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    _stop_event.clear()
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        _stop_event.set()


app = FastAPI(lifespan=lifespan)

@app.post("/best-sync", response_model=SyncResult)
def best_sync():
    return sync_best_items()


@app.get("/best", response_model=List[RankingItemOut])
def best():
    with SessionLocal() as db:
        rows = (
            db.query(RankingItem)
            .order_by(RankingItem.rank.is_(None), RankingItem.rank.asc())
            .all()
        )

    return [
        RankingItemOut(
            rank=r.rank,
            goods_no=r.goods_no,
            disp_cat_no=r.disp_cat_no,
            brand=r.brand,
            name=r.name,
            url=r.url,
            image_url=r.image_url,
            org_price=r.org_price,
            cur_price=r.cur_price,
            flags=r.flags.split(",") if r.flags else [],
            captured_at=r.captured_at,
        )
        for r in rows
    ]


@app.get("/scheduler-status", response_model=SchedulerStatus)
def scheduler_status():
    next_sync_at = datetime.now() + timedelta(seconds=_seconds_until_next_hour())
    return SchedulerStatus(
        last_sync_at=_last_sync_at,
        last_attempt_at=_last_attempt_at,
        last_error=_last_error,
        next_sync_at=next_sync_at,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
