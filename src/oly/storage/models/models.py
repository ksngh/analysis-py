from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RankingItem(Base):
    __tablename__ = "ranking_items"

    id = Column(Integer, primary_key=True, index=True)
    rank = Column(Integer, nullable=True, index=True)
    goods_no = Column(String(50), nullable=True, index=True)
    disp_cat_no = Column(String(50), nullable=True, index=True)
    brand = Column(String(200), nullable=True)
    name = Column(String(300), nullable=False)
    url = Column(String(500), nullable=True, index=True)
    image_url = Column(String(500), nullable=True)
    org_price = Column(Integer, nullable=True)
    cur_price = Column(Integer, nullable=True)
    flags = Column(Text, nullable=True)
    captured_at = Column(DateTime, nullable=False, index=True)
