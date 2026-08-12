"""Loads vendor_risk.csv and vendor_documents.json into the database.

Run with:  python -m app.seed
"""
import csv
import json

from .db import Base, engine, SessionLocal
from .models import VendorRisk, VendorDocument
from .config import DATA_DIR


def load_vendor_risk(db):
    if db.query(VendorRisk).count() == 0:
        with open(DATA_DIR / "vendor_risk.csv", newline="") as f:
            for row in csv.DictReader(f):
                db.add(VendorRisk(
                    vendor_id=row["vendor_id"],
                    vendor_name=row["vendor_name"],
                    product=row["product"],
                    status=row["status"],
                    risk_rating=row["risk_rating"],
                    assessment_date=row["assessment_date"],
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    authority_tier=int(row["authority_tier"]),
                ))
        db.commit()


def load_vendor_documents(db):
    if db.query(VendorDocument).count() == 0:
        with open(DATA_DIR / "vendor_documents.json") as f:
            docs = json.load(f)
        for d in docs:
            db.add(VendorDocument(
                document_id=d["document_id"],
                vendor_name=d["vendor_name"],
                product=d["product"],
                source_type=d["source_type"],
                authority_tier=d["authority_tier"],
                document_date=d["document_date"],
                result=d["result"],
                risk_rating=d["risk_rating"],
                content=d["content"],
            ))
        db.commit()


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        load_vendor_risk(db)
        load_vendor_documents(db)
    finally:
        db.close()


if __name__ == "__main__":
    seed()
    print("Seed complete.")
