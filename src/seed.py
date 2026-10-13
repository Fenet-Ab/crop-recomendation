"""Create farmer tables and insert demo profiles if the table is empty."""

from src.database import Base, get_engine, get_session
from src.models_db import Farmer
from src.sms_service import normalize_msisdn


def seed_data():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    db = get_session()
    try:
        if db.query(Farmer.id).first():
            print("Farmers table already has rows — skip seed.")
            return

        test_farmers = [
            Farmer(
                full_name="Abebe Kebede",
                phone_number=normalize_msisdn("251953392118"),
                region="Oromia",
                language_pref="om",
            ),
            Farmer(
                full_name="Lemma Merga",
                phone_number=normalize_msisdn("251940790416"),
                region="Amhara",
                language_pref="am",
            ),
        ]
        db.add_all(test_farmers)
        db.commit()
        print("Seeded test farmers into Supabase.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
