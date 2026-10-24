from models.schemas import UserInput

def collect_user_input() -> UserInput:
    print("=== SEO Blog Agent ===\n")
    company = input("Company name: ").strip()
    website = input("Company website (optional): ").strip() or None
    keyword = input("Target keyword (optional, press Enter to auto-suggest): ").strip() or None
    topic = input("Blog topic: ").strip()
    audience = input("Target audience: ").strip()
    tone = input("Tone [professional but friendly]: ").strip() or "professional but friendly"
    wc = input("Word count [1200]: ").strip()
    word_count = int(wc) if wc.isdigit() else 1200

    return UserInput(
        company_name=company,
        website=website,
        target_keyword=keyword,
        topic=topic,
        audience=audience,
        tone=tone,
        word_count=word_count,
    )