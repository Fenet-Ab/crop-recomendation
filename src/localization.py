"""Polite, single-language SMS for farmers (Amharic, Afan Oromo, or English)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

CROP_NAMES = {
    "Wheat": {"am": "ስንዴ", "om": "qamadii", "en": "wheat"},
    "Maize": {"am": "በቆሎ", "om": "boqqolloo", "en": "maize"},
    "Teff": {"am": "ጤፍ", "om": "xaafii", "en": "teff"},
    "Barley": {"am": "ገብስ", "om": "garbuu", "en": "barley"},
    "Sorghum": {"am": "ማሽላ", "om": "mishingaa", "en": "sorghum"},
    "Dagussa": {"am": "ዳጉሳ", "om": "daaguusaa", "en": "finger millet"},
    "Bean": {"am": "ባቄላ", "om": "baaqelaa", "en": "bean"},
    "Pea": {"am": "አተር", "om": "atara", "en": "pea"},
    "Niger seed": {"am": "ኑግ", "om": "nuugii", "en": "niger seed"},
    "Potato": {"am": "ድንች", "om": "dinnicha", "en": "potato"},
    "Red Pepper": {"am": "ቀይ በርበሬ", "om": "barbaree diimaa", "en": "red pepper"},
    "Fallow": {"am": "እረፍት መሬት", "om": "lafa boqonnaa", "en": "fallow land"},
}

GROUP_NAMES = {
    "Major_Cereals": {"am": "ዋና እህሎች", "om": "midhaan guddaa", "en": "major cereals"},
    "Cereals": {"am": "ሌሎች እህሎች", "om": "midhaan xixiqqoo", "en": "other cereals"},
    "Pulses": {"am": "ጥራጥሬዎች", "om": "daalattiwwan", "en": "pulses"},
    "Specialty": {"am": "ልዩ ሰብሎች", "om": "midhaan addaa", "en": "specialty crops"},
}

STAGE_RULES = [
    (("basal", "sowing", "planting", "transplant"), {
        "am": "በመዝራት ጊዜ",
        "om": "yeroo facaasuutti",
        "en": "at planting",
    }),
    (("tillering",), {
        "am": "ቁጥር ሲበዛ",
        "om": "yeroo biqiltuun baay'atu",
        "en": "when tillers appear",
    }),
    (("stem", "elongation"), {
        "am": "ግንዱ ሲረዝም",
        "om": "yeroo jigi dheeratu",
        "en": "when the stem grows tall",
    }),
    (("knee", "v6"), {
        "am": "ተክሉ የጉልበት ቁመት ሲደርስ",
        "om": "yeroo biqiltuun hanga jilbaatti guddatu",
        "en": "when the plant is knee-high",
    }),
    (("tassel", "v12"), {
        "am": "አበባ/ጆሮ ሲወጣ",
        "om": "yeroo daraaraan bahu",
        "en": "when flowers appear",
    }),
    (("panicle",), {
        "am": "ጆሮ ሲጀምር",
        "om": "yeroo mataan midhaanii jalqabu",
        "en": "when the head starts to form",
    }),
    (("branch",), {
        "am": "ቅርንጫፍ ሲወጣ",
        "om": "yeroo dameen bahu",
        "en": "when branches appear",
    }),
    (("stolon",), {
        "am": "ሥር ግንድ ሲዘረጋ",
        "om": "yeroo hiddi diriiratu",
        "en": "when runners start",
    }),
    (("tuber", "bulking"), {
        "am": "ድንቹ ሲያብጥ",
        "om": "yeroo dinnichi guddatu",
        "en": "when tubers fill",
    }),
    (("flower",), {
        "am": "በአበባ ወቅት",
        "om": "yeroo daraaraa",
        "en": "at flowering",
    }),
    (("fruit",), {
        "am": "ፍሬው ሲያድግ",
        "om": "yeroo firii guddatu",
        "en": "when fruit fills",
    }),
    (("starter",), {
        "am": "በመጀመሪያ መዝራት",
        "om": "yeroo facaasa jalqabaa",
        "en": "at first planting",
    }),
]

TRANSLATIONS = {
    "am": {
        "greeting": "ውድ {name}፣ ሰላም ይሁንልዎ።",
        "crop_rec": "ለእርሻዎ የሚመከረው ሰብል {crop} ነው።",
        "crop_rec_group": "ለእርሻዎ የሚመከረው ሰብል {crop} ነው (ከ{group} ቤተሰብ)።",
        "confidence": "ይህ ምክር በመረጃው መሠረት በ{confidence} በመቶ እምነት ተሰጥቷል።",
        "dap": "እባክዎ ዲኤፒ ማዳበሪያ {dap} ኪሎ ግራም በሄክታር ይጠቀሙ። ይህም ወደ {dap_bags} ቦርሳ ይጠጋል (አንዱ ቦርሳ 50 ኪሎ)።",
        "urea": "ዩሪያ ማዳበሪያ {urea} ኪሎ ግራም በሄክታር ይጠቀሙ። ይህም ወደ {urea_bags} ቦርሳ ይጠጋል።",
        "urea_none": "ለዚህ ሰብል ዩሪያ አያስፈልግም።",
        "split_dap_only": "{stage} ሁሉንም ዲኤፒ ማዳበሪያ በአፈሩ ውስጥ ያስገቡ።",
        "nutrients": "ይህ {n_kg} ኪሎ ናይትሮጅንና {p_kg} ኪሎ ፎስፈረስ ይሰጣል።",
        "how_to_use": "እባክዎ እንዲህ ይከፋፍሉ፦",
        "split_basal": "{stage} ሁሉንም ዲኤፒ እና {urea} ኪሎ ዩሪያ ያስገቡ (ከመቶ {pct} ክፍል)።",
        "split_top": "{stage} ተጨማሪ {urea} ኪሎ ዩሪያ ይስጡ (ከመቶ {pct} ክፍል)።",
        "legume_note": "ባቄላና አተር ናይትሮጅን ከአየር ስለሚያገኙ ዩሪያ አያስፈልግም ወይም በጣም ጥቂት ነው። ዋናው ማዳበሪያ ዲኤፒ ነው።",
        "fallow_note": "ይህ መሬት እረፍት ስለሆነ አሁን ማዳበሪያ አይስጡ። ለሚቀጥለው ወቅት ያዘጋጁት።",
        "thanks": "እባክዎ በጥንቃቄ ይጠቀሙ። መልካም ምርት እንመኛለን።",
        "footer": "ከግብርና ምክር አገልግሎት።",
    },
    "om": {
        "greeting": "Kabajamaa {name}, nagaa fi fayyaa isinii hawwina.",
        "crop_rec": "Qonna keessan irratti midhaan si'aawaa ta'e {crop} dha.",
        "crop_rec_group": "Qonna keessan irratti midhaan si'aawaa ta'e {crop} dha (maatii {group}).",
        "confidence": "Gorsi kun odeeffannoo qorannoo irratti hundaa'ee amantaa dhibbeentaa {confidence} qaba.",
        "dap": "Maaloo xa'oo Di'aapii kiiloo {dap} hektaraaf fayyadamaa. Kunis korojoo {dap_bags} dha (korojoon tokko kiiloo 50).",
        "urea": "Xa'oo Yuuriyaa kiiloo {urea} hektaraaf fayyadamaa. Kunis korojoo {urea_bags} dha.",
        "urea_none": "Midhaan kanaaf Yuuriyaan hin barbaachisu.",
        "split_dap_only": "{stage} xa'oo Di'aapii hunda lafa keessa galchaa.",
        "nutrients": "Kun naayitroojiinii kiiloo {n_kg} fi foosfoorasii kiiloo {p_kg} kenna.",
        "how_to_use": "Maaloo akkanaatti hir'isaa:",
        "split_basal": "{stage} Di'aapii hunda fi Yuuriyaa kiiloo {urea} kennaa (dhibbeentaa {pct}).",
        "split_top": "{stage} Yuuriyaa dabalataa kiiloo {urea} kennaa (dhibbeentaa {pct}).",
        "legume_note": "Baaqelaan fi ataraan naayitroojiinii qilleensa irraa waan fudhatanif Yuuriyaan baay'ee hin barbaachisu. Xa'oon ijoon Di'aapii dha.",
        "fallow_note": "Lafti kun boqonnaa waan ta'eef amma xa'oo hin kenninaa. Gocha itti aanuuf qopheessaa.",
        "thanks": "Maaloo of eeggannoodhaan fayyadamaa. Omisha gaarii isinii hawwina.",
        "footer": "Tajaajila gorsaa qonnaa irraa.",
    },
    "en": {
        "greeting": "Dear {name}, greetings and good health.",
        "crop_rec": "The crop we recommend for your field is {crop}.",
        "crop_rec_group": "The crop we recommend for your field is {crop} (from the {group} family).",
        "confidence": "This advice is given with {confidence} percent confidence from the farm data.",
        "dap": "Please apply {dap} kilograms of di-ammonium phosphate per hectare, about {dap_bags} bags (each bag is 50 kilograms).",
        "urea": "Please apply {urea} kilograms of urea per hectare, about {urea_bags} bags.",
        "urea_none": "This crop does not need urea.",
        "split_dap_only": "{stage}, put all the phosphate fertilizer into the soil.",
        "nutrients": "This supplies {n_kg} kilograms of nitrogen and {p_kg} kilograms of phosphorus.",
        "how_to_use": "Please split it like this:",
        "split_basal": "{stage}, apply all the phosphate fertilizer and {urea} kilograms of urea ({pct} percent).",
        "split_top": "{stage}, add {urea} kilograms of urea ({pct} percent).",
        "legume_note": "Beans and peas take nitrogen from the air, so urea is little or not needed. The main fertilizer is the phosphate bag.",
        "fallow_note": "This land is for rest, so do not apply fertilizer now. Prepare it for the next season.",
        "thanks": "Please use it carefully. We wish you a good harvest.",
        "footer": "From the farm advice service.",
    },
}


def normalize_lang(lang: str | None) -> str:
    value = (lang or "am").strip().lower()
    if value in ("or", "oro", "afaan oromo", "oromo"):
        return "om"
    if value in ("amh", "amharic", "አማርኛ"):
        return "am"
    if value in TRANSLATIONS:
        return value
    return "am"


def localize_crop(crop: str, lang: str) -> str:
    lang = normalize_lang(lang)
    entry = CROP_NAMES.get(crop) or CROP_NAMES.get((crop or "").title())
    if entry:
        return entry.get(lang) or entry["en"]
    return crop


def localize_group(group: str, lang: str) -> str:
    lang = normalize_lang(lang)
    key = (group or "").replace(" ", "_")
    entry = GROUP_NAMES.get(group) or GROUP_NAMES.get(key)
    if entry:
        return entry.get(lang) or entry["en"]
    return (group or "").replace("_", " ")


def localize_stage(stage: str, lang: str) -> str:
    lang = normalize_lang(lang)
    text = (stage or "").lower()
    for keys, names in STAGE_RULES:
        if any(k in text for k in keys):
            return names.get(lang) or names["en"]
    if lang == "am":
        return "በዚህ ዕድገት ወቅት"
    if lang == "om":
        return "yeroo guddina kanaatti"
    return "at this growth stage"


def translate(lang: str, key: str, **kwargs) -> str:
    lang = normalize_lang(lang)
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["am"])
    template = lang_dict.get(key) or TRANSLATIONS["am"].get(key, "")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def bags_from_kg(kg: float) -> float:
    return round(float(kg or 0) / 50.0, 1)


def _sentence_stage(stage: str, lang: str) -> str:
    if lang == "am" or not stage:
        return stage
    return stage[:1].upper() + stage[1:]


def _format_split_lines(lang: str, splits: Iterable[dict], dap_kg: float) -> list[str]:
    lines = [translate(lang, "how_to_use")]
    first = True
    for item in splits:
        stage = _sentence_stage(localize_stage(str(item.get("stage") or ""), lang), lang)
        urea = round(float(item.get("urea_kg_ha") or 0), 1)
        pct = int(round(float(item.get("percentage") or 0)))
        if first and dap_kg > 0 and urea <= 0:
            lines.append(translate(lang, "split_dap_only", stage=stage))
        elif first and dap_kg > 0:
            lines.append(translate(lang, "split_basal", stage=stage, urea=urea, pct=pct))
        elif urea > 0:
            lines.append(translate(lang, "split_top", stage=stage, urea=urea, pct=pct))
        first = False
    return lines


def build_sms_message(
    lang: str,
    name: str,
    crop: str,
    dap_kg: float,
    urea_kg: float,
    crop_group: str | None = None,
    splits: Optional[list[dict[str, Any]]] = None,
    is_legume: bool = False,
    confidence_pct: Optional[float] = None,
    n_kg: Optional[float] = None,
    p_kg: Optional[float] = None,
) -> str:
    """Build a polite, fully monolingual SMS from live crop and fertilizer results."""
    lang = normalize_lang(lang)
    crop_local = localize_crop(crop, lang)
    group_local = localize_group(crop_group, lang) if crop_group else ""
    dap = round(float(dap_kg or 0), 1)
    urea = round(float(urea_kg or 0), 1)

    lines = [translate(lang, "greeting", name=name, crop=crop_local)]
    if group_local:
        lines.append(translate(lang, "crop_rec_group", crop=crop_local, group=group_local))
    else:
        lines.append(translate(lang, "crop_rec", crop=crop_local))
    if confidence_pct:
        lines.append(translate(lang, "confidence", confidence=round(float(confidence_pct), 1)))

    if (crop or "").lower() == "fallow" or (dap <= 0 and urea <= 0):
        lines.append(translate(lang, "fallow_note"))
        lines.append(translate(lang, "thanks"))
        lines.append(translate(lang, "footer"))
        return "\n".join(lines)

    if dap > 0:
        lines.append(translate(lang, "dap", dap=dap, dap_bags=bags_from_kg(dap)))
    if urea > 0:
        lines.append(translate(lang, "urea", urea=urea, urea_bags=bags_from_kg(urea)))
    else:
        lines.append(translate(lang, "urea_none"))
    if n_kg is not None and p_kg is not None:
        lines.append(translate(lang, "nutrients", n_kg=round(float(n_kg), 1), p_kg=round(float(p_kg), 1)))

    if is_legume:
        lines.append(translate(lang, "legume_note"))
    if splits:
        lines.extend(_format_split_lines(lang, splits, dap))
    else:
        lines.append(
            translate(
                lang,
                "split_basal",
                stage=localize_stage("sowing", lang),
                urea=round(urea * 0.5, 1),
                pct=50,
            )
        )

    lines.append(translate(lang, "thanks"))
    lines.append(translate(lang, "footer"))
    return "\n".join(line for line in lines if line).strip()
