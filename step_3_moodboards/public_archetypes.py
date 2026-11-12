"""Customer-facing archetype copy and deterministic mapping for taste profiles.

Technical category clusters remain unchanged.  Clothes clusters that Ellina's
review split are resolved per brand with reviewed overrides plus a documented
dominant default for the rest of the cluster.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ARCHETYPES = (
    (1, "The Expedition", "Outdoor, technical, adventure", "You like knowing your clothes are built for anything. Technical, functional and ready for the elements, even if the furthest your jacket has made it so far is the local coffee shop. The wilderness can wait. You’ll be ready when it calls."),
    (2, "The Understated", "Quiet luxury, heritage minimalism", "You like the label on the inside. Quiet luxury at its best. No logo needed to make an impact. You’re building a timeless wardrobe slowly, intentionally, and it shows."),
    (3, "The Disruptor", "Avant-garde, conceptual, challenging", "You like clothes that start a conversation you didn’t ask for. A plain t-shirt and trousers? Never. For you, getting dressed is a statement: sometimes art, always intentional."),
    (4, "The Dreamer", "Romantic, whimsical, feminine softness", "You like floaty sleeves, soft prints, and the kind of dress that makes you feel like you’re in a film. When in doubt, a lace detail, a cute bag, a ribbon or a pearl will always save the day."),
    (5, "The Underground", "Street, skate, subculture", "Your clothes are how you express yourself. You liked the brands before they were cool, and even if the underground lifestyle is more of a memory now, the attitude never left your wardrobe."),
    (6, "Modern Minimalist", "Considered, edited, contemporary clean", "You want everything to go with everything. Clean silhouettes and simple colours make it easy to dress for any occasion, from the office to dinner to a night out with barely a second thought."),
    (7, "After Midnight", "Dark, directional, sensual, conceptual", "Dark, conceptual and sensual could all be you. You’re drawn to intriguing silhouettes, dark palettes, and the kind of outfit that makes people look twice without knowing why."),
    (8, "The Trove", "Maximalist, bold print, eclectic collector", "More is more in your wardrobe. Colours, patterns and textures: the whole lot. Your wardrobe looks like a very good vintage market, and that’s entirely intentional."),
    (9, "Sun Bleached", "Coastal slow, linen, washed ease", "You like linen that wrinkles, mornings that stretch, and clothes that feel like being on holiday even when you’re not."),
    (10, "The Workshop", "Heritage craft, workwear, selvedge", "You like things that are built to last. You’ve never thrown away a good leather jacket and you never will."),
    (11, "The Dresser", "Polished occasion, smart-dressed", "You like getting dressed up on purpose. Any occasion is the right occasion if you dress for it."),
    (12, "Endorphin", "Active, sport, movement, wellness", "You like looking good at 6am. Gym kit is a whole category in your wardrobe and you’re not sorry."),
    (13, "The Wanderer", "Global artisan, bohemian, travel craft", "You like clothes that come with a story. Handmade, artisanal and rooted in craft and culture from somewhere far away. Your wardrobe is a collection of places, not just pieces."),
    (14, "The Alumni", "Classic casual, prep, wardrobe uniform", "You treat your outfits like a uniform. Work, uni, a quick padel game and coffee with friends are all perfect occasions for your signature look."),
    (15, "Riviera", "Resort, coastal luxury, Mediterranean", "You like being somewhere warm, or at least dressing like you are. Light materials, colour, prints, and a glass of something cold always nearby."),
)

ARCHETYPE_BY_NAME = {
    name: {
        "public_cluster_id": f"archetype-{number:02d}",
        "public_cluster_number": number,
        "public_cluster_name": name,
        "public_cluster_core_aesthetic": core,
        "public_cluster_description": description,
    }
    for number, name, core, description in ARCHETYPES
}

# Dominant customer-facing label for every current numeric category cluster.
# These are display mappings only; the numeric cluster assignments are stable.
CLUSTER_PUBLIC_NAMES = {
    "bags": ("The Workshop", "The Underground", "The Understated", "The Expedition", "The Disruptor", "Modern Minimalist", "The Expedition", "The Wanderer", "Sun Bleached", "The Dresser", "Riviera", "The Trove", "The Dresser", "Modern Minimalist", "The Dresser", "The Wanderer", "Modern Minimalist"),
    "clothes": ("Sun Bleached", "The Dreamer", "The Dresser", "The Workshop", "Riviera", "The Dresser", "Modern Minimalist", "Modern Minimalist", "Modern Minimalist", "The Expedition", "The Wanderer", "The Alumni", "The Disruptor", "The Underground", "The Underground", "The Expedition", "Modern Minimalist"),
    "jewellery": ("The Understated", "The Understated", "The Underground", "The Dresser", "The Wanderer", "The Wanderer", "The Understated", "The Alumni", "The Expedition", "The Trove", "The Disruptor", "Modern Minimalist", "The Dresser", "Riviera", "The Dresser", "The Trove"),
    "shoes": ("Modern Minimalist", "The Dresser", "The Dresser", "The Dreamer", "The Underground", "The Underground", "The Alumni", "Endorphin", "The Understated", "Modern Minimalist", "The Disruptor", "The Expedition", "Sun Bleached", "The Expedition", "The Workshop", "The Dresser", "The Dresser", "Riviera", "The Dreamer", "The Alumni", "The Alumni", "The Dresser"),
    "swimwear_lingerie": ("The Wanderer", "After Midnight", "Riviera", "Modern Minimalist", "Riviera"),
}


def normalise(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.replace("&", " and ").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _names(*values: str) -> set[str]:
    return {normalise(value) for value in values}


# Brand-level decisions for the mixed live clothes clusters.  Brands not listed
# inherit the documented dominant label in CLUSTER_PUBLIC_NAMES.
CATEGORY_PUBLIC_OVERRIDES = {
    "clothes": {
    0: {
        "Endorphin": _names("ALO YOGA", "Beyond Yoga", "Gymshark", "NAGNATA", "Sweaty Betty", "Varley", "Vuori"),
    },
    1: {
        "The Trove": _names("Anna Sui", "DVF", "Diane von Furstenberg", "Farm Rio", "Hofmann Copenhagen", "La Double J", "Lisou", "Mary Katrantzou", "MSGM", "Pucci", "Rich Mnisi", "Stine Goya", "Tongoro", "Traffic People", "Trina Turk"),
    },
    2: {
        "The Understated": _names("Bally", "Celine", "Chanel", "Hermès", "Hermes", "Max Mara", "Nanushka", "Nour Hammour", "Saint Laurent", "Tod's", "Tods"),
    },
    5: {
        "The Trove": _names("Borgo De Nor", "Clio Peppiatt", "Cult Gaia", "DAVID KOMA", "DIMA AYAD", "Mirror Palais"),
        "The Dreamer": _names("Cecilie Bahnsen", "LoveShackFancy", "Simone Rocha"),
    },
    7: {
        "The Dresser": _names("Balmain", "Favourbrook", "Karl Lagerfeld", "Polo Ralph Lauren", "Thom Sweeney"),
    },
    8: {
        "The Dreamer": _names("DISSH", "Djerf Avenue", "ROTATE", "ROTATE Birger Christensen", "Jonathan Simkhai", "Significant Other", "Sessun", "TWIST & TANGO", "Twist & Tango"),
    },
    10: {
        "Sun Bleached": _names("Ciao Lucia", "Daydreamer LA", "The Handloom", "Handloom", "Los Angeles Allow"),
    },
    13: {
        "Endorphin": _names("ADANOLA", "Aim'n", "Bandit Running", "Castore", "Girlfriend Collective", "Gymshark", "Lululemon", "Nike", "Oner Active", "P.E Nation", "Sweaty Betty"),
        "The Expedition": _names("Aztech Mountain", "Burton", "Fusalp", "K WAY", "K-Way", "Perfect Moment"),
        "The Alumni": _names("Fred Perry", "Gant", "Lacoste"),
    },
    15: {
        "After Midnight": _names("A-COLD-WALL*", "A Cold Wall", "Han Kjøbenhavn", "Han Kjobenhavn", "Helmut Lang", "Mackintosh", "Nanamica", "OAMC", "SOIA & KYO", "Wardrobe NYC"),
        "Modern Minimalist": _names("Arket", "Aspesi", "Kloke", "Monique van Heist", "Norse Projects", "Studio Nicholson"),
        "The Workshop": _names("Alex Mill", "Arpenteur", "Ben Davis", "Dickies", "Kestin", "Knickerbocker NYC", "Private White V.C.", "Private White Vc", "Taylor Stitch", "Universal Works"),
    },
    16: {
        "Sun Bleached": _names("Aiayu", "Alex Crane", "Bite Studios", "Black Crane", "HHPLIFT", "Knowledge Cotton", "knowledge cotton apparel", "Mate the Label", "Ninety Percent", "Nobody's Child", "Outerknown", "Tentree", "Thought Clothing", "Tradlands", "World of Crow"),
        "The Wanderer": _names("Caralarga", "Doodlage", "Story Mfg"),
    },
    },
    "bags": {
        1: {
            "After Midnight": _names("Ann Demeulemeester", "Chrome Hearts", "Rick Owens"),
            "The Disruptor": _names("Coperni"),
            "The Workshop": _names("Belstaff"),
            "The Alumni": _names("Maison Kitsuné", "NOAH"),
            "Modern Minimalist": _names("Acne Studios"),
        },
        2: {
            "Modern Minimalist": _names("Courreges", "Reiss"),
            "The Workshop": _names("Livid"),
            "The Expedition": _names("Stutterheim"),
        },
        3: {"The Disruptor": _names("zero+maria cornejo"), "Sun Bleached": _names("HHPLIFT")},
        4: {"The Dreamer": _names("Maradji"), "The Trove": _names("Messina Hembry")},
        5: {
            "The Understated": _names("Cuyana", "DeMellier", "Harber London", "Mansur Gavriel", "Smythson", "Strathberry", "The Row", "Valextra"),
            "The Trove": _names("Okhtein"),
        },
        6: {
            "Endorphin": _names("Kappa", "Karhu", "LOCI", "MALBON", "Mizuno", "Northwave", "So iLL", "WAHTS"),
            "The Underground": _names("Filling Pieces", "Kith"),
            "The Disruptor": _names("Mihara Yasuhiro", "SACAI", "MOWALOLA"),
            "Modern Minimalist": _names("Naghedi"),
            "The Dresser": _names("DKNY", "The Fold"),
        },
        7: {"The Disruptor": _names("Junya Watanabe")},
        8: {
            "The Trove": _names("Anna Sui"),
            "The Understated": _names("Anonymous Ism", "Brunello Cucinelli", "Lemaire"),
            "The Wanderer": _names("Chan Luu", "Guadalupe Design", "Isabel Marant", "Ulla Johnson"),
            "The Disruptor": _names("Loewe"),
            "The Dresser": _names("Maje"),
        },
        9: {"The Trove": _names("Pucci"), "The Dreamer": _names("Simone Rocha")},
        11: {"The Dreamer": _names("Childs", "Chloé", "Chloe", "Patou", "Reformation", "Sandy Liang", "Sessun")},
        12: {
            "The Workshop": _names("Alex Mill", "American Trench", "Fortela", "Lucchese", "Tecovas"),
            "The Understated": _names("Bally", "Celine", "Hermès", "Hermes", "Max Mara", "Tod's", "Tods"),
            "The Underground": _names("Diesel", "Zadig & Voltaire"),
            "The Disruptor": _names("Balenciaga", "Marine Serre"),
            "The Trove": _names("Ganni", "Gucci", "Miu Miu", "Ted Baker"),
            "Modern Minimalist": _names("BY FAR", "Osoi", "Proenza Schouler"),
            "After Midnight": _names("Alexander McQueen", "McQueen"),
        },
        14: {
            "The Alumni": _names("J.Crew", "Thom Browne"),
            "The Trove": _names("Frances Valentine"),
            "The Dreamer": _names("Sezane", "Sézane", "Twinset", "Vanessa Bruno"),
        },
        15: {"Riviera": _names("Jacquemus", "Nicobar", "Pachute", "Pied Nu", "Summer Away", "Susmie's")},
    },
    "jewellery": {
        0: {"The Dreamer": _names("Simone Rocha", "Perle by Lola")},
        1: {"The Wanderer": _names("Aurelie Bidermann"), "The Underground": _names("Maria Tash"), "The Trove": _names("Yvonne Léon", "Yvonne Leon")},
        2: {"The Disruptor": _names("Ambush", "Ann Demeulemeester", "Maison Margiela", "MOWALOLA", "Panconesi", "Rick Owens", "SACAI", "Coperni")},
        4: {"The Trove": _names("Anna Sui", "Cult Gaia", "Frances Valentine"), "The Disruptor": _names("Chopova Lowena"), "The Understated": _names("Brunello Cucinelli"), "Modern Minimalist": _names("Agmes")},
        7: {"The Dreamer": _names("Lelet NY")},
        11: {"The Disruptor": _names("Completedworks", "Khiry"), "The Underground": _names("Martine Ali"), "The Trove": _names("Alexis Bittar", "Lizzie Fortunato")},
        14: {"After Midnight": _names("The Attico", "Miu Miu", "Joomi Lim", "Luv AJ"), "The Trove": _names("Pucci", "CASABLANCA")},
    },
    "shoes": {
        1: {"The Disruptor": _names("ABRA")},
        11: {"The Disruptor": _names("Chopova Lowena"), "The Dresser": _names("Gucci", "Carvela"), "The Underground": _names("Golden Goose")},
        12: {
            "The Understated": _names("Bottega Veneta", "Brunello Cucinelli"),
            "The Trove": _names("Anna Sui", "Cult Gaia", "Frances Valentine"),
            "The Wanderer": _names("Chan Luu", "Global Mamas", "Guadalupe Design", "Okhai"),
            "The Disruptor": _names("Maison Margiela"),
            "The Dresser": _names("Maje"),
        },
        13: {
            "Endorphin": _names("Adidas", "Air Jordan", "Kappa"),
            "The Underground": _names("Kith"),
            "The Disruptor": _names("Marine Serre", "MOWALOLA", "SACAI", "zero+maria cornejo"),
            "The Dresser": _names("DKNY", "The Fold"),
        },
        16: {"The Underground": _names("Filling Pieces"), "The Dreamer": _names("Drogheria Crivellini")},
    },
    "swimwear_lingerie": {
        0: {"Riviera": _names("Adriana Degreas")},
        1: {
            "The Dreamer": _names("Carine Gilson", "Davi", "For Love & Lemons", "Le Petit Trou", "Only Hearts", "Wild Lovers"),
            "The Dresser": _names("Intimissimi", "Natori", "Victoria's Secret"),
        },
        3: {"The Dreamer": _names("Cou Cou Intimates", "Fruity Booty", "Hanky Panky"), "Endorphin": _names("Saxx")},
        4: {"Modern Minimalist": _names("Baserange", "SKIMS", "Curvy Kate"), "The Dreamer": _names("Boux Avenue"), "Endorphin": _names("Left On Friday")},
    },
}

# Backwards-compatible alias used by existing tests and review tools.
CLOTHES_PUBLIC_OVERRIDES = CATEGORY_PUBLIC_OVERRIDES["clothes"]

GLOBAL_PUBLIC_OVERRIDES = {
    ("clothes", normalise("The Attico")): "After Midnight",
    ("bags", normalise("The Attico")): "After Midnight",
    ("bags", normalise("Alexander Wang")): "After Midnight",
    ("bags", normalise("Montce")): "Riviera",
    ("shoes", normalise("Comme des Garçons")): "The Disruptor",
    ("jewellery", normalise("Comme des Garçons")): "The Disruptor",
    ("bags", normalise("ALO YOGA")): "Endorphin",
    ("jewellery", normalise("ALO YOGA")): "Modern Minimalist",
    ("shoes", normalise("ALO YOGA")): "Endorphin",
    ("clothes", normalise("A Bathing Ape")): "The Underground",
    ("clothes", normalise("Mizuno")): "Endorphin",
    ("bags", normalise("MOWALOLA")): "The Disruptor",
    ("jewellery", normalise("Issey Miyake")): "The Disruptor",
    ("jewellery", normalise("Henrik Vibskov")): "The Disruptor",
    ("jewellery", normalise("Marni")): "The Disruptor",
    ("shoes", normalise("Fusalp")): "The Expedition",
    ("shoes", normalise("K WAY")): "The Expedition",
    ("shoes", normalise("K-Way")): "The Expedition",
    ("shoes", normalise("Herschel")): "The Expedition",
    ("swimwear_lingerie", normalise("Baserange")): "Modern Minimalist",
    ("swimwear_lingerie", normalise("SKIMS")): "Modern Minimalist",
    ("swimwear_lingerie", normalise("Von Dutch")): "The Underground",
    ("clothes", normalise("Duran Lantink")): "The Disruptor",
    ("clothes", normalise("Chrome Hearts")): "After Midnight",
    ("clothes", normalise("Tory Burch")): "The Alumni",
    ("clothes", normalise("Apparis")): "The Trove",
    ("clothes", normalise("Barneys Originals")): "The Workshop",
    ("clothes", normalise("Belstaff")): "The Workshop",
    ("clothes", normalise("SPANX")): "Endorphin",
    ("clothes", normalise("Taiga Takahashi")): "The Workshop",
    ("clothes", normalise("Cernucci")): "The Underground",
    ("clothes", normalise("Gravel & Gold")): "The Workshop",
    ("clothes", normalise("Alix Of Bohemia")): "The Wanderer",
    ("clothes", normalise("Bode")): "The Wanderer",
    ("clothes", normalise("Caron Callahan")): "The Workshop",
    ("clothes", normalise("Casey Casey")): "Modern Minimalist",
    ("clothes", normalise("Shrimps")): "The Dreamer",
    ("clothes", normalise("VISVIM")): "The Workshop",
    ("clothes", normalise("Anoeses")): "After Midnight",
    ("clothes", normalise("Another Aspect")): "Modern Minimalist",
    ("clothes", normalise("Desigual")): "The Trove",
    ("clothes", normalise("Loewe")): "The Disruptor",
    ("clothes", normalise("Miu Miu")): "The Alumni",
    ("clothes", normalise("Allison Wonderland")): "The Dreamer",
    ("clothes", normalise("Norma Kamali")): "After Midnight",
    ("clothes", normalise("Glassworks London")): "The Trove",
    ("clothes", normalise("ALIGNE")): "Modern Minimalist",
    ("clothes", normalise("DISSH")): "Modern Minimalist",
    ("clothes", normalise("Djerf Avenue")): "Modern Minimalist",
    ("clothes", normalise("Rat & Boa")): "After Midnight",
    ("clothes", normalise("Jonathan Simkhai")): "After Midnight",
    ("clothes", normalise("ROTATE Birger Christensen")): "After Midnight",
    ("clothes", normalise("Musier Paris")): "After Midnight",
    ("clothes", normalise("Faithfull The Brand")): "Riviera",
    ("clothes", normalise("Cara Cara")): "Riviera",
    ("clothes", normalise("Baum und Pferdgarten")): "The Trove",
    ("clothes", normalise("Ganni")): "The Trove",
    ("clothes", normalise("House Of Sunny")): "The Trove",
    ("clothes", normalise("Monki")): "The Trove",
    ("clothes", normalise("Munthe")): "The Trove",
    ("clothes", normalise("Olivia Rubin")): "The Trove",
    ("clothes", normalise("Palm Noosa")): "The Trove",
    ("clothes", normalise("Tanya Taylor")): "The Trove",
    ("clothes", normalise("Tyler McGillivary")): "The Trove",
    ("clothes", normalise("Alice + Olivia")): "The Dresser",
    ("clothes", normalise("Aniye By")): "The Dresser",
    ("clothes", normalise("Lovers + Friends")): "The Dresser",
    ("clothes", normalise("Milly")): "The Dresser",
    ("clothes", normalise("Pretty Lavish")): "The Dresser",
    ("clothes", normalise("Sugarlips")): "The Dresser",
    ("clothes", normalise("Ted Baker")): "The Dresser",
    ("clothes", normalise("Maria de la Orden")): "The Alumni",
    ("clothes", normalise("Atelier Delphine")): "Modern Minimalist",
    ("clothes", normalise("Dries Van Noten")): "The Trove",
    ("clothes", normalise("Karen Walker")): "The Trove",
    ("clothes", normalise("Wales Bonner")): "The Wanderer",
    ("clothes", normalise("Orange Culture")): "The Wanderer",
    ("clothes", normalise("Jude Connally")): "Riviera",
    ("clothes", normalise("Lioness")): "The Dresser",
    ("clothes", normalise("Never Fully Dressed")): "The Trove",
    ("clothes", normalise("Vilagallo")): "The Trove",
    ("clothes", normalise("Aime Leon Dore")): "The Alumni",
    ("clothes", normalise("Kith")): "The Alumni",
    ("clothes", normalise("Maison Kitsuné")): "The Alumni",
    ("clothes", normalise("CASABLANCA")): "Riviera",
    ("clothes", normalise("Jacquemus")): "Riviera",
    ("clothes", normalise("With Jean")): "Riviera",
    ("clothes", normalise("Damson Madder")): "The Trove",
    ("clothes", normalise("Lucy & Yak")): "The Trove",
    ("clothes", normalise("Maxhosa Africa")): "The Trove",
    ("clothes", normalise("Paul Smith")): "The Dresser",
    ("clothes", normalise("Gucci")): "The Dresser",
    ("clothes", normalise("Anna October")): "The Dreamer",
    ("clothes", normalise("S.S. Daley")): "The Dreamer",
    ("clothes", normalise("Twinset")): "The Dreamer",
    ("clothes", normalise("Ludovic De Saint Sernin")): "After Midnight",
    ("clothes", normalise("McQueen")): "After Midnight",
    ("clothes", normalise("The Kooples")): "After Midnight",
    ("clothes", normalise("Wolford")): "After Midnight",
    ("clothes", normalise("Suta")): "The Wanderer",
    ("clothes", normalise("Mackintosh")): "The Expedition",
    ("clothes", normalise("Nanamica")): "The Expedition",
    ("clothes", normalise("SOIA & KYO")): "Modern Minimalist",
    ("clothes", normalise("Wardrobe Nyc")): "Modern Minimalist",
    ("clothes", normalise("DAVID KOMA")): "The Dresser",
    ("clothes", normalise("DIMA AYAD")): "The Dresser",
    ("clothes", normalise("Mirror Palais")): "The Dreamer",
    ("clothes", normalise("Bailey 44")): "Modern Minimalist",
    ("clothes", normalise("Joah Brown")): "Endorphin",
    ("clothes", normalise("Spiritual Gangster")): "Endorphin",
    ("clothes", normalise("Missoni")): "The Trove",
    ("clothes", normalise("Nobody's Child")): "The Dreamer",
    ("clothes", normalise("Anine Bing")): "Modern Minimalist",
    ("clothes", normalise("TWP")): "Modern Minimalist",
    ("clothes", normalise("GAP")): "The Alumni",
    ("clothes", normalise("Golden Goose")): "The Underground",
    ("clothes", normalise("Needles")): "The Underground",
    ("clothes", normalise("Von Dutch")): "The Underground",
    ("clothes", normalise("Western Aloha")): "Riviera",
    ("clothes", normalise("Molly Goddard")): "The Dreamer",
    ("clothes", normalise("Samantha Pleet")): "The Dreamer",
    ("clothes", normalise("Sister Jane")): "The Dreamer",
    ("clothes", normalise("Moschino")): "The Disruptor",
    ("clothes", normalise("Polo Ralph Lauren")): "The Alumni",
    ("clothes", normalise("Deiji Studios")): "Sun Bleached",
    ("clothes", normalise("Erdem")): "The Dreamer",
    ("clothes", normalise("Siedrés")): "Riviera",
    ("clothes", normalise("Andersson Bell")): "The Disruptor",
    ("clothes", normalise("Denim Tears")): "The Underground",
    ("clothes", normalise("Diesel")): "The Underground",
    ("clothes", normalise("Jordanluca")): "After Midnight",
    ("clothes", normalise("Ksubi")): "The Underground",
    ("clothes", normalise("Marques Almeida")): "The Disruptor",
    ("clothes", normalise("Purple Brand")): "The Underground",
    ("clothes", normalise("Simon Miller")): "Modern Minimalist",
    ("shoes", normalise("Coperni")): "The Disruptor",
    ("shoes", normalise("LaBante London")): "Modern Minimalist",
    ("shoes", normalise("Melissa")): "The Trove",
    ("shoes", normalise("Onitsuka Tiger")): "The Alumni",
    ("shoes", normalise("Charlotte Stone")): "The Trove",
    ("shoes", normalise("Peterson Stoop")): "The Workshop",
    ("jewellery", normalise("YVMIN")): "The Disruptor",
    ("jewellery", normalise("Parts of Four")): "After Midnight",
    ("jewellery", normalise("Good Art HLYWD")): "The Underground",
    ("jewellery", normalise("MISHO")): "The Disruptor",
    ("jewellery", normalise("Hermès")): "The Understated",
    ("jewellery", normalise("Hermes")): "The Understated",
    ("jewellery", normalise("Marla Aaron")): "The Disruptor",

    # Underground re-review: preserve true street and subculture brands while
    # moving performance, conceptual, sensual and quiet-luxury outliers into
    # the customer-facing archetype their current text and board support.
    ("clothes", normalise("Adidas")): "Endorphin",
    ("clothes", normalise("Alma de Ace")): "The Alumni",
    ("clothes", normalise("AVAVAV")): "The Disruptor",
    ("clothes", normalise("BLACK MILK")): "The Trove",
    ("clothes", normalise("Camiel Fortgens")): "The Disruptor",
    ("clothes", normalise("Comme des Garcons Play")): "The Alumni",
    ("clothes", normalise("Coperni")): "The Disruptor",
    ("clothes", normalise("Courreges")): "The Disruptor",
    ("clothes", normalise("Fashion Brand Company")): "The Disruptor",
    ("clothes", normalise("Fear of God")): "The Understated",
    ("clothes", normalise("I.AM.GIA")): "After Midnight",
    ("clothes", normalise("IRO")): "After Midnight",
    ("clothes", normalise("Jean Paul Gaultier")): "The Disruptor",
    ("clothes", normalise("Jungmaven")): "Sun Bleached",
    ("clothes", normalise("Kappa")): "Endorphin",
    ("clothes", normalise("Karhu")): "Endorphin",
    ("clothes", normalise("KNWLS")): "After Midnight",
    ("clothes", normalise("LOCI")): "Modern Minimalist",
    ("clothes", normalise("MALBON GOLF")): "The Alumni",
    ("clothes", normalise("Marine Serre")): "The Disruptor",
    ("clothes", normalise("Miaou")): "After Midnight",
    ("clothes", normalise("MISBHV")): "After Midnight",
    ("clothes", normalise("New Balance")): "Endorphin",
    ("clothes", normalise("Open YY")): "The Disruptor",
    ("clothes", normalise("Priscavera")): "After Midnight",
    ("clothes", normalise("Puma")): "Endorphin",
    ("clothes", normalise("Superdry")): "The Alumni",

    # The Alumni remains the shared name for prep and everyday uniform. Split
    # the inherited denim cluster so heritage denim reads Workshop, clean denim
    # reads Modern Minimalist and only the genuinely downtown outlier remains
    # Underground.
    ("clothes", normalise("CQY Denim")): "The Workshop",
    ("clothes", normalise("Denime")): "The Workshop",
    ("clothes", normalise("Freenote Cloth")): "The Workshop",
    ("clothes", normalise("Imogene + Willie")): "The Workshop",
    ("clothes", normalise("Levi's")): "The Workshop",
    ("clothes", normalise("Livid")): "The Workshop",
    ("clothes", normalise("Naked & Famous")): "The Workshop",
    ("clothes", normalise("Nudie Jeans")): "The Workshop",
    ("clothes", normalise("Re/Done")): "The Workshop",
    ("clothes", normalise("AG Jeans")): "Modern Minimalist",
    ("clothes", normalise("amo denim")): "Modern Minimalist",
    ("clothes", normalise("ASKK NY")): "Modern Minimalist",
    ("clothes", normalise("Calvin Klein / Calvin Klein Jeans")): "Modern Minimalist",
    ("clothes", normalise("Citizens Of Humanity")): "Modern Minimalist",
    ("clothes", normalise("Closed")): "Modern Minimalist",
    ("clothes", normalise("Facon Jacmin")): "Modern Minimalist",
    ("clothes", normalise("Jacob Cohen")): "Modern Minimalist",
    ("clothes", normalise("Jeanerica")): "Modern Minimalist",
    ("clothes", normalise("Paige")): "Modern Minimalist",
    ("clothes", normalise("BLANK NYC")): "The Alumni",
    ("clothes", normalise("Chopova Lowena")): "The Disruptor",
    ("clothes", normalise("De La Vali")): "The Dresser",
    ("clothes", normalise("Day Birger et Mikkelsen")): "Modern Minimalist",
    ("clothes", normalise("Fabiana Pigna")): "Modern Minimalist",
    ("clothes", normalise("The Odder Side")): "Modern Minimalist",

    # Final Underground second pass. These rows were inherited from broad
    # streetwear clusters, but their board and text support a clearer public
    # archetype.
    ("clothes", normalise("Juicy Couture")): "The Alumni",
    ("clothes", normalise("Missguided")): "The Dresser",
    ("clothes", normalise("NOAH")): "The Alumni",
    ("clothes", normalise("Natasha Zinko")): "The Disruptor",
    ("clothes", normalise("Peachy Den")): "After Midnight",
    ("clothes", normalise("OLOW")): "Sun Bleached",
    ("clothes", normalise("Wood Wood")): "The Alumni",

    # Cross-category accessory audit. Public names remain identical across all
    # categories; only the category-specific descriptions differ.
    ("shoes", normalise("Filling Pieces")): "Modern Minimalist",
    ("bags", normalise("Filling Pieces")): "Modern Minimalist",
    ("jewellery", normalise("Filling Pieces")): "Modern Minimalist",
    ("shoes", normalise("Belstaff")): "The Workshop",
    ("shoes", normalise("Rick Owens")): "After Midnight",
    ("shoes", normalise("Ann Demeulemeester")): "After Midnight",
    ("shoes", normalise("Burton")): "The Dresser",
    ("shoes", normalise("Ganni")): "The Trove",
    ("shoes", normalise("Maison Mihara Yasuhiro")): "The Disruptor",
    ("shoes", normalise("The Attico")): "After Midnight",
    ("shoes", normalise("CASABLANCA")): "Riviera",
    ("shoes", normalise("Jacquemus")): "Riviera",
    ("shoes", normalise("Pucci")): "The Trove",
    ("shoes", normalise("Saltwater Sandals")): "Sun Bleached",
    ("shoes", normalise("Alex Mill")): "The Workshop",
    ("shoes", normalise("Fortela")): "The Workshop",
    ("shoes", normalise("LOCI")): "Modern Minimalist",
    ("bags", normalise("Maison Mihara Yasuhiro")): "The Disruptor",
    ("bags", normalise("MALBON GOLF")): "Endorphin",
    ("jewellery", normalise("Ganni")): "The Trove",
    ("jewellery", normalise("Bottega Veneta")): "The Understated",
    ("jewellery", normalise("Khaite")): "Modern Minimalist",
    ("jewellery", normalise("Prada")): "The Disruptor",
    ("jewellery", normalise("The Fold")): "The Dresser",
    ("jewellery", normalise("Maison Kitsuné")): "The Alumni",
    ("jewellery", normalise("LOCI")): "Modern Minimalist",
    ("jewellery", normalise("Aliita")): "The Dreamer",
    ("jewellery", normalise("Alison Lou")): "The Trove",
    ("jewellery", normalise("Bea Bongiasca")): "The Trove",
    ("jewellery", normalise("Fabergé")): "The Trove",
    ("jewellery", normalise("Loren Stewart")): "Modern Minimalist",
    ("jewellery", normalise("Tarin Thomas")): "Modern Minimalist",
    ("jewellery", normalise("Summer Away")): "Riviera",
    ("jewellery", normalise("Aspiga")): "Riviera",
    ("jewellery", normalise("Alex Mill")): "The Workshop",
    ("jewellery", normalise("Livid")): "The Workshop",
    ("jewellery", normalise("Miansai")): "Modern Minimalist",
    ("bags", normalise("Juicy Couture")): "The Trove",
    ("bags", normalise("Natasha Zinko")): "The Disruptor",
    ("bags", normalise("Junya Watanabe")): "The Disruptor",
    ("bags", normalise("Steve Madden")): "The Dresser",
    ("shoes", normalise("Juicy Couture")): "The Trove",
    ("shoes", normalise("Natasha Zinko")): "The Disruptor",
    ("shoes", normalise("Maison Kitsuné")): "The Alumni",
    ("jewellery", normalise("Juicy Couture")): "The Trove",
    ("jewellery", normalise("Natasha Zinko")): "The Disruptor",
    ("jewellery", normalise("Marine Serre")): "The Disruptor",
    ("jewellery", normalise("Maison Mihara Yasuhiro")): "The Disruptor",
    ("jewellery", normalise("Another Feather")): "The Understated",
    ("jewellery", normalise("Astrid & Miyu")): "Modern Minimalist",
    ("jewellery", normalise("Jennifer Fisher")): "The Disruptor",
    ("jewellery", normalise("Steve Madden")): "The Dresser",
    ("jewellery", normalise("Lizzie Mandler")): "Modern Minimalist",
    ("jewellery", normalise("Arielle De Pinto")): "The Disruptor",
    ("jewellery", normalise("Acne Studios")): "Modern Minimalist",
}


CATEGORY_PUBLIC_DESCRIPTIONS = {
    "bags": {
        "The Expedition": "You choose bags that can take a beating and still look considered, with technical fabrics, useful pockets and practical shapes built for movement.",
        "The Understated": "You prefer beautifully made bags with discreet hardware, excellent leather and no need for a visible logo.",
        "The Disruptor": "Your bag is the part of the outfit that starts the conversation: sculptural, unexpected and deliberately hard to ignore.",
        "The Dreamer": "You reach for soft shapes, pretty details, bows, pearls and romantic colour. A bag can be practical and still feel charming.",
        "The Underground": "Your bags borrow from street, skate and subculture: graphic, relaxed and a little rough around the edges.",
        "Modern Minimalist": "You want clean bags that work with everything: edited shapes, simple colour and easy day-to-night function.",
        "After Midnight": "You are drawn to dark, sensual bags with sharp shapes, unusual hardware and an after-hours edge.",
        "The Trove": "Your bags collect colour, print, texture and charms. More is more, and every piece has something to discover.",
        "Sun Bleached": "You like bags that feel sun-worn and easy, with woven fibres, washed canvas and relaxed coastal texture.",
        "The Workshop": "You value bags that are built to last: sturdy leather, honest construction and hardware that gets better with age.",
        "The Dresser": "You choose polished bags on purpose: structured shapes, refined finishes and the right piece for an occasion.",
        "Endorphin": "Your bags keep pace with an active day: streamlined, sporty and ready for the gym, commute or weekend.",
        "The Wanderer": "You collect bags with craft and place in them, including weaving, embroidery, beadwork and hands-on tradition.",
        "The Alumni": "You like a dependable signature bag: classic, neat and easy to carry from work to coffee to the weekend.",
        "Riviera": "You want your bag to feel ready for somewhere warm, with light materials, bright colour and effortless resort polish.",
    },
    "shoes": {
        "The Expedition": "You choose footwear made for weather, distance and uneven ground, even when the route is mostly city pavement.",
        "The Understated": "You prefer quietly excellent shoes: beautiful materials, restrained shapes and quality that speaks without a logo.",
        "The Disruptor": "Your shoes challenge the rest of the outfit. Sculptural soles, odd proportions and conceptual construction are the point.",
        "The Dreamer": "You like shoes with softness and charm: delicate straps, ballet shapes, colour and romantic detail.",
        "The Underground": "Your footwear comes from skate, street and subculture: worn-in, graphic and never too polished.",
        "Modern Minimalist": "You want clean, versatile shoes that simplify getting dressed and work across most of your wardrobe.",
        "After Midnight": "You gravitate to dark, sensual footwear: sharp boots, severe heels and silhouettes with an after-hours pull.",
        "The Trove": "Your shoe rack is a collection of colour, print, texture and unexpected finds. Restraint is not the objective.",
        "Sun Bleached": "You like easy shoes with a holiday feel, with natural texture, faded colour and relaxed warm-weather shapes.",
        "The Workshop": "You value footwear with honest construction: sturdy leather, workwear roots and soles made to last.",
        "The Dresser": "You enjoy polished footwear and a proper finishing touch: loafers, heels and occasion shoes chosen deliberately.",
        "Endorphin": "You like footwear engineered for movement, from technical runners to training shoes that still look good off-duty.",
        "The Wanderer": "You are drawn to shoes shaped by craft and culture, with woven, embroidered and handmade details rich with story.",
        "The Alumni": "You return to uniform favourites: court shoes, loafers and familiar classics that make every day easier.",
        "Riviera": "You like shoes made for warm weather: sandals, woven textures, colour and a polished holiday ease.",
    },
    "jewellery": {
        "The Expedition": "You like jewellery with engineered hardware, functional fastenings and a tough, utilitarian edge.",
        "The Understated": "You choose discreet, beautifully made jewellery with refined metal, quiet stones and details that reward a closer look.",
        "The Disruptor": "Your jewellery behaves like wearable sculpture: conceptual, unexpected and designed to provoke a second look.",
        "The Dreamer": "You reach for pearls, ribbons, flowers and delicate sparkle. Romantic details soften everything around them.",
        "The Underground": "Your jewellery carries subculture attitude: chains, piercings, oxidised metal and pieces that feel discovered early.",
        "Modern Minimalist": "You want jewellery that layers easily and works every day, with clean lines, simple metal and considered proportion.",
        "After Midnight": "You prefer dark, sensual jewellery with sharp metal, dramatic scale and an intriguing after-hours mood.",
        "The Trove": "You collect colour, charms, stones and texture. Your jewellery box should feel like an excellent vintage market.",
        "Sun Bleached": "You like jewellery with a coastal softness: shells, weathered metal, pale stones and easy natural texture.",
        "The Workshop": "You value visible craft: hammered metal, hand-finished surfaces and pieces made to become personal over time.",
        "The Dresser": "You enjoy jewellery that completes an occasion. It is polished, intentional and just formal enough to feel special.",
        "Endorphin": "You prefer light, practical jewellery that moves with you and never gets in the way of an active day.",
        "The Wanderer": "You collect jewellery rooted in place and handwork, including beads, carving, weaving and cultural craft traditions.",
        "The Alumni": "You return to reliable signatures: hoops, signets, chains and classic pieces worn like a personal uniform.",
        "Riviera": "You like sunlit jewellery with warm gold, bright stones and playful pieces that belong near the sea.",
    },
    "swimwear_lingerie": {
        "The Expedition": "You choose swim and base layers for performance: technical fabrics, secure construction and pieces ready for real activity.",
        "The Understated": "You prefer impeccable, discreet foundations with refined fabric, quiet colour and quality that is felt rather than announced.",
        "The Disruptor": "You are drawn to swim and lingerie that plays with construction, cut-outs and proportion in unexpected ways.",
        "The Dreamer": "You like soft colour, delicate lace, tiny prints and romantic details that make everyday pieces feel special.",
        "The Underground": "Your swim and intimates have a subcultural edge that feels graphic, nostalgic and a little rebellious.",
        "Modern Minimalist": "You want clean, comfortable pieces that layer and combine easily: simple colour, precise fit and no fuss.",
        "After Midnight": "You prefer dark, directional lingerie and swim with strong lines, sheer detail and an unmistakably nocturnal mood.",
        "The Trove": "You collect colour, pattern and texture across swim and lingerie. Matching is optional; personality is not.",
        "Sun Bleached": "You like soft, sun-faded swim and easy natural layers that feel as relaxed as a long morning by the water.",
        "The Workshop": "You value construction you can trust: supportive seams, durable fabric and pieces designed to be worn repeatedly.",
        "The Dresser": "You choose polished lingerie and swim deliberately: refined sets, elegant nightwear and pieces that feel occasion-ready.",
        "Endorphin": "You want swim and bodywear that performs: secure fits, active shapes and colour built for movement.",
        "The Wanderer": "You are drawn to crochet, handwork, print and swim pieces that carry a sense of travel and craft.",
        "The Alumni": "You rely on signature staples: classic cuts, familiar colours and dependable pieces that form an everyday uniform.",
        "Riviera": "You dress for somewhere warm, with polished swim, light cover-ups, colour and an easy Mediterranean confidence.",
    },
}

CATEGORY_PUBLIC_DISPLAY_NAMES: dict[tuple[str, str], str] = {}


def public_display_copy(category: str, public_name: str) -> tuple[str, str]:
    display_name = CATEGORY_PUBLIC_DISPLAY_NAMES.get((category, public_name), public_name)
    description = CATEGORY_PUBLIC_DESCRIPTIONS.get(category, {}).get(
        public_name,
        ARCHETYPE_BY_NAME[public_name]["public_cluster_description"],
    )
    return display_name, description


def public_name_for_brand(category: str, cluster: int, brand_name: str) -> tuple[str, str]:
    try:
        default_name = CLUSTER_PUBLIC_NAMES[category][cluster]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"No public archetype mapping for {category}-{cluster}") from exc
    normalized = normalise(brand_name)
    reviewed_override = GLOBAL_PUBLIC_OVERRIDES.get((category, normalized))
    if reviewed_override:
        return reviewed_override, "reviewed_brand_override"
    for public_name, brands in CATEGORY_PUBLIC_OVERRIDES.get(category, {}).get(cluster, {}).items():
        if normalized in brands:
            return public_name, "reviewed_brand_override"
    method = (
        "dominant_cluster_default"
        if cluster in CATEGORY_PUBLIC_OVERRIDES.get(category, {})
        else "direct_cluster_mapping"
    )
    return default_name, method


def enrich_public_archetypes(data_dir: Path) -> None:
    assignments_path = data_dir / "ios_brand_cluster_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    original_identity = assignments[["brand_name", "category", "cluster", "cluster_id"]].copy()

    public_rows = []
    for row in assignments.itertuples(index=False):
        public_name, method = public_name_for_brand(str(row.category), int(row.cluster), str(row.brand_name))
        public = ARCHETYPE_BY_NAME[public_name]
        display_name, display_description = public_display_copy(str(row.category), public_name)
        public_rows.append({
            **public,
            "public_display_name": display_name,
            "public_display_description": display_description,
            "public_mapping_method": method,
        })
    public_frame = pd.DataFrame(public_rows, index=assignments.index)
    for column in public_frame.columns:
        assignments[column] = public_frame[column]

    if not original_identity.equals(assignments[["brand_name", "category", "cluster", "cluster_id"]]):
        raise RuntimeError("Public archetype enrichment changed technical cluster identity")
    if (
        assignments["public_cluster_name"].eq("").any()
        or assignments["public_cluster_description"].eq("").any()
        or assignments["public_display_name"].eq("").any()
        or assignments["public_display_description"].eq("").any()
    ):
        raise RuntimeError("Public archetype enrichment left blank customer-facing copy")

    assignments.to_csv(assignments_path, index=False)
    assignments.to_csv(data_dir / "ios_brand_cluster_assignments.tsv", index=False, sep="\t")

    public_assignments = assignments[[
        "brand_name", "category", "cluster", "cluster_id", "public_cluster_id",
        "public_cluster_number", "public_cluster_name", "public_cluster_description",
        "public_display_name", "public_display_description",
        "public_mapping_method",
    ]].copy()
    public_assignments.to_csv(data_dir / "brand_public_cluster_assignments.csv", index=False)

    join_columns = [
        "brand_name", "category", "public_cluster_id", "public_cluster_number",
        "public_cluster_name", "public_cluster_description", "public_mapping_method",
        "public_display_name", "public_display_description",
    ]
    public_join = assignments[join_columns].copy()
    umap_path = data_dir / "ios_brand_umap.csv"
    umap = pd.read_csv(umap_path)
    technical_umap = umap[["brand_name", "category", "cluster", "cluster_id", "x", "y", "umap_x", "umap_y"]].copy()
    umap = umap.drop(columns=[column for column in join_columns[2:] if column in umap.columns])
    umap = umap.merge(public_join, on=["brand_name", "category"], how="left", validate="one_to_one")
    if umap["public_cluster_id"].isna().any():
        raise RuntimeError("UMAP public archetype join has missing rows")
    if not technical_umap.equals(umap[technical_umap.columns]):
        raise RuntimeError("Public archetype enrichment changed UMAP coordinates or numeric clusters")
    umap.to_csv(umap_path, index=False, float_format="%.8f")
    umap.to_csv(data_dir / "ios_brand_umap.tsv", index=False, sep="\t", float_format="%.8f")

    summary_path = data_dir / "ios_cluster_summary.csv"
    summary = pd.read_csv(summary_path)
    dominant_ids, dominant_names, dominant_descriptions = [], [], []
    dominant_display_names, dominant_display_descriptions, statuses = [], [], []
    for row in summary.itertuples(index=False):
        category = str(row.category)
        cluster = int(row.cluster)
        name = CLUSTER_PUBLIC_NAMES[category][cluster]
        public = ARCHETYPE_BY_NAME[name]
        display_name, display_description = public_display_copy(category, name)
        dominant_ids.append(public["public_cluster_id"])
        dominant_names.append(name)
        dominant_descriptions.append(public["public_cluster_description"])
        dominant_display_names.append(display_name)
        dominant_display_descriptions.append(display_description)
        members = assignments[
            assignments["category"].eq(category) & assignments["cluster"].eq(cluster)
        ]
        statuses.append(
            "split_with_brand_overrides"
            if members["public_cluster_id"].nunique() > 1
            else "cluster_mapping"
        )
    summary["dominant_public_cluster_id"] = dominant_ids
    summary["dominant_public_cluster_name"] = dominant_names
    summary["dominant_public_cluster_description"] = dominant_descriptions
    summary["dominant_public_display_name"] = dominant_display_names
    summary["dominant_public_display_description"] = dominant_display_descriptions
    summary["public_mapping_status"] = statuses
    summary.to_csv(summary_path, index=False)

    archetype_frame = pd.DataFrame([
        {
            "public_cluster_id": f"archetype-{number:02d}",
            "public_cluster_number": number,
            "public_cluster_name": name,
            "public_cluster_core_aesthetic": core,
            "public_cluster_description": description,
        }
        for number, name, core, description in ARCHETYPES
    ])
    archetype_frame.to_csv(data_dir / "public_cluster_archetypes.csv", index=False)

    summary_rows = []
    for (category, public_id), group in assignments.groupby(["category", "public_cluster_id"], sort=False):
        first = group.iloc[0]
        source_cluster_ids = sorted(
            group["cluster_id"].astype(str).unique(),
            key=lambda value: (value.rsplit("-", 1)[0], int(value.rsplit("-", 1)[1])),
        )
        summary_rows.append({
            "category": category,
            "public_cluster_id": public_id,
            "public_cluster_number": int(first["public_cluster_number"]),
            "public_cluster_name": first["public_cluster_name"],
            "public_cluster_description": first["public_cluster_description"],
            "public_display_name": first["public_display_name"],
            "public_display_description": first["public_display_description"],
            "n_brands": len(group),
            "source_cluster_ids": ", ".join(source_cluster_ids),
            "example_brands": ", ".join(group["brand_name"].astype(str).head(8)),
        })
    pd.DataFrame(summary_rows).sort_values(
        ["category", "public_cluster_number"]
    ).to_csv(data_dir / "ios_public_cluster_summary.csv", index=False)

    anchors = {
        ("clothes", "alo yoga"): "Endorphin",
        ("clothes", "marine layer"): "Sun Bleached",
        ("clothes", "pucci"): "The Trove",
        ("clothes", "carhartt wip"): "The Workshop",
        ("clothes", "a cold wall"): "After Midnight",
    }
    lookup = {
        (str(row.category), normalise(str(row.brand_name))): str(row.public_cluster_name)
        for row in assignments.itertuples(index=False)
    }
    for key, expected in anchors.items():
        if key in lookup and lookup[key] != expected:
            raise RuntimeError(f"Public archetype anchor mismatch for {key}: {lookup[key]} != {expected}")
