# reads brands_catalog.csv and writes a clean file.
import re
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
try:
    DATA_DIR
except NameError:
    DATA_DIR = Path('.')

CLEAN_INPUT  = DATA_DIR / 'brands_catalog.csv'
CLEAN_OUTPUT = DATA_DIR / 'brands_catalog_clean.csv'

def _norm(s):
    """Takes text and returns a spaced key by keeping letters and numbers."""
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

def _sq(s):
    """Takes text and returns only lowercase letters and numbers."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _dom(u):
    """Takes a URL and returns a clean domain by removing www."""
    return re.sub(r'^www\.', '', (urlparse(str(u)).netloc or str(u))).lower().strip('/')

# 1) Mass-market / fast-fashion / high-street labels with no distinct design
#    language excluded for a brand-aesthetic dataset. 
#    Kept on purpose despite being high-street (recognisable design language):
#    & Other Stories, Monki, Weekday, Superdry, Abercrombie & Fitch, House of CB, Quay, Vans.
EXCLUDE_MASS_MARKET = {_sq(x) for x in [
    'ASOS','Boohoo','BoohooMAN','PrettyLittleThing','Missguided','Shein','Romwe','Zaful','Cider','Edikted',
    'H&M','Zara','Bershka','Pull & Bear','Stradivarius','Massimo Dutti','Oysho','Mango','Primark','Topshop',
    'Topman','River Island','New Look','Next','Gap','Old Navy','Banana Republic','Forever 21',
    'Brandy Melville','Cotton On','Jack & Jones','Vero Moda','Only','Pieces','Selected','Esprit','C&A',
    'Reserved','Sinsay','Hollister','Aeropostale','PacSun','American Eagle','Aerie','In The Style','Motel',
    'SIMMI','Public Desire','Princess Polly','Oh Polly','Tiger Mist','Meshki','White Fox','Showpo','Hello Molly',
    'NA-KD','Femme Luxe','Ego','Quiz','Lipsy','Lasula','Nasty Gal','Boden','Joules','Fat Face','White Stuff','Temu',
    # mass-market sportswear (same class as Nike/Adidas, dropped in the pipeline). Vans kept by request.
    'Saucony','Skechers','New Balance',
]}
# Sportswear giants that also leak as model-number rows (e.g. 'New Balance 530').
MASS_PREFIXES = ('new balance',)

# 2) Nav / category / generic words that leaked in as "brands" (exact match).
JUNK_LABELS = {_norm(x) for x in [
    # apparel categories
    'clothing','clothes','apparel','fashion','denim','knitwear','cashmere','outerwear','activewear','loungewear',
    'sleepwear','tailoring','suits','suit','blazers','jackets','jacket','leather jackets','coats','coat','shirts',
    'shirt','t shirts','t shirt','tees','tops','top','bottoms','trousers','pants','jeans','skirts','dresses','dress',
    'sweaters','sweater','hoodies','swimwear','swim','swimsuit','swimsuits','lingerie','underwear','intimates',
    'intimates & lingerie','intimates lingerie',
    # footwear
    'shoes','footwear','sneakers','sneaker','trainers','boots','sandals','sandal','heels','loafers','slippers','baskets',
    'slip ons','slip on','slides','clogs','mules','wedges','flats','espadrilles','pumps',
    # bags
    'bags','bag','handbags','backpacks','backpack','backpacks 2','totes','luggage','luggage & travel','travel bags',
    'man bags','purses','clutches',
    # jewellery / accessories
    'jewellery','jewelry','jewels','rings','ring','necklaces','earrings','bracelets','pendants','charms','watches',
    'luxury watches','fashion watches','accessories','baby accessories','belts','scarves','scarf','hats','hat',
    'gloves','socks','sunglasses','glasses','eyewear','optical','cufflinks','ties','tie','brooches','anklets','chokers',
    # navigation / generic
    'collection','collections','store','stores','stories','brands','designers','new in','new arrival','new arrivals',
    'best sellers','bestsellers','sale','shop all','view all','all','gifts','gift cards','personalised gifts','home',
    'beauty','kids','kidswear','baby','women','woman','womens','men','mens','menswear','womenswear','print','prints',
    'blankets','cactus leather','vegan leather','wedding','bridal','crochet','finland','mali','page','september','hotness',
    'all gifts','all products','all hands','all clothing','artists','classic shoes','cotswold','cowboy boots',
    'cycling','kid s clothing','kids clothing','vegan shoes','wedding shoes','western boots',
    'pre owned','pre-owned','swimwear 1','wedding shop','wedding bands','wedding band','vintage jewelry',
    'vintage jewellery','streetwear','knitwear 2',
]}

# 3) Non-fashion leaks the site-classifier mislabelled.
DROP_BRANDS = {_norm(x) for x in [
    'Astier De Villatte', 'Yeti', 'MAC',                 
    'Cano', 'laps', 'vamp',                             
    # scraper artifacts / wrong-entity rows
    'Hockey', 'Bed', 'Body', 'CFDA', 'Goa', 'Prev', 'Vt', 'Skall',
    # corrupted / truncated names -> re-added correctly via ADD_BRANDS
    'Mad', 'Jack + G', 'Karma8a', 'oof', 'LAND', 'Jude', 'for all Mankind', 'MArc Jacobs',
    'Stella Mc Cartney', "Spring Summer '24", 'Knitwear 2',
]}

# Multi-brand retailers / boutiques / media platforms that are not brands. Pulled into a
# separate sources table (brands_sources.csv) and removed from the brand catalogue.
RETAILERS = {_norm(x) for x in [
    'Italist', 'Machine-A', 'Selfridges +', 'The Broken Arm', 'Goodhood', 'Slam Jam', 'Burke Mercantile',
    'One of a Few', 'Order of Style', 'Our Green House', 'Prefontaine', 'The Collective Park City',
    'Lewisburg Surf Shop', 'Ocelot Market', 'Immaculate Vegan', 'Ten Thousand Villages', 'Vegan Bags',
    'APOC Store', 'Bristol General Store', 'London Store', 'The Sneaker Store', 'Sneaker Space',
    'Hypebeast', 'No. 6 Store', 'Dear Neighbor', 'Advice from a Caterpillar', 'Better Gift Shop',
]}

# 3b) Duplicate brand variants the space-stripped de-dup can't catch (different
#     suffixes)
DROP_DUPLICATES = {_norm(x) for x in [
    'Nudie', 'Nudie Jeans Co',          # keep 'Nudie Jeans'
    'Ancient Greek Sandal',             # keep 'Ancient Greek Sandals'
    'Faithfull',                        # keep 'Faithfull The Brand'
    'Faherty Brand',                    # keep 'Faherty'
    'Donsje Amsterdam',                 # keep 'Donsje'
    'VAGABOND Shoemakers',              # keep 'Vagabond'
    'Vivienne Westwood 1',              # keep 'Vivienne Westwood'
    'Mihara Yasuhiro',                  # keep 'Maison Mihara Yasuhiro'
    'Serge Denim', 'Serge',             # keep 'Serge DeNimes'
]}

# 3c) Wrong Clearbit domains for trusted seeds.
WEBSITE_OVERRIDES = {_norm(k): v for k, v in {
    'Atoms': 'https://www.atoms.com',
    # corrected wrong Clearbit domains for trusted seeds (fail-closed -> manual URL)
    'Arnhem': 'https://arnhem.com.au', 'Ghost': 'https://www.ghost.co.uk', 'Kirsh': 'https://kirsh.co.kr',
    'ME+EM': 'https://www.meandem.com', 'Omnes': 'https://www.omnes.com',
    'People Tree': 'https://www.peopletree.co.uk', 'Quince': 'https://www.quince.com',
    'Recto': 'https://recto.co', 'Rixo': 'https://www.rixo.co.uk', 'Sui': 'https://wearesui.com',
    'Spell': 'https://spell.co',
    # QA pass: corrected wrong scraper/Clearbit domains
    'Nu-In': 'https://www.nu-in.com', 'TOMS': 'https://www.toms.com', 'Avenue 67': 'https://www.avenue67.com',
    'Bailey 44': 'https://www.bailey44.com', 'Norda': 'https://www.nordarun.com', 'Beaster': 'https://beaster.com',
    'Foufou': 'https://foufou.jp', 'CLANE': 'https://clane-design.com', 'TWP': 'https://www.twpclothing.com',
}.items()}

# known wrong-entity domains
DOMAIN_BLOCKLIST = {
    'ghost.org', 'arnhem.nl', 'canonrumors.com', 'clanespectaculos.com', 'kirshchiropractic.com.au',
    'meemapps.com', 'nuinvest.com.br', 'omneseducation.com', 'peopletreehospitals.com',
    'quinceanera-boutique.com', 'rectorlawfirm.com', 'rixos.com', 'suicidegirls.com', 'spellboy.com',
    'tomsguide.com', 'twperry.com', 'nordace.com', 'baileysblossoms.com', 'avenue.com',
    'hockey-reference.com', 'vt.edu', 'bodybuilding.com', 'prevention.com', 'macys.com', 'bedbathandbeyond.com',
    'goat.com', 'cfda.com',
}

# 4) Manual categories for classifier drift and unknown seeds.
CATEGORY_OVERRIDES = {_norm(k): v for k, v in {
     'Anya Hindmarch': 'only bags', 'Mulberry': 'only bags', 'Mlouye': 'only bags',
    'Marge Sherwood': 'only bags', 'Yuzefi': 'only bags', 'Alexis Bittar': 'only jewellery',
    'Aeyde': 'only shoes', 'ATP Atelier': 'only shoes', 'Neous': 'only shoes', 'Mansur Gavriel': 'only bags',
    'Agent Provocateur': 'only lingerie', 'Boux Avenue': 'only lingerie', 'Wacoal': 'only lingerie',
    'Hanky Panky': 'only lingerie', 'Curvy Kate': 'only lingerie', 'Modibodi': 'only lingerie',
    'Montelle': 'only lingerie', 'Playful Promises': 'only lingerie',
    'Becksondergaard': 'only accessories', 'Chrome Hearts': 'all', 'Botanica Workshop': 'only jewellery',
    'Foufou': 'clothes', 'VANS': 'all',
    'Accessorize': 'only accessories', 'Aldo': 'only shoes', 'Camper': 'only shoes', 'Chaco': 'only shoes',
    'Danner': 'only shoes', 'DC Shoes': 'only shoes', 'JW Pei': 'only bags', 'Okhtein': 'only bags',
    'Vagabond': 'only shoes',
    'Aspinal of London': 'only bags', 'Bembien': 'only bags', 'Bunney': 'only jewellery',
    'Charles & Keith': 'only shoes', 'Chie Mihara': 'only shoes', 'Dolce Vita': 'only shoes',
    'Duluth Pack': 'only bags', 'Fair Anita': 'only jewellery', 'Flabelus': 'only shoes',
    'Inuikii': 'only shoes', 'Jeffrey Campbell': 'only shoes', 'Kurt Geiger': 'only shoes',
    'La Canadienne': 'only shoes', 'LeLe Sadoughi': 'only jewellery', 'Montblanc': 'only bags',
    'Naghedi': 'only bags', 'Noonday Collection': 'only jewellery', 'Philippe Model': 'only shoes',
    'Poppy Lissiman': 'only accessories', 'Premiata': 'only shoes', 'Santoni': 'only shoes',
    'Sebago': 'only shoes', 'Sensi Studio': 'only bags', 'Shinola': 'only bags', 'Two Jeys': 'only jewellery',
    'Norda': 'only shoes',
    'CLANE': 'clothes', 'TWP': 'clothes',
    'Arnhem': 'clothes', 'Atoms': 'only shoes', 'Beaster': 'clothes', 'Bil Arabi': 'only jewellery',
    'Cano': 'only shoes', 'Cariuma': 'only shoes', 'Cefinn': 'clothes', 'CLANE': 'clothes', 'Ghost': 'clothes',
    'Kirsh': 'clothes', 'Kotn': 'clothes', 'Lenny Niemeyer': 'only swimwear', 'Lisou': 'clothes',
    'Maia Active': 'clothes', 'Mardi Mercredi': 'clothes', 'Marhen.J': 'only bags', 'ME+EM': 'clothes',
    'Nonlocal': 'clothes', 'Nu-In': 'clothes', 'Omnes': 'clothes', 'Particle Fever': 'clothes',
    'People Tree': 'clothes', 'Polène': 'only bags', 'Quince': 'all', 'Recto': 'clothes', 'Rich Mnisi': 'clothes',
    'Rixo': 'clothes', 'Rouje': 'clothes', 'Sezane': 'all', 'Songmont': 'only bags', 'Spell': 'clothes',
    'Sui': 'clothes', 'Thought Clothing': 'clothes', 'Tongoro': 'clothes',
}.items()}

# 5) Canon anchors absent from scraping 
ADD_BRANDS = [
    # non-wholesaling luxury
    ('Chanel', 'https://www.chanel.com', 'all'),
    ('Hermès', 'https://www.hermes.com', 'all'),
    ('Louis Vuitton', 'https://www.louisvuitton.com', 'all'),
    ('Dior', 'https://www.dior.com', 'all'),
    ('Saint Laurent', 'https://www.ysl.com', 'all'),
    ('Burberry', 'https://www.burberry.com', 'all'),
    ('Fendi', 'https://www.fendi.com', 'all'),
    ('Van Cleef & Arpels', 'https://www.vancleefarpels.com', 'only jewellery'),
    # major contemporary
    ('Jacquemus', 'https://www.jacquemus.com', 'all'),
    ('The Row', 'https://www.therow.com', 'all'),
    ('Ganni', 'https://www.ganni.com', 'all'),
    ('Reiss', 'https://www.reiss.com', 'all'),
    ('Maje', 'https://www.maje.com', 'all'),
    ('Sandro', 'https://www.sandro-paris.com', 'all'),
    ('Rohe', 'https://roheframes.com', 'clothes'),
    ('For Love & Lemons', 'https://www.forloveandlemons.com', 'clothes'),
    # streetwear
    ('Supreme', 'https://supreme.com', 'clothes'),
    ('Palace', 'https://www.palaceskateboards.com', 'clothes'),
    ('Kith', 'https://kith.com', 'all'),
    ('Fear of God', 'https://fearofgod.com', 'clothes'),
    ('Represent', 'https://representclo.com', 'clothes'),
    ('Corteiz', 'https://crtz.xyz', 'clothes'),
    # key footwear
    ('Common Projects', 'https://www.commonprojects.com', 'only shoes'),
    ('Veja', 'https://www.veja-store.com', 'only shoes'),
    ('Dr. Martens', 'https://www.drmartens.com', 'only shoes'),
    ('Salomon', 'https://www.salomon.com', 'only shoes'),
    # denim
    ('Re/Done', 'https://shopredone.com', 'clothes'),
    ('Mother', 'https://www.motherdenim.com', 'clothes'),
    # 2nd pass: directional / quiet-luxury / avant-garde anchors the JS pages missed
    ('Comme des Garçons', 'https://comme-des-garcons.com', 'all'),
    ('Rick Owens', 'https://www.rickowens.eu', 'all'),
    ('Yohji Yamamoto', 'https://www.yohjiyamamoto.co.jp', 'all'),
    ('Issey Miyake', 'https://www.isseymiyake.com', 'all'),
    ('Simone Rocha', 'https://www.simonerocha.com', 'all'),
    ('Molly Goddard', 'https://mollygoddard.com', 'clothes'),
    ('Our Legacy', 'https://www.ourlegacy.com', 'clothes'),
    ('Bode', 'https://bode.com', 'clothes'),
    ('Coperni', 'https://coperni.com', 'all'),
    ('Loro Piana', 'https://www.loropiana.com', 'all'),
    ('Max Mara', 'https://www.maxmara.com', 'all'),
    ('Zegna', 'https://www.zegna.com', 'all'),
    ('Brunello Cucinelli', 'https://www.brunellocucinelli.com', 'all'),
    ('Gianvito Rossi', 'https://www.gianvitorossi.com', 'only shoes'),
    ('Roger Vivier', 'https://www.rogervivier.com', 'only shoes'),
    # --- coverage pass: contemporary DTC womenswear ---
    ('Djerf Avenue', 'https://djerfavenue.com', 'clothes'),
    ('Odd Muse', 'https://oddmuse.co.uk', 'clothes'),
    ('Rat & Boa', 'https://ratandboa.com', 'clothes'),
    ('Mirror Palais', 'https://mirrorpalais.com', 'clothes'),
    ('Orseund Iris', 'https://orseundiris.com', 'clothes'),
    ('Musier Paris', 'https://musier-paris.com', 'clothes'),
    ('Siedrés', 'https://siedres.com', 'clothes'),
    ('Tove', 'https://www.tove-studio.com', 'clothes'),
    ('Christopher Esber', 'https://christopheresber.com.au', 'clothes'),
    ('Aje', 'https://aje.com.au', 'clothes'),
    ('Aritzia', 'https://www.aritzia.com', 'clothes'),
    # --- European contemporary mid-market ---
    ('A.P.C.', 'https://www.apc.fr', 'all'),
    ('ba&sh', 'https://ba-sh.com', 'clothes'),
    ('Claudie Pierlot', 'https://www.claudiepierlot.com', 'clothes'),
    ('IRO', 'https://www.iroparis.com', 'clothes'),
    ('The Kooples', 'https://www.thekooples.com', 'clothes'),
    ('Stine Goya', 'https://stinegoya.com', 'clothes'),
    ('ROTATE Birger Christensen', 'https://rotatebirgerchristensen.com', 'clothes'),
    ('Samsøe Samsøe', 'https://www.samsoe.com', 'clothes'),
    ('Rodebjer', 'https://rodebjer.com', 'clothes'),
    ('House of Dagmar', 'https://houseofdagmar.com', 'clothes'),
    ('HOPE Stockholm', 'https://hope-sthlm.com', 'clothes'),
    ('Loulou Studio', 'https://www.loulou-studio.com', 'clothes'),
    # --- menswear / heritage / streetwear ---
    ('Universal Works', 'https://universalworks.co.uk', 'clothes'),
    ("Drake's", 'https://www.drakes.com', 'clothes'),
    ('YMC', 'https://youmustcreate.com', 'clothes'),
    ('Patta', 'https://patta.nl', 'clothes'),
    ('Dime', 'https://www.dimemtl.com', 'clothes'),
    ('Drôle de Monsieur', 'https://droledemonsieur.com', 'clothes'),
    # --- luxury holes ---
    ('Versace', 'https://www.versace.com', 'all'),
    ('Ferragamo', 'https://www.ferragamo.com', 'all'),
    ('Bally', 'https://www.bally.com', 'all'),
    ('Balmain', 'https://www.balmain.com', 'all'),
    ('Canada Goose', 'https://www.canadagoose.com', 'clothes'),
    # --- outdoor / technical (cluster included cleanly; Arc'teryx/Salomon/Snow Peak kept) ---
    ('Patagonia', 'https://www.patagonia.com', 'clothes'),
    ('The North Face', 'https://www.thenorthface.com', 'all'),
    ('Columbia', 'https://www.columbia.com', 'all'),
    ('Goldwin', 'https://www.goldwin-global.com', 'clothes'),
    ('ROA', 'https://roa-hiking.com', 'only shoes'),
    ('Hoka', 'https://www.hoka.com', 'only shoes'),
    ('On', 'https://www.on.com', 'only shoes'),
    ('and wander', 'https://www.andwander.com', 'clothes'),
    # corrected names for corrupted/truncated scrape rows (originals dropped above)
    ('Madewell', 'https://www.madewell.com', 'clothes'),
    ('Jack Georges', 'https://www.jackgeorges.com', 'only bags'),
    ('Karma and Luck', 'https://www.karmaandluck.com', 'only jewellery'),
    ('OOFOS', 'https://www.oofos.com', 'only shoes'),
    ("Lands' End", 'https://www.landsend.com', 'clothes'),
    ('Jude Connally', 'https://www.judeconnally.com', 'clothes'),
    ('7 For All Mankind', 'https://www.7forallmankind.com', 'clothes'),
    ('Marc Jacobs', 'https://www.marcjacobs.com', 'all'),
    ('Stella McCartney', 'https://www.stellamccartney.com', 'all'),
    # directional high-street (kept for real aesthetic signal; & Other Stories/Monki/Weekday already kept)
    ('COS', 'https://www.cos.com', 'all'),
    ('Arket', 'https://www.arket.com', 'all'),
    ('Uniqlo', 'https://www.uniqlo.com', 'clothes'),
    ('J.Crew', 'https://www.jcrew.com', 'all'),
    # re-added with verified manual URLs (Clearbit had wrong entities)
    ('CLANE', 'https://clane-design.com', 'clothes'),
    ('TWP', 'https://www.twpclothing.com', 'clothes'),
]

df = pd.read_csv(CLEAN_INPUT)
n0 = len(df)
df['_n']  = df['brand_name'].map(_norm)
df['_sq'] = df['brand_name'].map(_sq)

# apply website + category overrides first
df['official_website'] = df.apply(lambda r: WEBSITE_OVERRIDES.get(r['_n'], r['official_website']), axis=1)
df['category'] = df.apply(lambda r: CATEGORY_OVERRIDES.get(r['_n'], r['category']), axis=1)

mass = df['_sq'].isin(EXCLUDE_MASS_MARKET) | df['_n'].str.startswith(MASS_PREFIXES)
junk = df['_n'].isin(JUNK_LABELS)
nonf = df['_n'].isin(DROP_BRANDS)
dupe = df['_n'].isin(DROP_DUPLICATES)
ret  = df['_n'].isin(RETAILERS)
block = df['official_website'].map(_dom).isin(DOMAIN_BLOCKLIST)   # fail-closed wrong-entity domains
# retailers/sources are not brands: save to a separate table, then drop from catalogue
(df[ret][['brand_name', 'official_website']].drop_duplicates('brand_name')
   .sort_values('brand_name', key=lambda s: s.str.lower())
   .to_csv(DATA_DIR / 'brands_sources.csv', index=False))
print(f'removing: {int(mass.sum())} mass-market, {int(junk.sum())} junk labels, {int(nonf.sum())} '
      f'non-fashion/artifact, {int(dupe.sum())} dup variants, {int(ret.sum())} retailers->brands_sources.csv', flush=True)
print(f'  + {int(block.sum())} blocklisted-domain rows', flush=True)
df = df[~(mass | junk | nonf | dupe | ret | block)].copy()

# de-dup A: by name (space-insensitive). Prefer a real category + readable spelling.
df['_unknown'] = (df['category'] == 'unknown')
df['_pref'] = df['brand_name'].map(lambda s: (s.isupper(), -(s.count(' ') + s.count('.') + s.count('&')), len(s)))
b = len(df)
df = df.sort_values(['_unknown', '_pref']).drop_duplicates('_sq', keep='first')
print(f'de-dup (name)    removed {b - len(df)} rows', flush=True)

# de-dup B: by website. catches short names and variant spellings.
# keeper = most complete name, then non-all-caps, then real category.
df['_dom'] = df['official_website'].map(_dom)
df['_wpref'] = df['brand_name'].map(lambda s: (-sum(c.isalpha() for c in str(s)), str(s).isupper(), len(str(s))))
b = len(df)
dedup = df[df['_dom'] != ''].sort_values(['_unknown', '_wpref']).drop_duplicates('_dom', keep='first')
df = pd.concat([dedup, df[df['_dom'] == '']], ignore_index=True)
print(f'de-dup (website) removed {b - len(df)} rows', flush=True)

clean = df[['brand_name', 'official_website', 'category']].copy()

# 5b) inject the canon anchors that aren't already present (by name or domain)
add = pd.DataFrame(ADD_BRANDS, columns=['brand_name', 'official_website', 'category'])
have = set(clean['brand_name'].map(_sq)) | set(clean['official_website'].map(_dom))
add = add[~(add['brand_name'].map(_sq).isin(have) | add['official_website'].map(_dom).isin(have))]
print(f'added {len(add)} canon anchors (of {len(ADD_BRANDS)})', flush=True)
clean = pd.concat([clean, add], ignore_index=True)

clean = (clean.sort_values('brand_name', key=lambda s: s.str.lower()).reset_index(drop=True))
assert not clean['official_website'].map(_dom).isin(DOMAIN_BLOCKLIST).any(), 'blocklisted domain leaked!'
clean.to_csv(CLEAN_OUTPUT, index=False)
print(f'\n{n0:,} -> {len(clean):,} brands written to {CLEAN_OUTPUT}', flush=True)
print('\ncategory counts:', flush=True)
print(clean['category'].value_counts().to_string(), flush=True)
clean.head(40)
