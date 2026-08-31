# Helper code for the curation notebook.

ALIASES = {
    'Comme Des Garcons': 'Comme des Garcons', 'Stussy': 'Stussy',
    'A Cold Wall': 'A-Cold-Wall*', 'A-Cold-Wall': 'A-Cold-Wall*',
}

STOP_PHRASES = {
    'new in','sale','women','men','kids','home','beauty','designers','brands','all brands','all designers',
    'clothing','shoes','bags','accessories','jewellery','jewelry','fragrance','gifts','editorial','journal',
    'account','wishlist','cart','bag','checkout','customer care','contact us','privacy policy','terms and conditions',
    'shipping','returns','size guide','search','view all','shop all','all','login','register','stores','newsletter',
    'facebook','instagram','tiktok','pinterest','youtube','twitter','x','email','language','currency','country',
    'black friday','gift card','gift cards','sustainability','about us','careers','help','faq','cookie policy',
}
NOISE_PATTERNS = [
    r'^\d+$', r'^[a-z]$', r'^[A-Z]$', r'^[A-Z] - [A-Z]$', r'^A-Z$', r'^0-9$',
    r'^(shop|view|discover|explore|buy|select|filter|sort|show|hide)\b',
    r'\b(size|colour|color|price|shipping|returns|delivery|newsletter|cookies?)\b',
    r'\b(women|mens|men|kids|beauty|home|interiors|sale|new in)\b',
]
JUNK_WORDS = re.compile(r'\b(cart|checkout|items?|articles?|rights reserved|magazine|journal|boutiques?|'
                        r'recipients?|newsletter|account|wishlist|shipping|returns|gift\s?cards?|search|menu|'
                        r'filter|sort|currency|country|exhibition|brands|designers|copyright|reserved|terms|'
                        r'privacy|cookies?)\b', re.I)
# Non-fashion categories rejected by name.
NON_FASHION = re.compile(r'\b(vodka|whisky|whiskey|gin|rum|tequila|wine|beer|liqueur|champagne|cognac|bourbon|'
                         r'spirits?|distillery|lighting|lamp|furniture|sofa|mattress|electronics|appliance|'
                         r'vacuum|speaker|headphones?|camera|laptop|software|hotel|airlines?|bank|insurance|'
                         r'mortgage|university|college|hospital|clinic|pharma|protein|supplement|vitamin|recipe|'
                         r'snack|coffee|tea|chocolate|candy|grocery|supermarket|casino|crypto|nft|skincare|'
                         r'cosmetics?|makeup|haircare|shampoo|fragrance|perfume|parfum|candles?|incense|'
                         r'diffuser|ceramics?|pottery|porcelain|tableware|homeware|interiors?|decor|bedding|'
                         r'kitchenware|cookware|cutlery|rugs?|stationery|publishing)\b', re.I)
CURRENCY = re.compile(r'[$EUR£¥₩]|^\s*[A-Z]{1,3}\s*-\s')

def strip_accents(t):
    """Takes text and returns plain text by removing accents."""
    return ''.join(c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c))

def canonical_key(name):
    """Takes a brand name and returns a clean matching name."""
    name = ALIASES.get(str(name).strip(), str(name).strip())
    name = strip_accents(name).lower().replace('&', ' and ')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', name)).strip()

# Mass-market / non-target labels to drop.
BAD_KEYS = {canonical_key(x) for x in [
    'Nike','Adidas','Puma','New Balance','Converse','UGG','Crocs','Birkenstock','The North Face',
    'Patagonia',"Levi's",'Calvin Klein','Tommy Hilfiger','Gucci Beauty','Dior Beauty','YSL Beauty','Armani Beauty',
]}

def normalize_name(name):
    """Takes a raw name and returns a tidier brand name by removing list noise."""
    s = html_lib.unescape(str(name)).strip().strip('"').strip()
    s = re.sub(r'\s*\(\s*\d+(\s+Articles?)?\s*\)\s*$', '', s, flags=re.I)
    s = re.sub(r'^\d+\s+', '', s)
    s = re.sub(r'^(shop|view|discover|explore|designer|brand)\s+', '', s, flags=re.I)
    return re.sub(r'\s+', ' ', s).strip(' .,-|/\\·•–—:_')

def reject_candidate(name):
    """Takes a name and returns True when the name should be dropped."""
    if not name:
        return True
    key = canonical_key(name)
    if not key or key in STOP_PHRASES or key in BAD_KEYS:
        return True
    if not (2 <= len(name) <= 60):                       return True
    if not re.match(r'^[A-Za-z&]', name):                return True
    if sum(c.isalpha() for c in name) < 2:               return True
    if len(key.split()) > 6:                             return True
    if CURRENCY.search(name) or JUNK_WORDS.search(name): return True
    if NON_FASHION.search(name):                         return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, key, flags=re.I):              return True
    return False

def clean_brand_name(raw):
    """Takes raw text and returns a clean brand name or None by filtering noise."""
    s = normalize_name(raw)
    s = ALIASES.get(s, s)
    return None if reject_candidate(s) else s

print('cleaning + early-filter helpers ready')

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'en-US,en;q=0.9',
}
RETRY_STATUSES = (429, 500, 502, 503, 504)
BRAND_PATH_HINTS = ['designer', 'designers', 'brand', 'brands', 'collections']
_SESSION = requests.Session(); _SESSION.headers.update(HEADERS)

def _http_get(url, timeout=30):
    """Takes a URL and returns the page response."""
    if USE_CURL_CFFI:
        return cffi_requests.get(url, headers=HEADERS, timeout=timeout, impersonate='chrome')
    return _SESSION.get(url, timeout=timeout)

def fetch_html(url, timeout=30, max_attempts=3):
    """Takes a URL and returns page HTML and request info."""
    meta = {'url': url, 'backend': 'curl_cffi' if USE_CURL_CFFI else 'requests'}
    for attempt in range(1, max_attempts + 1):
        meta['attempts'] = attempt
        try:
            r = _http_get(url, timeout)
            meta['status_code'] = r.status_code
            meta['ok'] = bool(getattr(r, 'ok', 200 <= r.status_code < 400))
            text = r.text or ''
            meta['content_length'] = len(text)
            if not meta['ok']:
                meta['error'] = f'HTTP {r.status_code}'
                if r.status_code in RETRY_STATUSES and attempt < max_attempts:
                    time.sleep(1.5 * attempt + random.uniform(0, 1)); continue
                return None, meta
            return text, meta
        except Exception as exc:
            meta['ok'] = False; meta['error'] = repr(exc)
            if attempt < max_attempts:
                time.sleep(1.5 * attempt + random.uniform(0, 1)); continue
            return None, meta
    return None, meta

def names_from_url_path(href):
    """Takes a link path and returns possible brand names from brand-like URLs."""
    out = []
    if not href: return out
    parts = [unquote(p) for p in urlparse(href).path.split('/') if p]
    if parts and any(h in p.lower() for p in parts for h in BRAND_PATH_HINTS):
        cand = re.sub(r'\.(html|php)$', '', re.sub(r'[-_]+', ' ', parts[-1]), flags=re.I).strip()
        if cand and not cand.isdigit():
            out.append(cand.title())
    return out

def parse_probable_brand_names(html, src):
    """Takes HTML and returns possible brand rows by reading links, scripts, and text."""
    soup = BeautifulSoup(html, 'html.parser')
    cands = []
    for a in soup.find_all('a'):
        for raw in (a.get_text(' ', strip=True), a.get('title'), a.get('aria-label')):
            nm = clean_brand_name(raw)
            if nm: cands.append(nm)
        for raw in names_from_url_path(a.get('href') or ''):
            nm = clean_brand_name(raw)
            if nm: cands.append(nm)
    text = soup.get_text('\n', strip=True)
    for script in soup.find_all('script'):
        blob = script.string or ''
        if len(blob) < 20: continue
        for raw in re.findall(r'"(?:name|brand|designerName|brandName)"\s*:\s*"([^"{}]{2,60})"', blob):
            nm = clean_brand_name(raw)
            if nm: cands.append(nm)
    for raw in re.split(r'[\n\t]+', text):
        nm = clean_brand_name(raw)
        if nm: cands.append(nm)
    rows, seen = [], set()
    for nm in cands:
        k = canonical_key(nm)
        if not k or k in seen: continue
        seen.add(k)
        rows.append({'brand_name': ALIASES.get(nm, nm), 'canonical_key': k,
                     'source_name': src['source_name'], 'source_family': src['source_family']})
    return pd.DataFrame(rows)

def brands_from_sitemap(base_url, src, locale='en-gb'):
    """Takes a sitemap URL and returns brand rows by reading designer sitemap links."""
    parsed = urlparse(base_url); root = f'{parsed.scheme}://{parsed.netloc}'
    names = set()
    idx, _ = fetch_html(f'{root}/sitemap_{locale}.xml')
    if not idx:
        return pd.DataFrame()
    for child in [u for u in re.findall(r'<loc>([^<]+)</loc>', idx) if 'azdesigners' in u.lower()]:
        ch, _ = fetch_html(child)
        if not ch: continue
        for u in re.findall(r'<loc>([^<]+)</loc>', ch):
            m = re.search(r'/designer/([^/?#]+)', u)
            if m:
                nm = clean_brand_name(re.sub(r'[-_]+', ' ', unquote(m.group(1))).title())
                if nm: names.add(nm)
    return pd.DataFrame([{'brand_name': ALIASES.get(n, n), 'canonical_key': canonical_key(n),
                          'source_name': src['source_name'], 'source_family': src['source_family']}
                         for n in sorted(names)])
print('fetch + parse helpers ready')

def query_sources(registry, sleep_s=1.5, per_host_gap=5.0, limit=None):
    """Takes the source registry and returns scraped brand rows plus a query log."""
    rows, logs, last_host = [], [], {}
    srcs = registry.head(limit) if limit else registry
    for i, (_, src) in enumerate(srcs.iterrows(), 1):
        # Some sources stay in the log but are not used for brand extraction.
        if src['source_name'] in LOG_ONLY_SOURCES:
            print(f"[{i}/{len(srcs)}] {src['source_name']} -> [logged only]", flush=True)
            logs.append({'ok': True, 'source_name': src['source_name'], 'n_names': 0, 'logged_only': True})
            continue
        host = urlparse(src['url']).netloc
        gap = per_host_gap - (time.time() - last_host.get(host, 0))
        if gap > 0: time.sleep(gap)
        sitemap = any(d in host for d in SITEMAP_SOURCES)
        print(f"[{i}/{len(srcs)}] {src['source_name']}{' [sitemap]' if sitemap else ''} -> {src['url']}", flush=True)
        if sitemap:
            df = brands_from_sitemap(src['url'], src); meta = {'ok': True}
        else:
            html, meta = fetch_html(src['url'])
            df = parse_probable_brand_names(html, src) if html else pd.DataFrame()
        last_host[host] = time.time()
        n = len(df)
        print(f"    -> {'ok' if meta.get('ok') else 'FAIL'} names={n} {meta.get('error','')}", flush=True)
        if n: rows.append(df)
        logs.append({**meta, 'source_name': src['source_name'], 'n_names': n})
        time.sleep(sleep_s + random.uniform(0, 0.5))
    cols = ['brand_name', 'canonical_key', 'source_name', 'source_family']
    live = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    return live, pd.DataFrame(logs)



# Hand-checked URLs skip the Clearbit lookup.
SEED_WEBSITES = {
    'Songmont':         'https://www.songmont.com',
    'Mardi Mercredi':   'https://mardi-mercredi.com',
    'Marge Sherwood':   'https://margesherwood.com',
    'Dragon Diffusion': 'https://www.dragondiffusion.com',
}

def resolve_website(name, timeout=10):
    """Takes a brand name and returns a likely website by asking Clearbit."""
    try:
        r = requests.get('https://autocomplete.clearbit.com/v1/companies/suggest',
                         params={'query': name}, timeout=timeout)
        if r.ok:
            d = r.json()
            if d and d[0].get('domain'):
                return 'https://' + d[0]['domain']
    except Exception:
        pass
    return ''

def _squash(s):
    """Takes text and returns only lowercase letters and numbers."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _domain_core(url):
    """Takes a URL and returns the main domain word by removing subdomains."""
    net = re.sub(r'^www\.', '', urlparse(url).netloc or url)
    p = net.split('.'); return p[0] if len(p) >= 2 else net

def name_in_domain(name, url):
    """Takes a brand name and URL and returns True when the domain matches the name."""
    n, d = _squash(name), _squash(_domain_core(url))
    if not n or not d: return False
    if len(n) < 3: return n == d
    return n in d or d in n or (len(n) >= 5 and n[:5] in d)



# Product-group signal words.
TAXONOMY = {
    'clothing':    ['clothing','ready-to-wear','ready to wear','dresses','tops','knitwear','trousers','shirts',
                    'outerwear','denim','t-shirts','coats','jackets','sweaters','womenswear','menswear','apparel',
                    'skirts','jumpsuits','blazers','pants'],
    'shoes':       ['shoes','footwear','sneakers','boots','sandals','heels','loafers','mules','trainers','pumps',
                    'espadrilles'],
    'bags':        ['handbags','totes','backpacks','purses','clutches','crossbody'],
    'jewellery':   ['jewelry','jewellery','rings','necklaces','earrings','bracelets','pendants','charms'],
    'accessories': ['belts','scarves','gloves','wallets','socks'],
    'eyewear':     ['eyewear','sunglasses','optical'],
    'swimwear':    ['swimwear','swimsuit','swimsuits','bikini','bikinis'],
    'lingerie':    ['lingerie','underwear','intimates','bras','briefs'],
}
TAX_RX = {g: re.compile(r'\b(' + '|'.join(re.escape(w) for w in ws) + r')\b', re.I) for g, ws in TAXONOMY.items()}
CAT_WORKERS, CAT_TIMEOUT, CAT_MINHITS = 16, 12, 2
# Tie-break order for one dominant group when there is no clothing.
GROUP_PRIORITY = ['shoes', 'bags', 'jewellery', 'swimwear', 'lingerie', 'eyewear', 'accessories']

def classify_site(url):
    """Takes a website URL and returns a product category by scanning page text."""
    try:
        r = _http_get(url, CAT_TIMEOUT)
        if not getattr(r, 'ok', False):
            return 'unknown'
        soup = BeautifulSoup(r.text, 'html.parser')
        title = (soup.title.string if soup.title else '') or ''
        nav = ' '.join(a.get_text(' ', strip=True) for a in soup.find_all(['a', 'button']))
        blob = f"{title} {nav} {soup.get_text(' ', strip=True)[:8000]}".lower()
        present = {g for g, rx in TAX_RX.items()
                   if len({m.group(0).lower() for m in rx.finditer(blob)}) >= CAT_MINHITS}
        if not present:
            return 'unknown'
        # Clothing-led brand with other product groups.
        if 'clothing' in present:
            return 'all' if (present & {'shoes', 'bags', 'jewellery', 'accessories'}) else 'clothes'
        # No clothing: use the dominant product group.
        for g in GROUP_PRIORITY:
            if g in present:
                return 'only ' + g
        return 'unknown'
    except Exception:
        return 'unknown'
