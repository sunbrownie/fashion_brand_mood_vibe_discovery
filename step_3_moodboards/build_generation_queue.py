#!/usr/bin/env python3
"""Build a deterministic, resumable image-generation queue from Ellina's manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "step_3_moodboards" / "helpers"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from moodboard_helpers import brands_df, moodboard_prompt, slugify  # noqa: E402


DEFAULT_MANIFEST = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
DEFAULT_OUTPUT = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "generation_queue.json"
DEFAULT_REVIEWS = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "candidate_reviews.json"
IMAGE_ACTIONS = {"full_regenerate", "targeted_regenerate", "catalog_add_and_generate"}
CATALOG_VERIFY_REGENERATE_IDS = {"ER-046", "ER-244"}


SPECIAL_PROMPT_OVERRIDES = {
    ("clothes", "A Bathing Ape"): (
        "Create a vertical six-panel clothing-only moodboard for A Bathing Ape. Show six distinct garments or "
        "fully clothed adult looks: a green ABC-camo shark hoodie with zip hood, a brown camo field jacket, a "
        "black varsity jacket, a bright graphic sweatshirt with no legible words, wide cargo trousers and a "
        "colour-block streetwear shell. Use Tokyo streetwear energy, saturated green, brown, black, red and blue. "
        "No shoes, bags, jewellery, product repeats, alternate views, captions or generated brand text."
    ),
    ("clothes", "Abercrombie & Fitch"): (
        "Create a current vertical six-panel Abercrombie & Fitch clothing moodboard focused on polished relaxed "
        "American wardrobe staples, not the old logo-heavy era. Show six distinct adult looks or garments: relaxed "
        "90s jeans, tailored wide-leg trousers, a linen shirt, soft knit cardigan, clean bomber jacket and an "
        "understated slip dress. Use warm cream, chocolate, denim blue, charcoal and olive with soft editorial "
        "light. No logos, text, fragrance, bags, footwear panels, repeated model, repeated garment or alternate view."
    ),
    ("clothes", "Adidas"): (
        "Create a vertical six-panel Adidas apparel moodboard with six distinct fully clothed adult looks or "
        "garments: classic Firebird track jacket, three-stripe track trousers, retro football jersey without words, "
        "technical running shell, minimalist training set and a heritage colour-block sweatshirt. Use black-white, "
        "royal blue, red, forest green and cream. Three-stripe construction may appear cleanly, but no generated "
        "lettering. Clothing only: no trainers, shoe panels, bags, balls, jewellery, repeated look or detail crop."
    ),
    ("clothes", "Arket"): (
        "Create a restrained vertical six-panel ARKET clothing moodboard expressing modern Nordic everyday design. "
        "Show six distinct adult looks or garments: long wool coat, brushed knit, crisp poplin shirt, relaxed denim, "
        "utility overshirt and soft unstructured tailoring. Use charcoal, navy, oatmeal, pale blue, olive and one "
        "rust accent with natural paper and Scandinavian architectural light. No logos, words, bags, footwear "
        "panels, furniture-only panels, repeated garment, repeated model or alternate crop."
    ),
    ("clothes", "AVAVAV"): (
        "Create a directional vertical six-panel AVAVAV clothing moodboard capturing satirical Milan runway energy. "
        "Show six different fully clothed adult looks or garments: exaggerated-shoulder blazer, distressed oversized "
        "hoodie, trompe-l'oeil fitted dress, deliberately crumpled tailoring, sculptural puffer and an asymmetrical "
        "deconstructed knit. Use acid green, black, grey, white and one vivid red accent with disruptive torn-paper "
        "composition. No shoes, four-toe footwear, bags, text, logos, nudity, repeated silhouette or alternate view."
    ),
    ("clothes", "Barbour"): (
        "Create a vertical six-panel Barbour clothing moodboard rooted in British countryside utility. Show six "
        "distinct adult looks or garments: olive waxed field jacket, navy quilted jacket, brown hunting coat with "
        "corduroy collar, tartan-lined raincoat, chunky lambswool knit and a practical cotton overshirt. Use olive, "
        "navy, tobacco, cream and muted tartan in rainy rural light. Clothing only; no bags, boots as hero products, "
        "dogs, guns, text, logos, repeated jacket, duplicate model or detail-only crop."
    ),
    ("clothes", "Balmain"): (
        "Create a vertical six-panel Balmain ready-to-wear moodboard centered on architectural Paris tailoring. "
        "Show six distinct fully clothed adult looks: sharp black double-breasted blazer with six gold buttons, "
        "sculpted-shoulder mini dress, crystal-embroidered evening jacket, military wool coat, structured leather "
        "jacket and clean high-waisted trouser suit. Use black, white, gold and one jewel-tone accent. Clothing only; "
        "no bags, shoes, jewellery tiles, text, repeated gold-button blazer or alternate crops."
    ),
    ("clothes", "Bandit Running"): (
        "Create a vertical six-panel Bandit Running apparel moodboard rooted in Brooklyn performance running. Show "
        "six distinct fully clothed adult running looks or garments: striped Cadence quarter tights, Superbeam half "
        "tights, Micromesh long sleeve, race singlet, technical shell and a Steady State recovery set. Use black, "
        "concrete taupe, cocoa, steel blue and safety orange in dawn city light. No shoes, hats, bottles, watches, "
        "text, race bibs, repeated athlete, repeated garment or alternate view."
    ),
    ("clothes", "Billabong"): (
        "Create a vertical six-panel Billabong clothing moodboard expressing Gold Coast surf heritage. Show six "
        "distinct garments or fully clothed adult looks: graphic surf tee without words, striped boardshorts, "
        "technical wetsuit, sun-faded crewneck, quilted folk jacket and a relaxed printed shirt. Use ocean blue, "
        "sunset orange, faded black, sand and tropical print in coastal light. No swimwear posing, surfboards, bags, "
        "shoes, text, logos, repeated product or duplicate model."
    ),
    ("clothes", "Bimba Y Lola"): (
        "Create a playful vertical six-panel Bimba y Lola clothing moodboard with bold Spanish colour and surreal "
        "graphic energy. Show six distinct fully clothed adult looks or garments: oversized technical puffer, vivid "
        "printed midi dress, colour-block knit, sculptural denim jacket, bright tailored suit and a patterned mesh "
        "layering top over an opaque base. Use cobalt, lime, red, pink and black. No handbags, shoes, jewellery, text, "
        "logos, repeated print, repeated model or alternate view."
    ),
    ("clothes", "Burton"): (
        "Create a vertical six-panel Burton clothing moodboard focused only on snowboard apparel. Show six distinct "
        "fully clothed adult looks or garments: technical shell jacket, insulated bib trousers, colour-block anorak, "
        "fleece midlayer, merino base-layer set and relaxed après hoodie. Use black, snow white, cobalt, rust, forest "
        "green and one neon accent against alpine weather. No snowboards, boots, goggles, helmets, bags, text, logos, "
        "repeated outfit or alternate crop."
    ),
    ("clothes", "Burberry"): (
        "Create a vertical six-panel Burberry clothing moodboard balancing British heritage and modern London edge. "
        "Show six distinct fully clothed adult looks or garments: honey gabardine trench, check-lined raincoat, sharp "
        "black tailoring, tartan wool skirt look, cavalry-inspired coat and a contemporary draped knit. Use honey, "
        "black, red, cream and controlled Burberry-style check. No bags, scarves as hero tiles, shoes, text, logos, "
        "repeated trench colourway, repeated model or detail-only crop."
    ),
    ("clothes", "Calvin Klein / Calvin Klein Jeans"): (
        "Create a vertical six-panel Calvin Klein and Calvin Klein Jeans clothing moodboard expressing clean New "
        "York minimalism and current relaxed denim. Show six distinct fully clothed adult looks or garments: a "
        "washed-blue 90s straight jean look, low-rise baggy denim, denim trucker jacket, crisp white tee-and-trouser "
        "look, lean black tailoring and a restrained black technical outerwear look. Use indigo, pale blue, white, "
        "black and concrete grey with spare studio light. No underwear-only imagery, nudity, fragrance, bags, shoes, "
        "text, logos, repeated denim wash, repeated model or alternate crop."
    ),
    ("clothes", "Canada Goose"): (
        "Create a vertical six-panel Canada Goose clothing moodboard focused on protective outerwear and modern "
        "layering. Show six distinct fully clothed adult looks or garments: an expedition parka, lightweight down "
        "jacket, waterproof rain shell, HyBridge-style knit jacket, technical fleece and a clean windwear anorak. "
        "Use black, snow white, navy, deep red and lichen green in cold urban and alpine light. No bags, footwear "
        "tiles, gloves or hats as hero products, fur-heavy vintage styling, text, logos, repeated parka or alternate view."
    ),
    ("clothes", "Carhartt WIP"): (
        "Create a vertical six-panel Carhartt WIP clothing moodboard rooted in European streetwear and hard-wearing "
        "workwear icons. Show six distinct fully clothed adult looks or garments: a cropped Detroit jacket with "
        "corduroy collar, Michigan chore coat, reinforced Double Knee trousers, hooded Active jacket, loose cargo "
        "trousers and a sturdy overshirt. Use Hamilton brown, tobacco, black, dark navy, olive and worn denim with "
        "visible canvas grain and triple-stitch construction. No tools, bags, shoes, caps, text, logos, repeated "
        "jacket, repeated model or detail-only crop."
    ),
    ("clothes", "Charles Jeffrey Loverboy"): (
        "Create a vertical six-panel Charles Jeffrey LOVERBOY clothing moodboard with joyful queer London club energy "
        "and Scottish punk craft. Show six distinct fully clothed adult looks or garments: a red-tartan tailored look, "
        "scarf-panel denim jacket, exuberant multicolour collage knit, knitted puffer, exaggerated striped shirt and "
        "polka-dot cardigan. Use red, cobalt, acid green, purple, black and tartan with handmade patching and theatrical "
        "editorial light. No hats, balaclavas, bags, shoes, text, logos, repeated tartan look or alternate crop."
    ),
    ("clothes", "Comme des Garcons Play"): (
        "Create a vertical six-panel Comme des Garcons PLAY clothing moodboard focused on the approachable graphic "
        "wardrobe line. Show six distinct garments or fully clothed adult looks: black V-neck cardigan, navy-and-white "
        "Breton striped long sleeve, grey crew-neck sweater, black varsity jacket, clean white tee and a black hoodie. "
        "Use red, black and pink heart-with-eyes motifs sparingly and accurately as small chest details, with black, "
        "white, navy, grey and camel. No fake lettering, extra typography, bags, shoes, jewellery, repeated heart "
        "placement, repeated garment or alternate view."
    ),
    ("clothes", "Conner Ives"): (
        "Create a vertical six-panel Conner Ives clothing moodboard expressing resourceful London-made Americana and "
        "upcycled glamour. Show six distinct fully clothed adult looks or garments: a reconstituted shirred T-shirt, "
        "bias-cut T-shirt maxi dress without legible words, recycled-spandex shell-belt dress, sequin jersey polo "
        "dress, scarf-panel mini dress and a reconstructed baseball-jersey look. Use saturated red, cobalt, gold, "
        "black, cream and vintage mixed prints with visible reclaimed construction. No loose belts or accessories as "
        "hero tiles, bags, shoes, readable slogans, logos, repeated model, repeated jersey or alternate crop."
    ),
    ("clothes", "Diesel"): (
        "Create a vertical six-panel Diesel clothing moodboard centered on rebellious Italian denim innovation. Show "
        "six distinct fully clothed adult looks or garments: extra-wide low-rise jeans, barrel-leg jeans, cropped "
        "trucker jacket, microstone devore denim dress, cable-knit-effect denim jacket and a clean black moto-denim "
        "look. Use indigo, acid-wash blue, dirty black, rust and one red accent with tactile abrasion and ironic "
        "editorial polish. No bags, shoes, belts as hero products, underwear, text, logos, repeated wash, repeated "
        "silhouette, repeated model or alternate crop."
    ),
    ("clothes", "Desigual"): (
        "Create a vertical six-panel Desigual clothing moodboard expressing colourful Barcelona patchwork and arty "
        "urban eclecticism. Show six distinct fully clothed adult looks or garments: scrap-denim patchwork jacket, "
        "kaleidoscopic midi dress, hybrid hoodie-trucker jacket, tapestry-knit sweater, mixed-print wide trousers and "
        "a bright asymmetrical shirt dress. Use red, cobalt, orange, turquoise, black and mixed florals with craftcore "
        "texture. No licensed characters, bags, shoes, text, logos, repeated patchwork, repeated model or alternate view."
    ),
    ("clothes", "Duran Lantink"): (
        "Create a vertical six-panel Duran Lantink clothing moodboard focused on reconstruction and surreal sculpted "
        "volume. Show six distinct fully clothed adult runway looks: oversized faux-suede coat, sculptured tartan mini "
        "dress over an opaque base, over-the-shoulders leather bomber, padded jersey trouser look, double-waisted skirt "
        "with tailored jacket and a tapered animal-print coat. Use camel, black, orange, purple, tartan and controlled "
        "zebra or leopard with exaggerated but wearable three-dimensional forms. No body-print illusion, nudity, "
        "lingerie, bags, shoes, text, logos, repeated padded anatomy, repeated model or alternate view."
    ),
    ("clothes", "Erdem"): (
        "Create a vertical six-panel Erdem clothing moodboard expressing romantic historical research through modern "
        "London craft. Show six distinct fully clothed adult looks or garments: rose-print draped midi dress, trailing-"
        "floral embroidered blazer, textured-satin evening gown, floral jacquard cardigan-and-skirt, structured cropped "
        "wool jacket with threadwork and a crisp floral shirt dress. Use forest green, ivory, rose, black, acid yellow "
        "and deep burgundy with refined embroidery and painterly botanicals. No bags, brooch-only tiles, shoes, text, "
        "logos, repeated floral layout, repeated model or alternate crop."
    ),
    ("clothes", "GAP"): (
        "Create a vertical six-panel Gap clothing moodboard expressing democratic American casual classics. Show six "
        "distinct fully clothed adult looks or garments: relaxed 90s loose jeans, modern khakis with Oxford shirt, "
        "Vintage Soft hoodie, denim utility overshirt, cable-knit crewneck and a clean wool car coat. Use mid-blue "
        "denim, white, navy, heather grey, khaki and red with natural daylight and inclusive everyday casting. No "
        "bags, shoes, caps, text, logos, repeated denim outfit, repeated model or alternate view."
    ),
    ("clothes", "Gant"): (
        "Create a vertical six-panel GANT clothing moodboard rooted in East Coast American sportswear and collegiate "
        "prep. Show six distinct fully clothed adult looks or garments: striped Oxford button-down, navy cable-knit "
        "sweater, red-and-navy varsity jacket, cotton-twill chinos, striped heavy rugger and a classic wool club blazer. "
        "Use navy, cream, Yale blue, oxblood red and khaki with crisp campus and studio light. No ties, bags, caps, "
        "shoes, text, crests, logos, repeated shirt, repeated model or alternate crop."
    ),
    ("clothes", "Good American"): (
        "Create a vertical six-panel Good American clothing moodboard centered on inclusive, body-confident denim. "
        "Show six distinct fully clothed adult looks across visibly varied body types: Always Fits straight jeans, "
        "Good Waist palazzo jeans, Good Legs skinny jeans, vintage wide jeans, classic bootcut jeans and a polished "
        "denim-and-oversized-blazer look. Use indigo, pale wash, black denim, white and warm brown with clean Los "
        "Angeles studio light. No bodysuit-only or underwear styling, bags, shoes, text, logos, repeated wash, repeated "
        "body type, repeated model or alternate view."
    ),
    ("clothes", "Gymshark"): (
        "Create a vertical six-panel Gymshark apparel moodboard focused on functional training rather than lifestyle "
        "merchandise. Show six distinct fully clothed adult athletic looks: Vital seamless training set, Adapt Fleck "
        "lifting set, mens stringer with training joggers, technical running shell with shorts over tights, relaxed "
        "rest-day hoodie-and-jogger set and a lightweight studio layer. Use black marl, mineral grey, deep blue, muted "
        "purple, green and one orange accent. No equipment, water bottles, shoes as hero products, exposed torso, "
        "sports-bra-only styling, text, logos, repeated pose, repeated set or alternate crop."
    ),
    ("clothes", "Gucci"): (
        "Create a vertical six-panel Gucci ready-to-wear moodboard balancing Florentine archival codes with modern "
        "eclectic polish. Show six distinct fully clothed adult looks: Horsebit silk-jacquard dress, GG-canvas jacket "
        "with midi skirt, printed silk shirt-and-trouser set, checked fine-wool skirt suit, 1970s velvet tailoring and "
        "a restrained green-red Web-trim knit look. Use deep brown, oxblood, bottle green, cream, gold and controlled "
        "monogram. No handbags, shoes, jewellery tiles, text, large logos, repeated monogram, repeated model or alternate view."
    ),
    ("clothes", "Hollister"): (
        "Create a vertical six-panel Hollister clothing moodboard expressing relaxed current California youth style. "
        "Show six distinct fully clothed adult looks: super-baggy jeans with plain tee, boxy washed hoodie, relaxed "
        "thermal Henley, brushed flannel overshirt, all-weather windbreaker and low-rise wide-leg trousers with fitted "
        "cotton top. Use mid-wash denim, heather grey, navy, faded red, cream and olive in warm coastal daylight. Adult "
        "models only. No swimwear, bags, fragrance, graphic words, logos, repeated denim look, repeated model or alternate crop."
    ),
    ("clothes", "Jaded London"): (
        "Create a vertical six-panel Jaded London clothing moodboard with reworked Y2K streetwear and directional club "
        "energy. Show six distinct fully clothed adult looks: mirror-detailed loose jeans, nylon balloon parachute pants, "
        "washed-satin bootcut cargos, cargo corset layered over an opaque top, ruched mesh midi skirt with covered top "
        "and a faux-fur knit cardigan look. Use black, khaki, ecru, metallic silver, washed blue and one acid accent. "
        "No bags, shoes, lingerie-only styling, exposed torso, text, logos, repeated cargo shape, repeated model or alternate view."
    ),
    ("clothes", "Jean Paul Gaultier"): (
        "Create a vertical six-panel Jean Paul Gaultier clothing moodboard celebrating iconic Paris irreverence with "
        "fully covered styling. Show six distinct adult looks: crystal-striped Breton sailor top, sculptural cone-bust "
        "jacket over an opaque high-neck base, tattoo-inspired mesh layered over black lining, corset-seamed tailored "
        "dress, pinstriped kilt-suit and a sharply sliced trench. Use navy, white, black, red and metallic silver with "
        "theatrical atelier light. No nudity, skin-print illusion, lingerie-only styling, bags, fragrance, shoes, text, "
        "logos, repeated stripes, repeated model or alternate crop."
    ),
    ("clothes", "Karl Lagerfeld"): (
        "Create a vertical six-panel KARL LAGERFELD clothing moodboard expressing monochrome Parisian tailoring with "
        "contemporary graphic edge. Show six distinct fully clothed adult looks: sharp black double-breasted suit, "
        "white shirt with dramatic collar, black-and-white boucle jacket set, zip-detail tailored blazer, streamlined "
        "biker jacket look and a transformable sequin evening dress. Use black, optic white, silver and one deep red "
        "accent. No sunglasses-only tiles, bags, shoes, cartoon portraits, text, logos, repeated blazer or alternate view."
    ),
    ("clothes", "Lacoste"): (
        "Create a vertical six-panel Lacoste clothing moodboard translating French tennis heritage into modern sport "
        "style. Show six distinct fully clothed adult looks: classic white pique polo, navy piped tracksuit, cable-knit "
        "tennis sweater, rain-proof technical tracksuit, hybrid track-jacket shirt and a pleated trench-skirt look. Use "
        "white, navy, court green, clay red and pale yellow with crisp club light. No racquets, balls, bags, caps, shoes "
        "as hero products, text, large crocodile graphics, repeated polo, repeated model or alternate crop."
    ),
    ("clothes", "Levi's"): (
        "Create a vertical six-panel Levi's clothing moodboard focused on authentic denim icons and current fits. Show "
        "six distinct fully clothed adult looks: rigid-blue 501 Original jeans, Ribcage wide-leg jeans, XL baggy straight "
        "jeans, 90s Trucker jacket, Western bootcut denim look and a dark-wash denim workshirt look. Use rigid indigo, "
        "light wash, black denim, ecru and one red accent with honest cotton texture. No bags, hats, shoes as hero "
        "products, text, red-tab closeups, logos, repeated wash, repeated model or alternate view."
    ),
    ("clothes", "Loro Piana"): (
        "Create a vertical six-panel Loro Piana clothing moodboard expressing tactile Italian quiet luxury through "
        "precious fibres and relaxed layers. Show six distinct fully clothed adult looks: navy Cashmere Storm System "
        "coat, camel cashmere crewneck with tailored trousers, elongated wool-cashmere car coat, CashDenim jeans with "
        "soft knit, calfskin Maremma bomber and an understated Sopra Visso wool cardigan look. Use navy, camel, oatmeal, "
        "umbrella-pine green and cream in soft natural light. No bags, shoes, hats, text, logos, generic beige-only "
        "styling, repeated coat, repeated model or alternate crop."
    ),
    ("clothes", "Lululemon"): (
        "Create a vertical six-panel lululemon apparel moodboard covering performance and all-day movement. Show six "
        "distinct fully clothed adult looks: Align high-rise pant with waist-length top, Define jacket and training pant, "
        "Wunder Train set with full-length top, oversized Scuba hoodie with straight-leg pant, mens Pace Breaker running "
        "look and ABC trouser with technical polo. Use black, true navy, cypress green, dusty blue, warm grey and one "
        "berry accent. No sports-bra-only styling, bags, yoga mats, bottles, shoes as hero products, text, logos, repeated "
        "leggings set, repeated model or alternate crop."
    ),
    ("clothes", "Mango"): (
        "Create a vertical six-panel Mango clothing moodboard expressing modern Barcelona high-street polish. Show six "
        "distinct fully clothed adult looks: straight linen-blend blazer with wide trousers, A-line linen dress, oversized "
        "cotton jacket with straight pants, satin gathered midi dress, high-waisted mom jeans with crisp poplin shirt and "
        "a lyocell bomber look. Use black, off-white, burnt orange, medium blue, pale pink and chocolate with warm city "
        "light. No bags, shoes, jewellery tiles, text, logos, repeated tailoring, repeated model or alternate view."
    ),
    ("clothes", "Martine Rose"): (
        "Create a vertical six-panel Martine Rose clothing moodboard rooted in London subculture, warped proportions "
        "and football style. Show six distinct fully clothed adult looks: oversized panelled football top, lace-trim "
        "football jersey layered over a tee, shrunken tailored jacket with wide trousers, twisted-sleeve sports top, "
        "oversized lime-striped polo and a red-yellow cut overshirt look. Use navy, light blue, lime, black, burgundy and "
        "red with raw street-casting energy. No shoes, bags, caps, readable sponsor text, logos, repeated jersey, repeated "
        "model or alternate crop."
    ),
    ("clothes", "Missoni"): (
        "Create a vertical six-panel Missoni clothing moodboard celebrating Italian knitwear colour and rhythm. Show "
        "six distinct fully clothed adult looks: multicolour zigzag maxi dress, tone-on-tone white raschel cardigan, "
        "pink-rust chevron cardigan with skirt, blue-green macro-zigzag mens cardigan, lightweight striped knit trouser "
        "set and a refined chevron polo mini dress over opaque tights. Use saturated orange, violet, turquoise, pink, "
        "green and cream. No swimwear, bags, shoes, text, logos, repeated zigzag scale, repeated model or alternate view."
    ),
    ("clothes", "Michael Kors"): (
        "Create a vertical six-panel Michael Kors clothing moodboard expressing polished New York jet-set sportswear. "
        "Show six distinct fully clothed adult looks: camel belted coat with tailoring, black sequined knit dress, crisp "
        "white trouser suit, printed satin-crepe dress, striped knit polo dress and an olive utility jacket with wide "
        "trousers. Use camel, black, white, gold, olive and one cobalt accent with clean city light. No handbags, shoes, "
        "watches, jewellery tiles, text, logos, repeated gold hardware, repeated model or alternate crop."
    ),
    ("clothes", "Missguided"): (
        "Create a vertical six-panel Missguided clothing moodboard expressing confident trend-led British high-street "
        "style. Show six distinct fully clothed adult looks: oversized blazer with wide trousers, washed baggy denim set, "
        "satin midi dress, utility cargo trousers with fitted long-sleeve top, faux-leather biker jacket look and a bold "
        "colour-block knit co-ord. Use black, charcoal, mid-blue denim, chocolate, hot pink and lime. No lingerie-only "
        "styling, exposed torso, bags, shoes, text, logos, repeated bodycon dress, repeated model or alternate view."
    ),
    ("clothes", "McQueen"): (
        "Create a vertical six-panel McQueen clothing moodboard focused on razor-sharp tailoring and sculptural romance. "
        "Show six distinct fully clothed adult looks: folded-lapel navy jacket with pencil skirt, black corset midi dress, "
        "grey flecked-pinstripe suit, metallic-tweed high-low coat, black floral-embroidery evening dress and an ivory "
        "shoulder-bow blouse with tailored trousers. Use black, navy, ivory, silver, blood red and one floral accent with "
        "dramatic atelier light. No bags, shoes, skull-logo tiles, text, logos, repeated corset, repeated model or alternate crop."
    ),
    ("clothes", "MKI MIYUKI ZOKU"): (
        "Create a vertical six-panel MKI MIYUKI ZOKU clothing moodboard emphasizing contemporary British menswear, "
        "fabric and relaxed fit. Show six distinct fully clothed adult looks: cropped leather racer jacket, boxy twisted-"
        "weave shirt, linen double-breasted suit with pleated wide trousers, garment-dyed nylon track set, hyperlight "
        "hooded jacket and a mohair-blend crewneck with wide-leg joggers. Use black, mushroom, bone, charcoal, mint and "
        "deep olive with clean Leeds-studio light. No bags, shoes, caps, text, logos, repeated track set, repeated model "
        "or alternate view."
    ),
    ("clothes", "Moncler"): (
        "Create a vertical six-panel Moncler clothing moodboard balancing refined city down with Grenoble alpine "
        "performance. Show six distinct fully clothed adult looks or garments: glossy short Maya-style puffer, "
        "colour-block Grenoble ski shell with technical bib trousers, lightweight quilted field jacket, down-front "
        "knit cardigan, long belted down coat and a clean waterproof hooded shell. Use black, optical white, navy, "
        "lacquer red and one ice-blue accent in crisp mountain and studio light. No bags, boots, goggles, helmets, "
        "text, logos, repeated black puffer, repeated model or alternate crop."
    ),
    ("clothes", "Moschino"): (
        "Create a vertical six-panel Moschino clothing moodboard expressing playful Milan surrealism and polished "
        "camp. Show six distinct fully clothed adult looks: trompe-l'oeil tailored jacket, colourful chain-print silk "
        "shirt with trousers, couture-inspired black biker dress, bright heart-motif knit look, cloud-print poplin "
        "shirt with shorts and an exaggerated red evening coat. Use red, black, white, gold, sky blue and hot pink "
        "with witty but wearable construction. No bags, shoes, perfume, readable slogans, logos, repeated chain or "
        "heart motif, repeated model or alternate view."
    ),
    ("clothes", "Mugler"): (
        "Create a vertical six-panel Mugler ready-to-wear moodboard centered on sculpted body architecture and sharp "
        "Paris tailoring. Show six distinct fully clothed adult looks: spiral-seamed black jacket with trousers, "
        "sculpted-shoulder cobalt suit, opaque-lined illusion-mesh midi dress, corset-seamed red dress, cutaway white "
        "tailored coat over a covered base and an asymmetric black knit dress. Use black, cobalt, white, deep red and "
        "silver with precise dramatic light. No nudity, exposed torso, lingerie-only styling, bags, shoes, fragrance, "
        "text, logos, repeated cutout treatment, repeated model or alternate crop."
    ),
    ("clothes", "Napapijri"): (
        "Create a vertical six-panel Napapijri clothing moodboard rooted in colourful expedition utility. Show six "
        "distinct fully clothed adult looks or garments: Rainforest pullover anorak, archival Skidoo-style anorak, "
        "high-pile zip fleece, shiny technical puffer, rugged utility overshirt with cargo trousers and a relaxed "
        "archive-knit look. Use orange, navy, forest green, cream, black and bright blue with outdoor and clean studio "
        "light. No bags, hats, boots, flags, text, logos, repeated anorak colourway, repeated model or alternate crop."
    ),
    ("clothes", "Needles"): (
        "Create a vertical six-panel Needles clothing moodboard expressing Japanese-Americana eccentricity and relaxed "
        "craft. Show six distinct fully clothed adult looks or garments: purple side-stripe track jacket, wide H.D. "
        "track trousers with a plain top, snowflake mohair cardigan, reconstructed seven-cut flannel shirt, snap-front "
        "Western shirt and a paisley-jacquard relaxed suit. Use purple, black, olive, rust, cream and teal with tactile "
        "velour, mohair and patchwork. A small butterfly motif may appear once only. No bags, shoes, hats, text, logos, "
        "repeated track set, repeated model or alternate view."
    ),
    ("clothes", "Never Fully Dressed"): (
        "Create a vertical six-panel Never Fully Dressed clothing moodboard celebrating inclusive, feel-good London "
        "maximalism. Show six distinct fully clothed adult looks across visibly varied body types: reversible animal-"
        "print wrap dress, colourful abstract-print shirt-and-trouser co-ord, floral wrap midi dress, bright knit "
        "cardigan-and-wide-leg-trouser set, elevated pink tailored suit and a playful fruit-print skirt with a plain "
        "top. Use joyful pink, orange, cobalt, green, chocolate and cream. Every print must be different. No bags, "
        "shoes, jewellery tiles, text, logos, repeated print, repeated body type, repeated model or alternate crop."
    ),
    ("clothes", "Neighbourhood"): (
        "Create a vertical six-panel NEIGHBORHOOD clothing moodboard rooted in Tokyo motorcycle, military and workwear "
        "culture. Show six distinct fully clothed adult looks or garments: embroidered souvenir jacket, Savage-wash "
        "Type-1 denim jacket, coated M-43 field jacket, washed duck work jacket, wide utility cargo trousers with a "
        "plain tee and a reflective technical shell look. Use black, indigo, olive, charcoal, tobacco and one silver "
        "accent with heavy-duty texture. No motorcycles, incense objects, bags, shoes, hats, text, logos, repeated "
        "jacket construction, repeated model or alternate crop."
    ),
    ("clothes", "New Balance"): (
        "Create a vertical six-panel New Balance apparel moodboard balancing performance and relaxed sportswear. Show "
        "six distinct fully clothed adult looks: reflective RC running jacket with tights, Trackside woven jacket and "
        "pants, Made-in-USA core fleece hoodie look, quilted lifestyle bomber, basketball padded vest layered over a "
        "tee and woven cargo trousers with a clean sweatshirt. Use navy, athletic grey, black, cherry red, sport green "
        "and cream. Clothing only; no trainers, shoes, bags, caps, text, logos, repeated tracksuit, repeated model or "
        "alternate view."
    ),
    ("clothes", "Nike"): (
        "Create a vertical six-panel Nike apparel moodboard spanning modern sport and Sportswear icons. Show six "
        "distinct fully clothed adult looks: sculpted Tech Fleece hoodie and joggers, ACG storm shell with trail pants, "
        "Dri-FIT running jacket with shorts over tights, crisp tennis dress with coverage shorts, loose basketball "
        "warm-up set and a colour-block Windrunner with woven trousers. Use black, white, volt, royal blue, orange and "
        "earthy ACG brown. Clothing only; no shoes, trainers, bags, balls, caps, text, logos, repeated swooshes, repeated "
        "model or alternate crop."
    ),
    ("clothes", "No Problemo"): (
        "Create a vertical six-panel No Problemo clothing moodboard with playful London streetwear and retro sci-fi "
        "outsider energy. Make a full short-sleeve T-shirt a prominent hero product, shown clearly from neckline to hem, "
        "with the exact readable NO PROBLEMO chest mark. Also show a distinct black NO PROBLEMO crewneck sweatshirt, "
        "striped long-sleeve tee, forest ripstop workwear, technical outerwear and a silver down puffer. Use black, "
        "washed grey, forest green, cream, silver and restrained fluorescent yellow. No hats, shoes, bags, repeated "
        "garments, repeated people, alternate crops or lettering beyond the exact NO PROBLEMO product mark."
    ),
    ("clothes", "Oner Active"): (
        "Create a vertical six-panel Oner Active womenswear moodboard focused on confident strength training and "
        "versatile movement. Show six distinct fully clothed adult athletic looks across varied body types: Effortless "
        "seamless leggings with a waist-length training top, UnifiedMove jacket with pocket leggings, SoftMotion long-"
        "sleeve Pilates set, Timeless square-neck vest with full-length trousers, relaxed midweight sweatshirt and "
        "joggers, and technical shorts layered over fitted training tights. Use black, cocoa, earth green, granite blue, "
        "rosewood and charged pink. No sports-bra-only styling, exposed torso, bags, shoes, equipment, text, logos, "
        "repeated leggings colour, repeated model or alternate crop."
    ),
    ("clothes", "Open YY"): (
        "Create a vertical six-panel OPEN YY clothing moodboard expressing experimental Seoul casual chic. Show six "
        "distinct fully clothed adult looks: brown unstructured out-pocket jacket, khaki belted cargo field jacket, "
        "constructed white dress shirt with asymmetric trousers, ivory open-back pointelle knit over an opaque base, "
        "deconstructed layered skirt look and a sculptural red draped dress. Use brown, khaki, ivory, charcoal, red and "
        "powder blue with clean contemporary styling. No bags, shoes, hats, exposed back or torso, text, logos, repeated "
        "layering device, repeated model or alternate view."
    ),
    ("clothes", "Palm Angels"): (
        "Create a vertical six-panel Palm Angels clothing moodboard combining Los Angeles skate attitude with Milan "
        "luxury streetwear. Show six distinct fully clothed adult looks: black-white classic track jacket and pants, "
        "purple velour track set, palm-print camp shirt with plain trousers, flame-motif knit, distressed denim jacket "
        "look and a dark-grey varsity jacket with wide trousers. Use black, white, purple, sunset orange, washed blue "
        "and dark grey. No readable chest or back lettering, logos, bags, shoes, sunglasses, repeated track colourway, "
        "repeated model or alternate crop."
    ),
    ("clothes", "Paul Smith"): (
        "Create a vertical six-panel Paul Smith clothing moodboard expressing British tailoring with witty colour. "
        "Show six distinct fully clothed adult looks: louche inside-out navy suit, wide-shouldered double-breasted check "
        "suit, painterly floral shirt with dark trousers, bright signature-stripe knit, colour-panelled wool blazer look "
        "and a softly washed herringbone coat. Use inky navy, burgundy, forest green, mustard, cobalt and controlled "
        "rainbow accents. No bags, shoes, ties as hero items, text, logos, repeated stripe, repeated model or alternate view."
    ),
    ("clothes", "Polo Ralph Lauren"): (
        "Create a vertical six-panel Polo Ralph Lauren clothing moodboard celebrating polished American prep. Show six "
        "distinct fully clothed adult looks: blue Oxford shirt with chinos, cream cable-knit sweater with tailored "
        "trousers, navy club blazer look, rugby-striped shirt with denim, suede field jacket with corduroy trousers and "
        "a heritage tartan wool coat. Use navy, cream, camel, forest green, oxblood and Yale blue. No large polo-player "
        "marks, text, logos, bags, shoes, ties as hero items, repeated knit, repeated model or alternate crop."
    ),
    ("clothes", "Puma"): (
        "Create a vertical six-panel PUMA apparel moodboard centered on T7 heritage and current sport style. Show six "
        "distinct fully clothed adult looks: black-white classic T7 track set, relaxed garnet tartan T7 jacket look, "
        "earth-green cropped T7 jacket with straight trousers, technical running shell with tights, loose basketball "
        "warm-up set and a PUMATECH utility jacket with four-way-stretch pants. Use black, buttercream, garnet, Persian "
        "blue, earthy green and one red accent. No cat marks, logos, text, trainers, balls, bags, repeated T7 colourway, "
        "repeated model or alternate view."
    ),
    ("clothes", "P.E Nation"): (
        "Create a vertical six-panel P.E Nation womenswear moodboard fusing Australian performance activewear with a "
        "street-sport edge. Show six distinct fully clothed adult looks across varied body types: black Baseline leggings "
        "with covered long-sleeve top, colour-block running jacket with track pants, Studio Soft Pilates top with flared "
        "pants, oversized lifestyle sweatshirt with loose trousers, Active Lite shell with shorts over tights and a "
        "relaxed knit-and-cargo travel look. Use black, cream, cobalt, rust, khaki and soft grey. No sports-bra-only "
        "styling, exposed torso, equipment, bags, shoes, text, logos, repeated leggings set, repeated model or alternate crop."
    ),
    ("clothes", "Pull & Bear"): (
        "Create a vertical six-panel Pull & Bear clothing moodboard expressing current relaxed European high-street "
        "style. Show six distinct fully clothed adult looks: baggy dark-wash jeans with fitted knit, chocolate faux-"
        "leather bomber, preppy pleated midi skirt with shirt and cardigan, wide tailored trousers with relaxed knit, "
        "oversized washed-denim jacket and a burgundy sweatshirt with loose joggers. Use dark denim, chocolate, burgundy, "
        "cream, charcoal and forest green. No bags, shoes, belts or hats as hero items, text, logos, repeated denim look, "
        "repeated model or alternate view."
    ),
    ("clothes", "Herschel"): (
        "Create a vertical editorial clothing moodboard for Herschel Supply apparel with exactly six distinct "
        "garment-led panels: a weather-resistant shell jacket, a fleece layer, a relaxed cotton sweatshirt, a "
        "clean everyday tee, utility trousers and a quilted overshirt. Use fully clothed adult models, hangers or "
        "product flat lays in black, navy, khaki, cream and one muted seasonal colour. The direction is practical "
        "West Coast layering for everyday movement. Do not show backpacks, luggage, duffles, caps, wallets or any "
        "other accessories. No repeated garment, repeated person, alternate view, text, labels or invented logos."
    ),
    ("bags", "Collina Strada"): (
        "Create a playful vertical product-only Collina Strada bag moodboard with exactly six distinct complete "
        "bags: a black Wave Knot bag, chocolate painted-plaid baguette, black Mini Mist bag, sculptural cloud bag, "
        "floral organza pouch and a colourful deadstock-textile shoulder bag. Give every panel a different "
        "silhouette and construction while expressing the brand's climate-aware, humorous New York maximalism. "
        "No people, worn views, shoes, clothing, repeated colourways, repeated products, detail-only crops, text, "
        "labels or logos."
    ),
    ("clothes", "Tkees"): (
        "Create a vertical editorial clothing moodboard for Tkees apparel. Focus only on relaxed adult "
        "casualwear: joggers, soft sweatshirts, fitted tanks, simple tees and coordinated lounge sets in "
        "white, black, heather grey, oatmeal and muted earth tones. Show six to nine distinct clothing "
        "fragments using fully clothed adult models, mannequins, folded garments and fabric details. "
        "Do not show footwear, sandals, flip-flops, bags, jewellery, swimwear, logos or text. Every garment "
        "must be different; no repeated pose, crop, colourway or alternate view of the same item. Use a "
        "clean tactile torn-paper magazine collage with varied panels and deliberate negative space."
    ),
    ("jewellery", "Shrimps"): (
        "Create a vertical product-only editorial jewellery moodboard for Shrimps. Use exactly six torn-paper "
        "panels, each containing one different pair of earrings and no other jewellery: diamanté shrimp drops, "
        "bee-and-pearl drops, sculptural gold-and-pearl drops, green enamel floral drops, white floral pearl "
        "drops, and one playful gold sculptural pair. Show every pair exactly once, never worn and never repeated. "
        "No people, necklaces, rings, bracelets, blue, words, labels or logos. Use pink, ivory, gold and small "
        "green accents with tactile art-school romantic styling and clear separation between panels."
    ),
    ("jewellery", "Simon Miller"): (
        "Create a vertical product-only editorial jewellery moodboard for Simon Miller using exactly six panels. "
        "Each panel contains one different statement earring pair shown exactly once: green palm, orange starfish, "
        "yellow banana, gummy-green raffia, natural raffia, and black-natural raffia fringe. No people, worn views, "
        "rings, necklaces, cuffs, pearls, plain hoops, repeated products, text, labels or logos. Use vivid Los "
        "Angeles sunlight, pale stone and crisp torn-paper shapes without adding extra panels."
    ),
    ("jewellery", "Sofie Schnoor"): (
        "Create a restrained six-panel vertical jewellery moodboard for Sofie Schnoor with no people. Use exactly "
        "three product panels: one sculptural gold earring pair, a second visibly different gold earring pair, and "
        "one refined gold pendant necklace. Show each product once only. The other three panels are non-product "
        "brand context: raw black leather texture, warm limestone, and Copenhagen architectural shadow. Do not "
        "invent rings, pearls, enamel, charms, bracelets or extra jewellery. No text, labels, logos or repeated "
        "objects; express clean Scandinavian restraint with a raw-yet-feminine edge."
    ),
    ("jewellery", "Tabayer"): (
        "Create a vertical product-only fine-jewellery moodboard for Tabayer with exactly six torn-paper panels. "
        "Show six different Oera-family designs, each once: a coiled-knot hoop pair, sculptural amulet pendant, "
        "architectural cuff, interlocking signet ring, diamond-pavé knot earrings and a hard-stone Oera pendant. "
        "An earring pair means exactly two earrings total in that panel, never four. Use Fairmined yellow and white "
        "gold, restrained diamond pavé and at most one deep green stone. No people, "
        "worn views, repeated motifs as alternate angles, oversized generic molten shapes, text, labels or logos."
    ),
    ("jewellery", "Tecovas"): (
        "Create a vertical product-only Western jewellery moodboard for Tecovas with exactly six panels and one "
        "different design per panel: engraved gold hoop pair, silver rope-chain necklace, compass-star signet ring, "
        "small pearl-drop pair, engraved silver cuff and a restrained horseshoe pendant. Keep the language rugged, "
        "unisex and Texas-inspired with leather and weathered-stone backdrops. No people, loose pearls, scenery-only "
        "panels, repeated products, alternate views, text, labels or logos."
    ),
    ("jewellery", "THOMAS SABO"): (
        "Create a vertical product-only THOMAS SABO moodboard with exactly six panels and one unique product per "
        "panel: a sterling-silver charm bracelet, a gold-plated charm necklace, a blackened-silver Rebel At Heart "
        "ring, a zirconia hoop pair, a freshwater-pearl pendant and a silver Connect chain. Each piece must use a "
        "different symbolic vocabulary; do not repeat moons, stars, hearts, stones or charms between panels. No "
        "people, worn views, loose gemstones, text, brand name, labels or logos."
    ),
    ("jewellery", "Toteme"): (
        "Create a vertical product-only TOTEME jewellery moodboard with exactly six panels and one unique piece "
        "per panel: black-and-gold Signature hoops, Rope enamel drop earrings, a black-and-gold Signature enamel "
        "bangle, a restrained 24k gold-plated chain necklace, an onyx gold-plated pendant necklace, and a slim "
        "diamond midi signet ring. Use minimal Stockholm styling in black, ecru, silver and gold. No people, worn "
        "views, repeated products, loose pearls, sculpture-only panels, words, labels or logos."
    ),
    ("jewellery", "Valentino"): (
        "Create a restrained six-panel product-only Valentino Garavani jewellery moodboard. Show one different "
        "current Maison-code product per panel: VLogo enamel-and-crystal ring, Coeur Royal enamel earrings, slim "
        "VLogo metal bracelet, VLogo metal necklace, Rockstud leather bracelet and Coeur Royal enamel bracelet. "
        "Use polished gold, black, ivory and one controlled red accent. Keep scale wearable and graphic, not bridal "
        "or opulent. No giant gemstones, pearl strands, crystal-drop earrings, people, repeats, text or labels."
    ),
    ("jewellery", "Vera Wang"): (
        "Create a restrained ring-led Vera Wang LOVE jewellery moodboard with exactly six panels. Each panel shows "
        "one visibly different engagement ring or wedding band once: marquise solitaire, oval solitaire, radiant "
        "solitaire, slim diamond band, scalloped-frame ring and clean two-tone bridal set. Give every piece a subtle "
        "blue-sapphire accent associated with the collection. Product-only; no people, earrings, necklaces, cuffs, "
        "pearls, tiaras, floral crystal sprays, repeated rings, text or logos. Use clean white, charcoal and silver."
    ),
    ("jewellery", "Veronica Beard"): (
        "Create a colourful six-panel product-only Veronica Beard jewellery moodboard with one unique piece per "
        "panel: Cat's Eye cabochon earrings, Tiger's Eye beaded necklace, Cat's Eye collar necklace, Nautilus shell "
        "beaded necklace, pearl pendant cord necklace and a bright paracord charm bracelet. Use gold, tiger-eye "
        "brown, turquoise blue, shell white, coral pink and black. No people, worn views, repeated products, loose "
        "stones, scenery panels, text, labels or logos."
    ),
    ("jewellery", "Yumi Kim"): (
        "Create a vertical product-only Yumi Kim jewellery moodboard with exactly six distinct panels: one large "
        "Cascade gold statement necklace, Aura gold earrings, Aurelia gold hoops, Beachy chain-drop earrings, a "
        "Bloom gold ring and a bold stacked bangle set. Balance three large statement pieces with three smaller "
        "pieces. Use joyful floral colour in pink, coral, turquoise and gold. Show every product once; no people, "
        "worn views, repeated charms, loose pearls, words, labels or logos."
    ),
    ("bags", "Acne Studios"): (
        "Create a vertical product-only Acne Studios bag moodboard with exactly six panels and six recognizable "
        "popular silhouettes: a black Musubi midi shoulder bag with obi-knot sides, cognac Musubi tote, distressed "
        "Multipocket shoulder bag, structured Camero camera bag, elongated Bowlina bowling bag and compact Platt "
        "shoulder bag. Use black, cognac, dusty blue and one vivid seasonal colour. One whole bag per panel only; "
        "no people, detail-only crops, repeated models, generic totes, text, labels or invented logos."
    ),
    ("bags", "ALO YOGA"): (
        "Create a vertical luxury product-only moodboard for the ALO Atelier Bag Collection with exactly six "
        "different Italian-made silhouettes: Voyage cylindrical duffle, Unwind cinch shoulder bag, Tranquility "
        "leather-mesh tote, Odyssey structured bowler, Balance slouchy bucket and Mini Voyage. Use responsible "
        "calfskin, suede and laser-cut leather mesh in black, cream and rich brown, with custom metal hardware and "
        "one small hanging intention crystal per bag. No gym backpacks, nylon belt bags, yoga mats, people, repeated "
        "silhouettes, text, labels or logos."
    ),
    ("bags", "AMI Paris"): (
        "Create a vertical product-only AMI Paris leather-goods moodboard with exactly six distinct current bag "
        "silhouettes: black Paris Paris shoulder bag, brown Paris Paris top-handle bag, burgundy Mimi bag, yellow "
        "Carrousel mini bag, beige Etienne crossbody and black Voulez-Vous bag. The Paris Paris family must use the "
        "correct Ami de Coeur metal stud: a small heart sitting above an A-shaped stem, not a plain heart keyhole. "
        "Show each whole bag once, in refined Parisian black, brown, burgundy, beige and yellow. No people, detail "
        "duplicates, invented locks, text, labels or unrelated logos."
    ),
    ("bags", "Anya Hindmarch"): (
        "Create a cheeky six-panel product-only Anya Hindmarch bag moodboard. Show six complete, different bags "
        "once each: black Eyes tote, bright red Bow clutch, crystal Embellished Eyes clutch, playful brown dog-shaped "
        "crossbody, powder-blue cloud-patch tote and a practical labelled Multi Pocket tote. Use expressive eyes, "
        "bows, colour and witty object design, but never repeat a bag as a detail crop. No people, standalone closeups, "
        "loose plastic bottles, words, labels or unrelated logos."
    ),
    ("bags", "Brunello Cucinelli"): (
        "Create a quiet-luxury six-panel product-only Brunello Cucinelli bag moodboard. Show six complete and "
        "different Italian-crafted bags: woven nappa leather tote, soft suede bucket, structured grained-leather "
        "top-handle, relaxed suede hobo, compact leather crossbody and refined monili-trim evening clutch. Use "
        "chocolate, tobacco, taupe, camel and warm grey. No people, straw, canvas, raffia, material-only detail crops, "
        "repeat views, words, labels or logos."
    ),
    ("bags", "CASABLANCA"): (
        "Create a modern, witty six-panel product-only Casablanca Paris bag moodboard. Use six different clean "
        "graphic silhouettes: monogram tennis-racket crossbody, orange-shaped mini bag, airline-travel pouch, green "
        "table-tennis shoulder bag, playing-card box bag and crisp colour-block weekender. Work in white, emerald, "
        "orange, sky blue and black with sharp contemporary geometry and a humorous luxury-sport attitude. No gold "
        "shell clasps, crystal mesh, sequins, glitter, boho styling, people, repeats, text, labels or logos."
    ),
    ("bags", "Celine"): (
        "Create a six-panel product-only Celine hero-bag moodboard with one complete iconic family per panel: black "
        "Luggage tote with winged gussets, tan Classique Triomphe box bag, tan half-moon Ava Triomphe, structured "
        "camel 16 top-handle, taupe Belt bag with long front straps and a black-and-gold Victoire chain shoulder bag. "
        "Use exact recognizable proportions and correct double-C Triomphe hardware only where appropriate. No "
        "people, generic totes, detail crops, duplicate bags, invented turn locks, scenery, text, labels or logos."
    ),
    ("bags", "Chan Luu"): (
        "Create a six-panel product-only Chan Luu bag moodboard with one complete different cool-girl bag per panel: "
        "black sequin shoulder bag, silver paillette pouch, burgundy leather mini bag with long silk-ribbon ties, "
        "navy satin knot bag, chocolate suede baguette with ribbon handle and a charcoal beaded evening pouch. Use "
        "black, silver, burgundy, navy and chocolate. No raffia, straw, shells, beach scenery, people, detail crops, "
        "repeated silhouettes, words, labels or logos."
    ),
    ("bags", "Chanel"): (
        "Create a six-panel product-only Chanel bag moodboard with one complete different icon per panel: black "
        "quilted 2.55 Reissue with chain, red Classic Flap, ivory-and-black tweed Classic Flap, black Boy bag, jewel-"
        "tone Chanel 19 and a small colourful quilted top-handle. Make chain straps prominent and use correct quilted "
        "proportions. No brown, tan or suede, canvas totes, people, detail crops, duplicate black flaps, text or labels."
    ),
    ("bags", "Chopova Lowena"): (
        "Create a six-panel product-only Chopova Lowena bag moodboard with six visibly different complete bags: red "
        "tartan studded tote, black leather grommet bucket, plaid-and-leather hobo, small studded tartan shoulder bag, "
        "black chain-covered mini bag and a red-black patchwork satchel. Use punk plaid, heavy silver rings, chains, "
        "grommets, studs and charm clusters. No straw, people, worn views, detail crops, repeated bags, text or logos."
    ),
    ("bags", "chrome hearts"): (
        "Create a six-panel product-only Chrome Hearts bag moodboard with six distinct complete black leather bags: "
        "large cross-appliqué tote, dagger-hardware messenger, cemetery-cross belt bag, silver-cross bucket, gothic "
        "chain duffle and compact fleur-de-lis shoulder bag. Give each bag a different silhouette and one controlled "
        "sterling-silver motif family. No people, worn views, hardware-only crops, repeated bags, graveyard scenery, "
        "words, labels or unrelated logos."
    ),
    ("bags", "Comme des Garçons"): (
        "Create a six-panel product-only Comme des Garçons bag moodboard with six visibly different complete bags: "
        "a stark black leather tote with oversized white CDG letterform graphics, a red-and-black geometric shopper, "
        "a sculptural asymmetric black shoulder bag, a glossy black box bag, a crumpled metallic-silver pouch and a "
        "flat monochrome document bag. Keep the art direction cerebral, graphic, monochrome and avant-garde. No rope "
        "handles, nautical styling, conventional luxury lock bags, people, detail crops, repeated silhouettes or "
        "unrelated logos. Any lettering must be clean, intentional and limited to the single CDG graphic tote."
    ),
    ("bags", "Coperni"): (
        "Create a six-panel product-only Coperni bag moodboard with six different complete iconic silhouettes: a "
        "black Swipe bag, silver Mini Swipe, red Heart tote, transparent glass Swipe, black Origami bag and a white "
        "Air Swipe bag. Emphasize futuristic curved geometry, precise minimal construction and sharp Paris-tech "
        "styling. No generic rectangular tote, repeated Swipe colourways, people, worn views, detail crops, words, "
        "labels or invented logos."
    ),
    ("bags", "Cult Gaia"): (
        "Create a six-panel product-only Cult Gaia bag moodboard with six different sculptural statement bags: a "
        "natural bamboo Ark clutch, black openwork Ark, translucent emerald acrylic clutch, pearl spherical top-"
        "handle, metallic fan-shaped minaudière and an irregular white sculptural shoulder bag. Balance acrylic, "
        "black openwork and unusual hard-shell forms so the board is not dominated by beige straw. No repeated Ark "
        "bags, ordinary raffia totes, people, worn views, detail crops, text, labels or logos."
    ),
    ("bags", "Cuyana"): (
        "Create a polished six-panel product-only Cuyana bag moodboard with six complete, distinct, angular leather "
        "silhouettes: deep-green structured tote, burgundy top-handle, black trapeze shoulder bag, oxblood saddle bag, "
        "camel geometric crossbody and navy structured work satchel. Use refined edge paint, restrained hardware and "
        "clean architectural proportions. No floppy boho bags, pale-beige-only palette, people, repeated products, "
        "detail crops, text, labels or logos."
    ),
    ("bags", "DeMellier"): (
        "Create a chic six-panel product-only DeMellier bag moodboard with six complete, distinct silhouettes: black "
        "New York tote, burgundy Vancouver shoulder bag, tan Nano Montreal top-handle, deep-green Tokyo hobo, ivory "
        "Santa Monica crossbody and chocolate Midi Paris bag. Use quiet polished leather and varied subtle hardware; "
        "only two bags may feature a prominent metal clasp. No identical flap bags, repeated clasp closeups, people, "
        "worn views, text, labels or invented logos."
    ),
    ("bags", "Deux Mains"): (
        "Create a six-panel product-only Deux Mains bag moodboard emphasizing rich Haitian craft texture. Show six "
        "different complete bags: woven-leather tote, embossed-leather bucket, hand-braided crossbody, pebbled-leather "
        "top-handle, suede-and-leather shoulder bag and a patchworked leather clutch. Use cognac, black, rust, deep "
        "green and cream with visible handwork and clean modern shapes. No repeated silhouettes, people, worn views, "
        "texture-only crops, words, labels or logos."
    ),
    ("bags", "DKNY"): (
        "Create a six-panel product-only DKNY bag moodboard centered on women's New York accessories. Show six "
        "different complete bags: black-and-white DKNY monogram tote, red logo shoulder bag, black quilted chain bag, "
        "silver mini crossbody, cobalt structured satchel and cream hobo. Use crisp metropolitan styling and make "
        "only two bags logo-led. No backpacks, menswear cues, people, worn views, repeated silhouettes, detail crops "
        "or accidental text beyond the intentional clean DKNY monogram."
    ),
    ("bags", "Filippa K"): (
        "Create a restrained six-panel product-only Filippa K bag moodboard with six distinct complete Scandinavian "
        "bags: black soft tote, taupe minimalist shoulder bag, chocolate sculptural hobo, grey compact crossbody, "
        "cream structured top-handle and muted-olive utility bag. Use clean contemporary proportions, matte leather "
        "and calm neutral backdrops. No chairs, furniture, people, interiors, repeated props, duplicate bags, text, "
        "labels or logos."
    ),
    ("bags", "Gabriela Hearst"): (
        "Create a six-panel product-only Gabriela Hearst bag moodboard balancing classic craft with playful form. "
        "Show six distinct complete icons: black Nina bracelet bag, burgundy Demi bracelet bag, forest-green Diana "
        "top-handle, woven cognac Baez tote, ivory accordion-fold clutch and a sculptural gold evening minaudière. "
        "Use luxurious leather, quiet woven craft and precise rounded geometry. No generic beige tote, repeated "
        "bracelet bags, people, detail crops, words, labels or logos."
    ),
    ("bags", "Ganni"): (
        "Create a playful six-panel product-only Ganni bag moodboard for a Copenhagen cool-girl wardrobe. Show six "
        "different complete bags: one leopard-print shoulder bag, bright-red Bou bag, lime-green knot mini bag, "
        "metallic-silver hobo, bubblegum-pink recycled tote and cobalt-blue crescent crossbody. Use fun colour, quirky "
        "proportions and confident contemporary styling. Exactly one leopard bag; no beige minimalism, backpacks, "
        "people, repeats, detail crops, words, labels or logos."
    ),
    ("bags", "Gimaguas"): (
        "Create a six-panel product-only Gimaguas bag moodboard with youthful Mediterranean colour, texture and "
        "unusual shape. Show six different complete bags: one legitimate natural-straw tote, cherry-red ruched mini "
        "bag, cobalt woven shoulder bag, silver metallic crescent, lime beaded pouch and chocolate sculptural leather "
        "hobo. Keep the mix tactile, playful and nightlife-ready. Exactly one straw bag; no repeated materials, "
        "people, worn views, detail crops, text, labels or logos."
    ),
    ("bags", "J.Crew"): (
        "Create a colourful six-panel product-only J.Crew bag moodboard with polished American-preppy variety. Show "
        "six different complete bags: striped canvas-and-leather tote, cherry-red structured bucket, emerald suede "
        "shoulder bag, navy top-handle satchel, yellow woven clutch and pink-and-orange colour-block crossbody. Use "
        "bright collegiate colour, clean leather trim and relaxed East Coast styling. No repeated totes, beige-only "
        "palette, people, worn views, detail crops, words, labels or logos."
    ),
    ("bags", "J.W. Anderson"): (
        "Create a six-panel product-only J.W. Anderson bag moodboard with six unmistakably different complete designs: "
        "the realistic grey Pigeon clutch, a playful green Frog clutch, black Chain Link shoulder bag, red Bumper moon "
        "bag, tan Corner bag and a new sculptural silver chain-strap top-handle. Mix wit, surreal animal objects and "
        "precise contemporary leatherwork. Show each design once only; no paired duplicates, alternate views, people, "
        "detail crops, words, labels or logos."
    ),
    ("bags", "Jacquemus"): (
        "Create a vivid six-panel product-only Jacquemus bag moodboard showing six unmistakably different constructions: "
        "one yellow Le Chiquito mini top-handle, one cobalt Le Valérie curved shoulder bag, one red Le Turismo bowling "
        "bag, one emerald Rond Carré hard clutch, one orange Cuerda shopper and one silver Ovalo oval clutch. Exactly "
        "one flap bag and exactly one miniature top-handle; all other products must differ radically in silhouette and "
        "closure. Use sunny South-of-France colour. No Bambino variations, beige dominance, people, text or labels."
    ),
    ("bags", "Jil Sander"): (
        "Create a quiet six-panel product-only Jil Sander bag moodboard with six different complete minimalist icons: "
        "black Cannolo shoulder bag, cream Goji bamboo-handle top bag, burgundy Tangle crossbody, dark-green folded "
        "Origami bag, white sculptural Curve bag and a silver frame clutch. Use exact clean geometry, unbranded surfaces "
        "and restrained modernist styling. No generic shopping tote, repeated shapes, people, detail crops, text or logos."
    ),
    ("bags", "Jonathan Simkhai"): (
        "Create a six-panel product-only Jonathan Simkhai evening-bag moodboard with six complete, visibly different "
        "occasion pieces: silver crystal-mesh pouch, black satin sculptural clutch, pearl-beaded mini bucket, gold metal "
        "frame bag, ivory pleated wristlet and deep-red embellished shoulder bag. Make the board glamorous, architectural "
        "and night-ready. No daytime totes, repeated crystal pouches, people, worn views, detail crops, text or logos."
    ),
    ("bags", "JW Pei"): (
        "Create a six-panel product-only JW Pei bag moodboard with six different recognizable popular shapes: green "
        "Gabbi ruched hobo, black Maze shoulder bag, ivory Joy shoulder bag, orange Abacus top-handle, cobalt Tessa "
        "crushed bag and silver Harlee shoulder bag. Give each a distinct silhouette and colour. Show each whole bag "
        "once only; no paired duplicates, alternate views, people, detail crops, text, labels or logos."
    ),
    ("bags", "Lindsay Nicholas New York"): (
        "Create a six-panel product-only Lindsay Nicholas New York bag moodboard with six complete, distinct clean-"
        "lined bags: burgundy crescent shoulder bag, cobalt structured tote, forest-green soft hobo, ivory woven clutch, "
        "red compact crossbody and black sculptural top-handle. Use refined New York minimalism with richer colour and "
        "varied shapes. No padlocks, lock clasps, generic luggage hardware, repeated silhouettes, people, text or logos."
    ),
    ("bags", "Maison Kitsuné"): (
        "Create a six-panel product-only Maison Kitsuné bag moodboard with the signature fox clearly present. Show six "
        "different complete bags: navy tote with a small fox-head patch, red mini crossbody shaped like a fox face, cream "
        "canvas shopper with an embroidered fox, black nylon shoulder bag with a fox charm, tan leather bucket with a "
        "subtle fox clasp and a cobalt compact backpack with one fox patch. Keep it playful Paris-meets-Tokyo. No repeated "
        "fox bag, people, detail crops, generic animal motifs, words, labels or unrelated logos."
    ),
    ("bags", "Maje"): (
        "Create a six-panel product-only Maje bag moodboard centered on the popular M-fringe family while preserving "
        "variety. Show one black M shoulder bag with long leather fringe, plus five different complete silhouettes: "
        "burgundy quilted chain bag, cream structured top-handle, metallic silver mini bag, forest-green suede hobo and "
        "red compact crossbody. Exactly one fringed bag; no duplicate M bags, people, detail crops, text or labels."
    ),
    ("bags", "Mar Y Sol"): (
        "Create a six-panel product-only Mar Y Sol bag moodboard with colorful handwoven raffia craft. Show six distinct "
        "complete bags: turquoise-and-natural striped tote, coral woven bucket, yellow round crossbody, cobalt clutch, "
        "pink-orange market basket and one classic natural raffia shoulder bag. Use joyous saturated color alongside "
        "beige straw, not beige alone. No repeated basket silhouettes, people, detail crops, text, labels or logos."
    ),
    ("bags", "Marc Jacobs"): (
        "Create a six-panel product-only Marc Jacobs bag moodboard with six distinct complete hero bags: black canvas "
        "The Tote Bag with one clean rectangular wordmark, small cobalt leather Tote Bag, red Snapshot camera bag, silver "
        "Stam bag, pink quilted J Marc shoulder bag and a yellow mini bucket. Make the large tote and small tote visibly "
        "different in size, material and construction. No straw, no double red zipper bag, people, detail crops, repeated "
        "products or accidental text beyond the intentional tote wordmark."
    ),
    ("bags", "Marge Sherwood"): (
        "Create a six-panel product-only Marge Sherwood bag moodboard with hip Seoul styling, playful bag charms and "
        "strong color. Show six different complete silhouettes: cherry-red baguette with silver charms, lime-green soft "
        "hobo, cobalt crinkled shoulder bag, metallic-silver bowling bag, cream bean-shaped mini bag with bead charms and "
        "chocolate sculptural top-handle. Every bag must have a different shape; use charms on only two. No generic luxury "
        "flaps, repeated bags, people, detail crops, text, labels or logos."
    ),
    ("bags", "Marine Serre"): (
        "Create a six-panel product-only Marine Serre bag moodboard with the crescent-moon code unmistakable. Show six "
        "different complete bags: black leather shoulder bag with silver moon buckle, red moon-print crescent bag, tan "
        "recycled-canvas tote with repeating moon print, metallic-silver mini bag with one moon clasp, black quilted "
        "crossbody with moon hardware and a cobalt cylindrical bag with subtle crescent pattern. Balance buckle and print "
        "signatures without repeating a silhouette. No people, detail crops, unrelated logos, words or labels."
    ),
    ("bags", "Marni"): (
        "Create a six-panel product-only Marni bag moodboard with six complete bags and six entirely different surface "
        "treatments: black-and-white Museo color block, red Trunk bag, blue-and-green woven Market tote, yellow leather "
        "bucket, brown shearling hobo and a pink sculptural clutch. No print, stripe, motif or color combination may repeat "
        "between panels. Keep the mood eccentric, artful and boldly modern. No people, detail crops, duplicate bags, text, "
        "labels or logos."
    ),
    ("bags", "Miu Miu"): (
        "Create a six-panel product-only Miu Miu bag moodboard with six complete, recognizable feminine bags: red "
        "perforated matelassé leather shoulder bag, cobalt Arcadie top-handle, pink Wander hobo, green Beau bowling bag, "
        "yellow quilted mini bag and a black leather clutch. Use visible clean MIU MIU metal lettering on exactly two bags, "
        "bright color and tactile perforated or matelassé leather. No paisley, velvet, key chains, boho motifs, repeated "
        "products, people, detail crops or accidental text."
    ),
    ("shoes", "ABRA"): (
        "Create a vertical six-panel product-only ABRA Paris footwear moodboard with one complete single shoe per "
        "panel: black Baby Boot, tall red Baby Boot, silver Kitten Spike Ballerina, brown Duck Loafer, leopard Sneaker "
        "Ballerina with bow and a black XXL Flip Flop. Make every silhouette, height and material visibly different "
        "with playful surreal Paris styling. No people, feet, worn views, shoe pairs, duplicate colourways, alternate "
        "angles, bags, clothing, text, labels, logos or extra products."
    ),
    ("shoes", "Air Jordan"): (
        "Create a vertical six-panel product-only Air Jordan footwear moodboard with one complete single hero shoe per "
        "panel: black-red Air Jordan 1 high, white-blue Air Jordan 3, yellow-black Air Jordan 4, black-white Air Jordan 6, "
        "black patent-and-white Air Jordan 11 and a modern wolf-grey Air Jordan 40 performance shoe. Preserve visibly "
        "different sole, collar and panel constructions. No people, feet, worn views, shoe pairs, balls, duplicate "
        "models, alternate angles, text, labels, malformed logos or extra products."
    ),
    ("shoes", "Asics"): (
        "Create a vertical six-panel product-only ASICS SportStyle footwear moodboard with one complete single hero shoe "
        "per panel: silver-blue GEL-KAYANO 14, cream GEL-NYC, grey-green GT-2160, white-silver GEL-1130, brown technical "
        "GEL-VENTURE 6 and black sculptural GEL-KINETIC. Keep each sole and upper construction visibly different while "
        "using precise layered mesh and synthetic panels. No people, feet, worn views, shoe pairs, duplicate runners, "
        "alternate angles, text, labels, malformed lettering or extra products."
    ),
    ("shoes", "Adidas"): (
        "Create a vertical six-panel product-only adidas footwear moodboard with one complete single hero shoe per panel: "
        "black-white Samba OG with gum sole, cobalt suede Gazelle Indoor, white-black shell-toe Superstar, yellow-blue "
        "SL 72, burgundy Handball Spezial and white technical Adizero EVO SL. Use clean three-stripe construction but no "
        "generated words. No people, feet, worn views, shoe pairs, duplicate terrace shoes, alternate angles, text, "
        "labels, malformed lettering or extra products."
    ),
    ("shoes", "Golden Goose"): (
        "Create a vertical six-panel product-only Golden Goose sneaker moodboard with one complete single hero shoe per "
        "panel: distressed white Super-Star with black star, white-red Ball Star, silver Mid Star high-top, grey mesh "
        "Running Sole, black suede Slide high-top and violet nylon Marathon. Make silhouette and material differences "
        "clear; use a single clean star patch per shoe. No people, feet, worn views, shoe pairs, duplicate low-tops, "
        "alternate angles, loose laces, text, labels, malformed lettering or extra products."
    ),
    ("shoes", "Birkenstock"): (
        "Create a vertical six-panel product-only Birkenstock footwear moodboard with one complete single hero item per "
        "panel: tan two-strap Arizona, taupe suede Boston clog, black Gizeh thong sandal, red Madrid Big Buckle slide, "
        "dark-brown backstrap Tokio clog and white Bend Low sneaker. Show the anatomical cork footbed clearly where "
        "appropriate and keep all six silhouettes distinct. No people, feet, worn views, shoe pairs, duplicate sandals, "
        "alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "EYTYS"): (
        "Create a vertical six-panel product-only EYTYS footwear moodboard with one complete single hero shoe per panel: "
        "black suede Doja slip-on, off-white canvas Anthem plimsoll, white leather Balestra sporty dress shoe, silver-grey "
        "technical Trophy sneaker, black architectural Piston boot and barolo leather Poem boot. Emphasize EYTYS' bold "
        "proportions, substantial soles and stark Scandinavian styling while keeping every construction visibly different. "
        "No people, feet, worn views, shoe pairs, duplicate models, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Geox"): (
        "Create a vertical six-panel product-only Geox footwear moodboard with one complete single hero shoe per panel: "
        "silver technical Spherica sneaker, navy Amphibiox waterproof ankle boot, burgundy Walk Pleasure pump, cream "
        "Nebula slip-on, teal Aerantis runner and coral Climasandal. Make the breathable perforated soles and comfort-led "
        "engineering believable, with six clearly distinct use cases and silhouettes. No people, feet, worn views, shoe "
        "pairs, duplicate trainers, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Gia Borghini"): (
        "Create a vertical six-panel product-only Gia Borghini footwear moodboard with one complete single hero shoe per "
        "panel: brown leather Rosie 3 square-toe thong sandal with braided ankle strap, cream sculptural Perni slide, black "
        "Marte platform boot, minimal black Studio thong, metallic-silver architectural Muse thong wedge and white strappy "
        "Icon heel. Use sleek Italian leather, sculptural curves and fashion-forward proportions, with six unmistakably "
        "different constructions. No people, feet, worn views, shoe pairs, duplicate sandals, alternate angles, text, "
        "labels, logos or extra products."
    ),
    ("shoes", "Church's"): (
        "Create a vertical six-panel product-only Church's footwear moodboard with one complete single hero shoe per panel: "
        "polished black Consul Oxford, oxblood Shannon wholecut Derby, tan Diplomat half-brogue Oxford, walnut Chetwynd "
        "full-brogue Oxford, forest-green suede Pembrey loafer and dark-brown Amberley Chelsea boot. Render precise English "
        "welted construction, refined leather and visibly different toe, lacing and upper details. No people, feet, worn "
        "views, shoe pairs, duplicate brogues, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Gucci"): (
        "Create a vertical six-panel product-only Gucci footwear moodboard with one complete single hero shoe per panel: "
        "black leather Horsebit 1953 loafer, red leather Princetown backless slipper, white-green Ace low-top sneaker, beige "
        "distressed Screener sneaker, burgundy Vittoria slingback pump and dark-brown sleek ankle boot. Use restrained "
        "horsebit and web-stripe house codes only where structurally appropriate, with every silhouette distinct. No "
        "people, feet, worn views, shoe pairs, duplicate loafers or sneakers, alternate angles, words, labels, malformed "
        "lettering or extra products."
    ),
    ("shoes", "Crocs"): (
        "Create a vertical six-panel product-only Crocs footwear moodboard with one complete single hero shoe per panel: "
        "cobalt Classic Clog with ventilation holes and pivoting heel strap, bone sculptural Echo II clog, cherry-red Dylan "
        "Platform Clog, pale-lilac Mellow 2 Recovery slide, black Brooklyn Low Wedge Sandal and white On The Clock Work "
        "Slip-On. Make all six molded-foam silhouettes clearly different and show no decorative charms. No people, feet, "
        "worn views, shoe pairs, duplicate clogs, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Clarks"): (
        "Create a vertical six-panel product-only Clarks footwear moodboard with one complete single hero shoe per panel: "
        "maple-suede Wallabee, sand-suede Desert Boot with crepe sole, beeswax-leather Desert Trek, red-suede Torhill Hi "
        "with chunky ribbed sole, navy-nubuck Nalle Lace and black-suede Wallabee mule. Preserve the brand's moccasin seams, "
        "natural crepe and comfort-led British design while making every silhouette clearly different. No people, feet, "
        "worn views, shoe pairs, duplicate models, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Ecco"): (
        "Create a vertical six-panel product-only ECCO footwear moodboard with one complete single hero shoe per panel: "
        "white leather Soft 7 trainer, burgundy nubuck Gruuv trainer with flexible segmented sole, grey technical BIOM 720 "
        "BREATHRU, brown Metropole Vienna leather loafer, tan Cozmo two-strap sandal and black RECEPTOR XP waterproof outdoor "
        "shoe. Emphasize premium leather, anatomical comfort and direct-injected soles with six distinct constructions. No "
        "people, feet, worn views, shoe pairs, duplicate trainers, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Converse"): (
        "Create a vertical six-panel product-only Converse footwear moodboard with one complete single hero shoe per panel: "
        "black Chuck Taylor All Star high-top, parchment Chuck 70 low-top, red suede One Star Pro, white-green leather "
        "Weapon low-top, black Run Star Hike platform high-top and cobalt AS-1 Pro skate shoe. Keep rubber toe caps, foxing, "
        "star motifs and proportions structurally believable while making all six silhouettes distinct. No people, feet, "
        "worn views, shoe pairs, duplicate Chucks, alternate angles, words, labels, malformed lettering or extra products."
    ),
    ("shoes", "Carvela"): (
        "Create a vertical six-panel product-only Carvela footwear moodboard with one complete single hero shoe per panel: "
        "white leather Connected Zip trainer, chocolate suede Click loafer, red Positano lace slingback heel, silver Silvia "
        "mule heel, black leather Ignite Chelsea boot and tan suede Spirit slouch knee boot. Express polished contemporary "
        "London occasion-to-everyday style with six clearly different silhouettes. No people, feet, worn views, shoe pairs, "
        "duplicate heels or boots, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Diemme"): (
        "Create a vertical six-panel product-only Diemme footwear moodboard with one complete single hero shoe per panel: "
        "black full-grain Roccia Vet alpine boot, tobacco-suede Cornaro ankle boot, grey-fabric Roccia Basso low hiker, orange "
        "technical Grappa Hiker, dark-fallow suede Movida leisure shoe and burgundy crackled-leather Licata loafer. Show "
        "handcrafted Italian outdoor construction, premium materials and sturdy soles, with all six silhouettes distinct. "
        "No people, feet, worn views, shoe pairs, duplicate hikers, alternate angles, text, labels, logos or extra products."
    ),
    ("shoes", "Hunter"): (
        "Create a vertical six-panel product-only Hunter footwear moodboard with one complete single hero shoe per panel: "
        "dark-green Original Tall Rain Boot, yellow PLAY Short Rain Boot, black Original Chelsea Boot, brown Southall "
        "insulated duck boot, red waterproof rubber mule and navy webbing sandal. Preserve matte vulcanized rubber, practical "
        "weatherproof construction and clean British utility while making every height and silhouette distinct. No people, "
        "feet, worn views, shoe pairs, duplicate rain boots, alternate angles, text, labels, logos or extra products."
    ),
    ("jewellery", "Vivienne Westwood"): (
        "Create a vertical six-panel Vivienne Westwood jewellery moodboard with exactly one different hero piece per panel: "
        "silver crystal New Petite Orb pendant, gold Mini Bas Relief orb earrings, pearl Mini Bas Relief choker, red crystal "
        "Diamante Heart ring, gunmetal punk padlock-and-chain bracelet with one small orb plaque and a sculptural silver "
        "Octavie ear cuff. Mix polished product still lifes with at most two elegant worn crops on fully clothed adult models. "
        "Keep the orb, pearls and punk hardware precise. No repeated item, duplicate worn view, words, letters, labels, "
        "watermarks, unrelated logos, extra jewellery or malformed symbols."
    ),
    ("jewellery", "Roxanne Assoulin"): (
        "Create a vertical six-panel Roxanne Assoulin jewellery moodboard with exactly one different joyful hero piece per "
        "panel: rainbow enamel Affogato bracelet, multicolour Fruit Salad bracelet, oversized red Big Puffy Heart cord "
        "necklace, cobalt Crystal Granita cord necklace, striped carnelian-agate necklace and bright crystal Le Petite drop "
        "earrings. Use saturated red, blue, green, orange and pink with playful bead, enamel, stone and cord textures. Mix "
        "still lifes with at most two worn crops on fully clothed adult models. No repeated hearts, repeated products, words, "
        "letters, labels, watermarks, unrelated logos or extra jewellery."
    ),
    ("jewellery", "Roxanne First"): (
        "Create a vertical six-panel Roxanne First fine-jewellery moodboard with exactly one different hero piece per panel: "
        "rainbow-sapphire tennis necklace, rainbow-sapphire Sprinkle hoop earrings, multicolour Skittle bracelet, diamond "
        "Shooting Star pendant, polished gold Puffy Heart ring and playful pearl-and-rainbow-sapphire mushroom beaded "
        "necklace. Show delicate 14k gold, ethical diamonds and vivid gemstones designed for layering and stacking. Mix "
        "clean still lifes with at most two worn crops on fully clothed adult models. No initials, alphabet charms, repeated "
        "items, words, labels, watermarks, unrelated logos or extra jewellery."
    ),
    ("jewellery", "Sonia Petroff"): (
        "Create a vertical six-panel Sonia Petroff jewellery moodboard with exactly one different vintage-inspired hero piece "
        "per panel: gold Lobster Vibe necklace, colourful Parrot earrings, red Classic Eye bracelet, silver Swan crystal "
        "brooch, turquoise Cascata pinky ring and green Zinnia floral earrings. Use bold 1960s-70s nature motifs, 24k-gold "
        "overlay, silver palladium, cabochon colour and crystal sparkle. Mix product still lifes with at most two worn crops on "
        "fully clothed adult models. No repeated animal or flower, words, labels, watermarks, unrelated logos or extra jewellery."
    ),
    ("jewellery", "Zoe Mohm"): (
        "Create a vertical six-panel Zoé Mohm jewellery moodboard with exactly one different hand-built talisman per panel: "
        "sculptural silver mermaid ring, oxidised-silver bull ring, hand-woven silver-wire cuff, hammered eye pendant, brass "
        "hand pendant and an abstract horn-like lost-wax silver ring. Make the pieces raw, symbolic, surreal and visibly "
        "handmade through hammered, twisted, woven and cast surfaces. Mix artist-studio still lifes with at most two worn "
        "crops on fully clothed adult models. No repeated motif, mass-market polish, words, labels, watermarks, unrelated "
        "logos or extra jewellery."
    ),
    ("jewellery", "William Welstead"): (
        "Create a vertical six-panel William Welstead fine-jewellery moodboard with exactly one different hero piece per "
        "panel: vivid pink spinel solitaire ring, ruby eternity ring, irregular antique rose-cut diamond ring, briolette-and-"
        "diamond bead earrings, natural pearl and diamond pendant and Colombian emerald sculptural ring. Celebrate natural "
        "untreated stones, imperfect antique cuts, Mughal colour and restrained handmade English settings. Mix macro still "
        "lifes with at most two refined worn crops on fully clothed adult models. No repeated ring setting, generic pavé, "
        "words, labels, watermarks, unrelated logos or extra jewellery."
    ),
    ("jewellery", "YVMIN"): (
        "Create a vertical six-panel YVMIN jewellery moodboard with exactly one different surreal hero piece per panel: "
        "gradient strawberry pavé earring, sculptural apple-core earring, split-cat mother-of-pearl studs, pearl bow-tassel "
        "earrings, colourful crystal heart-with-skirt ring and shell-coin double-sided ring. Make miniature fruit, cats, bows, "
        "pearls and everyday-object transformations playful, uncanny and meticulously crafted. Mix product still lifes with "
        "at most two worn crops on fully clothed adult models. No repeated fruit or cat, words, letters, labels, watermarks, "
        "unrelated logos or extra jewellery."
    ),
    ("jewellery", "Yellow Swallow"): (
        "Create a vertical six-panel Yellow Swallow jewellery moodboard with exactly one different hero piece per panel: "
        "blue Aqua Bloom crystal bracelet, Cherry Blossoms pearl necklace, black-diamond flower brooch, pink Ribbon crystal "
        "cuff, black-nickel Lily metal necklace and dark crystal Maman floral brooch. Express the Korean brand's balance of "
        "girlhood and maturity through classic pearls and crystals darkened by cold metal, black nickel and rough floral "
        "forms. Mix still lifes with at most two worn crops on fully clothed adult models. No repeated flowers, duplicate "
        "products, words, letters, labels, watermarks, unrelated logos or extra jewellery."
    ),
}


SAFE_BRAND_DIRECTIONS = {
    ("clothes", "Amazuìn"): (
        "Use elegant sheer-layer dresses and body-aware separates on fully clothed adult models, plus one "
        "clearly separate swimwear product shown only on a dress form or studio flat lay. Work in black, khaki, "
        "off-white and cocoa; keep every silhouette distinct and avoid beach posing."
    ),
    ("clothes", "Karoline Vitto"): (
        "Center curve-inclusive adult models in sculptural, body-contouring jersey garments with asymmetric "
        "cutouts and signature metal-ring construction. Add burgundy, olive or cobalt to black; show complete "
        "opaque outfits and avoid generic tailoring or knitwear."
    ),
    ("clothes", "LOCI"): (
        "Treat the apparel offer as a narrow casual capsule only: six different complete looks built from hoodies, "
        "sweatshirts, sweatpants, relaxed tees and one fleece jacket in black, grey, cream, olive and one muted accent. "
        "Do not add tailoring, overcoats, dresses, formal knitwear or footwear product imagery."
    ),
    ("clothes", "Miaou"): (
        "Use unmistakable Y2K corset-inspired tops layered over opaque base garments, fitted trousers, ruched "
        "skirts and printed mesh layered over camisoles. Work in chocolate, cream, red and denim; avoid generic "
        "tailoring, plain knitwear and exposed torsos."
    ),
    ("clothes", "Organic Basics"): (
        "Show restrained sustainable everyday basics on fully clothed adult models: distinct tees, sweatshirts, "
        "joggers and leggings, supported by folded organic-cotton garments and close fabric details. Use quiet "
        "neutral colours and do not repeat a garment or crop."
    ),
    ("clothes", "ROTATE Birger Christensen"): (
        "Create confident Copenhagen party styling with short shorts paired with an oversized blazer, opaque "
        "top and tights, plus rounded statement sleeves and a distinct party dress. Use saturated jewel colours "
        "and metallic accents, with fully clothed adult models in strong neutral poses."
    ),
    ("clothes", "Siedrés"): (
        "Use vivid Mediterranean colour, energetic print and asymmetric construction across visibly different "
        "silhouettes: a draped dress, a printed separate, a cutout layered over an opaque base and a sculptural "
        "knit. Do not repeat a print, garment or pose."
    ),
    ("clothes", "Sugarlips"): (
        "Show contemporary feminine day-to-evening clothing with distinct dresses, tops, skirts and tailored "
        "separates. Give every tile a different silhouette, colour, crop and setting; avoid duplicated products "
        "and generic stock-photo styling."
    ),
    ("clothes", "The Attico"): (
        "Use high-impact Milan evening glamour: a fitted dress, sharp oversized tailoring, strong shoulders and "
        "a jewel-tone statement separate on fully clothed adult models. Exclude interiors and jumper closeups; "
        "keep footwear incidental rather than a dedicated tile."
    ),
    ("jewellery", "Shrimps"): (
        "Make earrings unmistakably dominant: six different pairs shown once each, drawing on the brand's "
        "playful art-and-textile language through diamanté shrimp motifs, bees, sculptural gold forms, green "
        "accents and faux-pearl drops. Do not show rings. Do not use blue. Do not repeat a necklace, charm, "
        "model or product through a worn view and a still life."
    ),
    ("jewellery", "Simon Miller"): (
        "Use the brand's vibrant Los Angeles mod language and make the assortment statement-earring led: a "
        "green palm pair, orange starfish pair, yellow banana pair, gummy-green raffia pair, natural raffia "
        "pair and black-natural fringe pair, with at most one restrained gold accent. Every pair appears once; "
        "avoid generic plain hoops, plain bands, silver chains, pearls and minimal luxury styling."
    ),
    ("jewellery", "Sofie Schnoor"): (
        "Treat the jewellery offer honestly as a very small Scandinavian capsule. Show exactly three distinct "
        "jewellery products once each: two different sculptural gold earring pairs and one refined gold necklace. "
        "Complete the six-panel composition with non-product context only—raw black leather texture, warm stone "
        "and Copenhagen architectural shadow—expressing the brand's clean, raw-yet-feminine aesthetic. Do not "
        "invent rings, charm stacks, pearls, pastel enamel or extra jewellery styles."
    ),
}


CATEGORY_GUARDS = {
    "clothes": (
        "This is a clothing-only board. Do not include dedicated bags, handbags, shoes, footwear, jewellery "
        "or accessory product tiles. Incidental styling must never become a focal object."
    ),
    "shoes": "This is a footwear-only board. Do not include dedicated bag, jewellery or clothing product tiles.",
    "bags": "This is a bags-only board. Do not include dedicated shoe, jewellery or clothing product tiles.",
    "jewellery": "This is a jewellery-only board. Do not include dedicated bag, shoe or clothing product tiles.",
}


def existing_row(category: str, brand_name: str):
    matches = brands_df[
        brands_df["category"].eq(category)
        & brands_df["brand_name"].str.casefold().eq(brand_name.casefold())
    ]
    return None if matches.empty else matches.iloc[0]


def addition_prompt(brand_name: str, category: str) -> str:
    category_subjects = {
        "clothes": "distinctive garments on adult models, signature silhouettes, styling layers and material details",
        "shoes": "distinctive footwear silhouettes, on-foot styling, sole and material details",
        "bags": "distinctive bag silhouettes, carried styling, closures, handles and material details",
        "jewellery": "distinctive jewellery pieces, worn styling, settings, links, stones and metal details",
    }
    return (
        f"Create a vertical 2:3 editorial magazine moodboard for {brand_name}, focused only on {category}. "
        f"Make it unmistakably specific to the brand using widely recognisable current design codes and "
        f"category-appropriate signature products. Show 6-9 distinct fragments: {category_subjects[category]}. "
        "Use tactile torn-paper collage construction, varied crop sizes and deliberate negative space. "
        "Every hero product, person, pose, colourway, crop and setting must be visibly distinct. "
        "Use no more than two images of the same person and adult fashion models only. "
        "Keep every product readable and category-correct. Do not repeat an item through another angle, crop, "
        "mirror, rotation or colourway. No rigid grid, generic beige luxury, bean-shape filler, captions, "
        "watermarks, accidental words, invented labels or unrelated logos. High-resolution portrait format."
    )


def moderation_safe_prompt(prompt: str) -> str:
    """Keep fashion direction intact while avoiding ambiguous adult-content wording."""
    replacements = {
        "more sensual": "more elegant and body-aware",
        "sensual": "elegant and body-aware",
        "sexier": "more confident and evening-focused",
        "sexy": "confident and evening-focused",
        "explicit": "provocative graphic",
    }
    for source, replacement in replacements.items():
        prompt = re.sub(rf"\b{re.escape(source)}\b", replacement, prompt, flags=re.IGNORECASE)
    return (
        prompt
        + " If people are shown, they must be adult fashion models. Absolutely no typography anywhere in the image: "
        "do not write the brand name, initials, slogans, captions, labels or decorative letters."
    )


def model_safe_prompt(brand_name: str, category: str, reviewer_note: str) -> str:
    cleaned_note = reviewer_note
    replacements = {
        r"s\s*e\s*x\s*y": "confident evening styling",
        r"s\s*e\s*x\s*ier": "more confident and evening-focused",
        r"sensual": "elegant and body-aware",
        r"explicit": "cheeky graphic",
    }
    for pattern, replacement in replacements.items():
        cleaned_note = re.sub(pattern, replacement, cleaned_note, flags=re.IGNORECASE)
    category_subject = {
        "clothes": "fully clothed adult models, garments on hangers, folded clothing, tailoring on dress forms, fabric and construction details",
        "shoes": "fully clothed adult models styling footwear, product still lifes, soles, stitching, hardware and material details",
        "bags": "fully clothed adult models carrying bags, product still lifes, handles, closures, hardware and material details",
        "jewellery": "fully clothed adult models wearing jewellery, product still lifes, settings, links, stones and metal details",
    }[category]
    brand_direction = SAFE_BRAND_DIRECTIONS.get((category, brand_name))
    direction_sentence = f"Specific brand direction: {brand_direction} " if brand_direction else ""
    return (
        f"Create a vertical editorial torn-paper moodboard for {brand_name}, focused only on {category}. "
        f"Reviewer correction brief, rewritten neutrally: {cleaned_note}. Show six to nine distinct fragments: "
        f"{category_subject}. Every product must be different; do not repeat a product through another angle, "
        "crop, mirror, rotation or colourway. Use adult models only, fully clothed in complete outfits, standing or "
        "seated in neutral editorial poses. No exposed torso, lingerie-like styling, swimwear-style posing, intimate "
        "framing, bedroom, typography, labels, captions, watermark or unrelated logos. Combine models with studio "
        "still lifes, flat lays, hangers and dress forms. Keep the brand direction recognisable, "
        "category-correct, tactile, polished and contemporary with varied crop sizes and negative space. "
        f"{direction_sentence}"
        f"{CATEGORY_GUARDS[category]}"
    )


def latest_reviews(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviews = payload.get("reviews", payload if isinstance(payload, list) else [])
    latest = {}
    for review in reviews:
        latest[review["review_id"]] = review
    return latest


def apply_brand_direction(
    prompt: str,
    category: str,
    brand_name: str,
    *,
    already_specialized: bool = False,
) -> str:
    """Append verified brand direction to ordinary prompts as well as safe prompts."""
    brand_direction = SAFE_BRAND_DIRECTIONS.get((category, brand_name))
    if not brand_direction or already_specialized:
        return prompt
    return prompt + f" Specific brand direction: {brand_direction}"


def retry_prompt(
    prompt: str,
    category: str,
    review: dict | None,
    safe_output: bool,
    reviewer_note: str = "",
) -> str:
    additions = [CATEGORY_GUARDS[category]]
    if review:
        additions.append(
            "The previous candidate was rejected for this exact reason: "
            f"{review['notes']} Correct that problem explicitly in this new version."
        )
    strict_shoe_first_pass = bool(
        re.search(
            r"repeat|same\s+(?:image|shoe|sandal)|close[ -]?ups?|words on|remove (?:the )?bag",
            reviewer_note,
            flags=re.IGNORECASE,
        )
    )
    if category == "shoes" and (
        strict_shoe_first_pass or (review and review.get("status") == "rejected")
    ):
        additions.append(
            "For this shoe correction, use a product-only editorial collage with exactly one unique shoe or pair "
            "per panel. Do not show people, feet, worn views, detail closeups, material swatches, soles, or "
            "alternate angles. Six to nine panels must mean six to nine genuinely different shoe designs."
        )
    if category == "clothes" and review and review.get("status") in {"rejected", "pending"}:
        additions.append(
            "For this clothing retry, use exactly six panels led by six visibly different complete garments or "
            "fully clothed adult outfits. Models may appear and should keep their hands empty. Do not include a "
            "dedicated handbag, shoe, jewellery, scarf, sock or accessory panel; do not place an accessory in the "
            "foreground; and do not use accessory-only still lifes. Footwear may appear only incidentally as part "
            "of a full outfit."
        )
    if safe_output:
        additions.append(
            "Moderation-safe rendering: people are welcome, but use only fully clothed adult fashion models in "
            "neutral editorial poses. Support them with product flat lays, hanging garments, dress forms and "
            "close material details. If the brief specifically requires swimwear, show that product only on a "
            "dress form or studio flat lay. Avoid exposed torsos, lingerie-like styling, provocative poses, "
            "bedrooms and intimate framing."
        )
    return prompt + " " + " ".join(additions)


def next_candidate_path(root: Path, category: str, brand_name: str) -> Path:
    folder = root / category
    slug = slugify(brand_name)
    for number in range(1, 100):
        stem = folder / f"{slug}__candidate_{number:02d}"
        if not any(stem.with_suffix(ext).exists() for ext in (".png", ".jpg", ".jpeg", ".webp")):
            return stem.with_suffix(".png")
    raise RuntimeError(f"No candidate slot available for {category}/{slug}")


def needs_generation(record: dict) -> bool:
    return (
        record["recommended_action"] in IMAGE_ACTIONS
        or record["review_id"] in CATALOG_VERIFY_REGENERATE_IDS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--category", choices=["clothes", "shoes", "bags", "jewellery"])
    parser.add_argument("--request-type", choices=["existing_board", "new_brand"])
    parser.add_argument("--include-existing-candidates", action="store_true")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--safe-output", action="store_true")
    parser.add_argument("--review-id", action="append", default=[])
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    reviews = latest_reviews(args.reviews)
    corrected_root = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2"
    queue = []
    for record in payload["records"]:
        if not needs_generation(record):
            continue
        if record["implementation_status"] == "implemented_v2":
            continue
        if args.category and record["category"] != args.category:
            continue
        if args.request_type and record["request_type"] != args.request_type:
            continue
        if args.review_id and record["review_id"] not in set(args.review_id):
            continue
        review = reviews.get(record["review_id"])
        if args.retry_rejected and (not review or review.get("status") != "rejected"):
            continue
        candidate_path = next_candidate_path(
            corrected_root,
            record["category"],
            record["brand_name"],
        )
        if not args.include_existing_candidates and candidate_path.stem.endswith("_02"):
            continue
        row = existing_row(record["category"], record["brand_name"])
        special_prompt = SPECIAL_PROMPT_OVERRIDES.get((record["category"], record["brand_name"]))
        if args.safe_output:
            base_prompt = model_safe_prompt(
                record["brand_name"], record["category"], record["reviewer_note"]
            )
        else:
            base_prompt = special_prompt or (
                moodboard_prompt(row, reviewer_note=record["reviewer_note"])
                if row is not None
                else addition_prompt(record["brand_name"], record["category"])
            )
            base_prompt = apply_brand_direction(
                base_prompt,
                record["category"],
                record["brand_name"],
                already_specialized=bool(special_prompt),
            )
        prompt = retry_prompt(
            moderation_safe_prompt(base_prompt),
            record["category"],
            review,
            args.safe_output,
            record["reviewer_note"],
        )
        queue.append(
            {
                "review_id": record["review_id"],
                "category": record["category"],
                "brand_name": record["brand_name"],
                "request_type": record["request_type"],
                "recommended_action": record["recommended_action"],
                "reviewer_note": record["reviewer_note"],
                "prompt": prompt,
                "candidate_path": str(candidate_path.relative_to(ROOT)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "queue": queue}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(queue)} queued generations to {args.output}")


if __name__ == "__main__":
    main()
