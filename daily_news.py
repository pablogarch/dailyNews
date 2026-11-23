import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import feedparser
import re
import math
import os
from collections import Counter

print("🔖 Versión del script: v8 (fuentes únicas + más fuentes españolas)")

# ================== CONFIGURACIÓN ==================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD") # Contraseña de aplicación
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

RSS_FEEDS = {
    # España (11 fuentes)
    "El País": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "El Mundo": "https://e00-elmundo.uecdn.es/rss/portada.xml",
    "elDiario.es": "https://www.eldiario.es/rss/",
    "20minutos": "https://www.20minutos.es/rss/",
    "La Vanguardia": "https://www.lavanguardia.com/rss/home.xml",
    "ABC": "https://www.abc.es/rss/",
    "El Confidencial": "https://rss.elconfidencial.com/espana/",
    "Público": "https://www.publico.es/rss/",
    "La Razón": "https://www.larazon.es/rss/",
    "El Periódico": "https://www.elperiodico.com/es/rss/",
    "El Español": "https://www.elespanol.com/rss/",
    # Europa (8 fuentes)
    "BBC Europe": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "Euronews": "https://www.euronews.com/rss",
    "Politico Europe": "https://www.politico.eu/feed/",
    "Euractiv": "https://www.euractiv.com/feed/",
    "France 24 Europe": "https://www.france24.com/en/europe/rss",
    "Deutsche Welle Europe": "https://rss.dw.com/rdf/rss-en-eu",
    "The Guardian Europe": "https://www.theguardian.com/world/europe-news/rss",
    "EUobserver": "https://euobserver.com/rss",
    # Mundo (4 fuentes)
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "France 24 World": "https://www.france24.com/en/rss",
}

SOURCE_REGION = {
    "El País": "spain", "El Mundo": "spain",
    "elDiario.es": "spain", "20minutos": "spain", "La Vanguardia": "spain", "ABC": "spain",
    "El Confidencial": "spain", "Público": "spain", "La Razón": "spain",
    "El Periódico": "spain", "El Español": "spain",
    "BBC Europe": "europe", "Euronews": "europe", "Politico Europe": "europe",
    "Euractiv": "europe", "France 24 Europe": "europe",
    "Deutsche Welle Europe": "europe", "The Guardian Europe": "europe", "EUobserver": "europe",
    "BBC World": "world", "The Guardian World": "world", "Al Jazeera": "world",
    "France 24 World": "world",
}

MIN_SOURCES = 3
ENTRIES_PER_FEED = 15
SIM_THRESHOLD = 0.15

NOISE_REGEX = re.compile(
    r"(podcast|bulletin|sponsored|newsletter|advertisement|live blog|live updates|"
    r"latest news|what to know|in pictures|photo gallery)",
    re.IGNORECASE,
)

STOPWORDS = set("""
a an the and or but if then else when where who whom whose which that this these those
is are was were be been being have has had do does did will would shall should may might must
can could of in on at to for with by from as into through during before after above below
between among against about up down out off over under again further once here there all any
both each few more most other some such no nor not only own same so than too very s t just
el la los las un una unos unas y o pero si entonces cuando donde quien quienes cuyo cuya que
este esta estos estas ese esa esos esas es son era eran ser estar haber tener hacer
de del en al a para por con sin sobre entre contra desde hasta durante antes después
muy más menos ya también como pero sino solo aún así
""".split())

# ================== 1. RSS ==================
def fetch_rss_articles():
    articles, skipped = [], 0
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:ENTRIES_PER_FEED]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link or len(title) < 15 or NOISE_REGEX.search(title):
                    skipped += 1
                    continue
                summary = entry.get("summary", "") or entry.get("description", "")
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                summary = re.sub(r"\s+", " ", summary)[:220] + ("..." if len(summary) > 220 else "")
                articles.append({
                    "source": source, "title": title, "link": link,
                    "summary": summary, "region": SOURCE_REGION.get(source, "world"),
                })
        except Exception as e:
            print(f"⚠️ Error con {source}: {e}")
    print(f"📥 Artículos obtenidos de RSS: {len(articles)} (descartados {skipped} por ruido)")
    return articles

# ================== 2. Tokenización + TF-IDF ==================
def tokenize(text):
    words = re.findall(r"\b[a-záéíóúüñ]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def text_for_vector(article):
    return article["title"] + " " + (article.get("summary") or "")[:120]

def build_idf(texts_tokens):
    df = Counter()
    for tokens in texts_tokens:
        for w in set(tokens):
            df[w] += 1
    n = len(texts_tokens)
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

def tfidf_vector(tokens, idf):
    return {w: c * idf.get(w, 1.0) for w, c in Counter(tokens).items()}

def cosine(d1, d2):
    common = set(d1) & set(d2)
    if not common:
        return 0.0
    dot = sum(d1[w] * d2[w] for w in common)
    n1 = math.sqrt(sum(v * v for v in d1.values()))
    n2 = math.sqrt(sum(v * v for v in d2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

# ================== 3. Single-linkage ==================
def group_articles(articles):
    texts_tokens = [tokenize(text_for_vector(a)) for a in articles]
    idf = build_idf(texts_tokens)
    vectors = [tfidf_vector(t, idf) for t in texts_tokens]
    n = len(articles)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if cosine(vectors[i], vectors[j]) >= SIM_THRESHOLD:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(articles[node])
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        groups.append(component)
    print(f"🧩 Grupos formados: {len(groups)}")
    print("🔝 Top 20 grupos por tamaño:")
    for g in sorted(groups, key=len, reverse=True)[:20]:
        fuentes = len({a['source'] for a in g})
        print(f"   {len(g):2} arts / {fuentes:2} fuentes → {g[0]['title'][:75]}")
    return groups

# ================== 4. Clasificación (fuentes únicas) ==================
def classify_group_by_reach(group):
    regions = {}
    for a in group:
        regions.setdefault(a["region"], set()).add(a["source"])
    spain = len(regions.get("spain", set()))
    europe = len(regions.get("europe", set()))
    world = len(regions.get("world", set()))
    if world > europe and world > spain:
        return "world"
    if europe > spain:
        return "europe"
    return "spain"

# ================== 5. TOP N por región ==================
def pick_top_by_reach(articles, top_spain=3, top_europe=3, top_world=5):
    groups = group_articles(articles)
    temas = []
    for group in groups:
        fuentes_totales = len({a["source"] for a in group})
        if fuentes_totales < MIN_SOURCES:
            continue
        region = classify_group_by_reach(group)
        idf = build_idf([tokenize(text_for_vector(a)) for a in group])
        vecs = [tfidf_vector(tokenize(text_for_vector(a)), idf) for a in group]
        if len(group) == 1:
            best = group[0]
        else:
            best_idx = max(range(len(group)),
                           key=lambda i: sum(cosine(vecs[i], vecs[j])
                                             for j in range(len(group)) if j != i))
            best = group[best_idx]
        regions = {}
        for a in group:
            regions.setdefault(a["region"], set()).add(a["source"])
        temas.append({
            "title": best["title"],
            "summary": best["summary"] or "(Sin resumen disponible)",
            "link": best["link"],
            "sources": fuentes_totales,
            "region": region,
            "debug": f"s={len(regions.get('spain', set()))} "
                     f"e={len(regions.get('europe', set()))} "
                     f"w={len(regions.get('world', set()))}",
        })

    print(f"\n🔎 Grupos que pasan el umbral ({MIN_SOURCES}+ fuentes): {len(temas)}")
    for t in temas:
        print(f"   [{t['region']:6}] {t['sources']}f ({t['debug']}) → {t['title'][:70]}")

    def top_n(region, n):
        f = [t for t in temas if t["region"] == region]
        f.sort(key=lambda x: x["sources"], reverse=True)
        return f[:n]

    return top_n("spain", top_spain), top_n("europe", top_europe), top_n("world", top_world)

# ================== 6. Enviar correo ==================
def send_email(top_spain, top_europe, top_world):
    if not (top_spain or top_europe or top_world):
        print("No hay artículos para enviar.")
        return
    msg = MIMEMultipart()
    msg["From"] = Header(SENDER_EMAIL)
    msg["To"] = Header(RECEIVER_EMAIL)
    msg["Subject"] = Header("☀️ Resumen matutino: TOP España / Europa / Mundo", "utf-8")
    lines = ["Buenos días,\n",
             f"Noticias más repetidas en las últimas 24 horas (mínimo {MIN_SOURCES} fuentes por noticia).\n"]

    def add_block(titulo, articulos):
        lines.append("\n" + "=" * 60)
        lines.append(titulo)
        lines.append("=" * 60 + "\n")
        if not articulos:
            lines.append(f"(No se encontraron noticias con al menos {MIN_SOURCES} fuentes)\n")
            return
        for i, art in enumerate(articulos, 1):
            lines.append(f"{i}. {art['title']}")
            lines.append(f"   Fuentes que cubren el tema: {art['sources']}")
            if art["summary"]:
                lines.append(f"   Resumen: {art['summary']}")
            lines.append(f"   Enlace: {art['link']}\n")

    add_block("🇪🇸 TOP 3 — España", top_spain)
    add_block("🇪🇺 TOP 3 — Europa", top_europe)
    add_block("🌍 TOP 5 — Mundo", top_world)
    msg.attach(MIMEText("\n".join(lines), "plain", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        print("✅ Correo enviado correctamente.")
    except Exception as e:
        print(f"❌ Error al enviar: {e}")

# ================== EJECUCIÓN ==================
if __name__ == "__main__":
    print("🚀 Iniciando resumen matutino...")
    all_articles = fetch_rss_articles()
    print(f"📰 Total de artículos: {len(all_articles)}")
    top_spain, top_europe, top_world = pick_top_by_reach(all_articles)
    print(f"\n🇪🇸 España: {len(top_spain)} | 🇪🇺 Europa: {len(top_europe)} | 🌍 Mundo: {len(top_world)}")
    send_email(top_spain, top_europe, top_world)