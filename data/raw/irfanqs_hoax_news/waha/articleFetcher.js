// ╔══════════════════════════════════════════════════════════════╗
// ║  articleFetcher.js                                            ║
// ║  Fetch URL artikel, ekstrak teks bersih dari HTML, lalu pilih  ║
// ║  paragraf yg paling mirip klaim. Dipakai saat eskalasi.       ║
// ╚══════════════════════════════════════════════════════════════╝
const axios = require('axios');
const cheerio = require('cheerio');

// Header browser-like agar lebih jarang ditolak situs berita
const FETCH_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml',
  'Accept-Language': 'id-ID,id;q=0.9,en;q=0.8',
};

// Cache sederhana per-proses (URL -> teks artikel)
const _cache = new Map();
const CACHE_MAX = 200;   // batas entri agar memori tdk meledak
function _cacheSet(k, v) {
  if (_cache.size >= CACHE_MAX) _cache.delete(_cache.keys().next().value);
  _cache.set(k, v);
}

// Selector kandidat utk konten utama (urut prioritas)
const CONTENT_SELECTORS = [
  'article',
  '[itemprop="articleBody"]',
  '.detail__body-text', '.read__content', '.detail-content',  // pola berita ID umum
  'main',
  '#content',
];

// Selector noise yg HARUS dibuang sebelum ekstraksi
const NOISE_SELECTORS = [
  'script', 'style', 'noscript', 'iframe', 'svg',
  'nav', 'header', 'footer', 'aside',
  '.related', '.related-news', '.baca-juga', '.tag', '.tags',
  '.share', '.social', '.comment', '.comments',
  '.advertisement', '.ad', '.ads', '[class*="iklan"]',
  '.breadcrumb', '.author', '.metadata',
];

function _cleanText(s) {
  return String(s).replace(/\s+/g, ' ').replace(/\u00a0/g, ' ').trim();
}

/**
 * Ekstrak teks artikel dari HTML.
 * Strategi: buang noise -> coba selector konten utama -> fallback ke <p> terbanyak.
 * Mengembalikan { fullText, paragraphs[] } atau null jika gagal mendapatkan teks berarti.
 */
function extractArticleText(html) {
  let $;
  try { $ = cheerio.load(html); } catch { return null; }

  NOISE_SELECTORS.forEach(sel => $(sel).remove());

  // Coba selector spesifik dulu
  let container = null;
  for (const sel of CONTENT_SELECTORS) {
    const el = $(sel).first();
    if (el.length && _cleanText(el.text()).length > 300) { container = el; break; }
  }

  // Fallback: ambil semua <p> di body
  let paragraphs = [];
  if (container) {
    container.find('p').each((_, el) => {
      const t = _cleanText($(el).text());
      if (t.length >= 40) paragraphs.push(t);   // buang <p> pendek (caption, dll)
    });
  }
  if (paragraphs.length < 2) {
    paragraphs = [];
    $('body p').each((_, el) => {
      const t = _cleanText($(el).text());
      if (t.length >= 40) paragraphs.push(t);
    });
  }

  if (paragraphs.length === 0) return null;

  // Dedupe paragraf identik (sering muncul di template berita ID)
  const seen = new Set();
  paragraphs = paragraphs.filter(p => { if (seen.has(p)) return false; seen.add(p); return true; });

  const fullText = paragraphs.join('\n');
  if (fullText.length < 200) return null;       // terlalu pendek = ekstraksi gagal

  return { fullText, paragraphs };
}

/**
 * Fetch satu URL dgn timeout. Mengembalikan { url, paragraphs, fullText } atau
 * { url, error } jika gagal. TIDAK pernah throw — pemanggil tdk perlu try/catch.
 */
async function fetchArticle(url, timeoutMs = 8000) {
  if (_cache.has(url)) return _cache.get(url);

  try {
    const res = await axios.get(url, {
      headers: FETCH_HEADERS,
      timeout: timeoutMs,
      maxRedirects: 3,
      responseType: 'text',
      validateStatus: s => s >= 200 && s < 400,
      // Beberapa situs kirim gzip; axios meng-handle otomatis.
    });
    const html = res.data;
    if (typeof html !== 'string' || html.length < 500) {
      const out = { url, error: 'response too short' };
      _cacheSet(url, out); return out;
    }
    const extracted = extractArticleText(html);
    if (!extracted) {
      const out = { url, error: 'no article text extracted' };
      _cacheSet(url, out); return out;
    }
    const out = { url, ...extracted };
    _cacheSet(url, out);
    return out;
  } catch (err) {
    const out = { url, error: err.code || err.message || 'fetch failed' };
    _cacheSet(url, out);
    return out;
  }
}

/**
 * Fetch banyak URL paralel; selalu kembalikan array hasil per-URL.
 * Sukses & gagal sama-sama masuk supaya pemanggil bisa decide fallback.
 */
async function fetchArticlesParallel(urls, timeoutMs = 8000) {
  return Promise.all(urls.map(u => fetchArticle(u, timeoutMs)));
}

/**
 * Cosine TF leksikal — duplikasi mini dari fakecheck.js supaya modul ini
 * tidak punya circular import. Stopword Indonesia disederhanakan.
 */
const _STOP = new Set(['yang','di','ke','dari','dan','atau','itu','ini','dengan','untuk','pada','adalah','akan','tidak','juga','sudah','para','oleh','karena','sebagai','dalam','telah','bahwa','seorang','sebuah','ada','saat','setelah','sebelum','tentang','namun','tetapi','lebih','sangat','hanya','masih','dapat','tersebut','adanya']);
function _tok(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9\s]/g,' ').split(/\s+/)
    .filter(w => w.length > 2 && !_STOP.has(w));
}
function _tfCos(a, b) {
  const tf = (toks) => { const m=new Map(); toks.forEach(t=>m.set(t,(m.get(t)||0)+1)); return m; };
  const x = tf(_tok(a)), y = tf(_tok(b));
  const terms = new Set([...x.keys(), ...y.keys()]);
  let dot=0, m1=0, m2=0;
  for (const t of terms) { const u=x.get(t)||0, v=y.get(t)||0; dot+=u*v; m1+=u*u; m2+=v*v; }
  return (m1 && m2) ? dot/(Math.sqrt(m1)*Math.sqrt(m2)) : 0;
}

/**
 * Pilih `topN` paragraf paling mirip dgn klaim dari sebuah artikel.
 * Mengembalikan string siap-kirim (digabung dgn newline), max ~1500 char
 * agar muat di max_length 256 token NLI/NER.
 */
function selectRelevantParagraphs(paragraphs, claim, topN = 3, maxChars = 1500) {
  if (!Array.isArray(paragraphs) || paragraphs.length === 0) return '';
  const scored = paragraphs.map(p => ({ p, s: _tfCos(p, claim) }));
  scored.sort((a, b) => b.s - a.s);
  const out = [];
  let acc = 0;
  for (const { p } of scored) {
    if (out.length >= topN) break;
    if (acc + p.length > maxChars) {
      // potong paragraf terakhir agar pas
      const room = maxChars - acc;
      if (room > 80) out.push(p.slice(0, room) + '…');
      break;
    }
    out.push(p);
    acc += p.length;
  }
  return out.join('\n');
}

module.exports = {
  fetchArticle,
  fetchArticlesParallel,
  extractArticleText,
  selectRelevantParagraphs,
};