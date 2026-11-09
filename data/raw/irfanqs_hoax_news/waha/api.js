// Kode utama
const axios = require('axios');
require('dotenv').config();
const { fetchArticlesParallel, selectRelevantParagraphs } = require('./articleFetcher');

const FACT_CHECK_URL = 'https://factchecktools.googleapis.com/v1alpha1/claims:search';
const SERP_API_URL = 'https://serpapi.com/search';

// ── WAHA: Kirim pesan teks ────────────────────────────────────
async function sendMessage(chatId, text) {
  await axios.post(`${process.env.WAHA_API_URL}/api/sendText`, {
    chatId,
    text,
    session: process.env.WAHA_SESSION || 'default'
  }, {
    timeout: 10000,
    headers: { 'X-Api-Key': process.env.WAHA_API_KEY }
  });
}

// ── BART Summarize ────────────────────────────────────────────
async function summarize(text) {
  const res = await axios.post(`${process.env.BART_API_URL}/summarize`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 60000
  });
  return res.data.summary;
}

// ── BERT Classification ───────────────────────────────────────
async function classifyBert(text) {
  const res = await axios.post(`${process.env.BERT_API_URL}/predict`, { text }, {
    headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 60000
  });
  return res.data;
}

// ── Google Fact Check API ─────────────────────────────────────
async function factCheck(query) {
  const res = await axios.get(FACT_CHECK_URL, {
    params: { query, key: process.env.FACT_CHECK_API_KEY, languageCode: 'id' }, timeout: 20000
  });
  const claims = res.data.claims || [];
  return claims.slice(0, 3).map(claim => ({
    klaim: claim.text || '',
    rating: claim.claimReview?.[0]?.textualRating || '',
    sumber: claim.claimReview?.[0]?.publisher?.name || '',
    url: claim.claimReview?.[0]?.url || ''
  }));
}

// ── SerpAPI Search: ambil 10, buang medsos ────────────────────
const SOCIAL_DOMAINS = [
  'instagram.com','facebook.com','fb.com','fb.watch','m.facebook.com',
  'twitter.com','x.com','t.co','tiktok.com','threads.net','threads.com'
];
function isSocialMedia(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, '').toLowerCase();
    return SOCIAL_DOMAINS.some(d => host === d || host.endsWith('.' + d));
  } catch { return false; }
}
async function searchSerpApi(query) {
  const res = await axios.get(SERP_API_URL, {
    params: { engine: 'google', q: `cek fakta ${query}`,
              api_key: process.env.SERPAPI_KEY, num: 10, hl: 'id', gl: 'id' },
    timeout: 15000
  });
  const organicResults = res.data.organic_results || [];
  return organicResults
    .filter(r => r.snippet && r.link && !isSocialMedia(r.link))
    .slice(0, 10)
    .map(r => ({ judul: r.title || '', snippet: r.snippet || '', url: r.link || '' }));
}

// ── Keyword patterns ──────────────────────────────────────────
const MEDIA_NOUN = '(?:berita|kabar|informasi|info|foto|video|gambar|klaim|narasi|unggahan|postingan|konten|pesan)';
const HOAX_PATTERNS = [
  /\bhoaks?\b/i, /\b(?:di|meng|peng)hoaks?\b/i,
  /\bdisinformasi\b/i, /\bmisinformasi\b/i,
  /\btidak\s+benar\b/i, /\btidak\s+terbukti\b/i,
  /\bmenyesatkan\b/i, /\bkeliru\b/i, /\bkekeliruan\b/i,
  /\[\s*(?:salah|hoaks?|disinformasi|misinformasi|keliru|penipuan)\s*\]/i,
  new RegExp(`\\b${MEDIA_NOUN}\\s+\\w*\\s*(?:palsu|bohong|hoaks?|menyesatkan|keliru)\\b`, 'i'),
  /\bkabar\s+bohong\b/i,
];
const FACT_PATTERNS = [
  /\bfaktanya\b/i, /\bbenar\s+adanya\b/i, /\bterbukti\s+benar\b/i, /\bdipastikan\s+benar\b/i,
  /\bmembenarkan\b/i, /\bdibenarkan\b/i, /\bterverifikasi\b/i, /\bvalid(?:asi)?\b/i,
  /\bsahih\b/i, /\bakurat\b/i, /\bkeakuratan\b/i,
  /\[\s*(?:benar|fakta|valid)\s*\]/i,
];
function hasPattern(text, patterns) { return patterns.some(p => p.test(text)); }
function articleHasHoaxKeyword(r) { return hasPattern(`${r.judul} ${r.snippet}`.toLowerCase(), HOAX_PATTERNS); }
function articleHasFactKeyword(r) { return hasPattern(`${r.judul} ${r.snippet}`.toLowerCase(), FACT_PATTERNS); }

// ── Regex NER fallback (jika /ner gagal) ──────────────────────
function extractNumbers(text) {
  const matches = text.match(/\b\d[\d.,]*\b/g) || [];
  return matches
    .map(n => parseFloat(n.replace(/\./g, '').replace(',', '.')))
    .filter(n => !isNaN(n) && n >= 10 && (n < 1900 || n > 2100));
}
function detectNumberMismatch(summary, serpResults) {
  const summaryNums = extractNumbers(summary);
  if (summaryNums.length === 0) return { mismatch: false };
  const allArticleText = serpResults.map(r => r.judul + ' ' + r.snippet).join(' ');
  const articleNums = extractNumbers(allArticleText);
  if (articleNums.length === 0) return { mismatch: false };
  const mismatched = summaryNums.filter(sNum =>
    !articleNums.some(aNum => (sNum > aNum ? sNum/aNum : aNum/sNum) <= 1.2)
  );
  return { mismatch: mismatched.length > 0, mismatchedNums: mismatched };
}
function formatValueDisplay(_, value) { return value; }

// ── Similarity ────────────────────────────────────────────────
const ID_STOPWORDS = new Set([
  'yang','di','ke','dari','dan','atau','itu','ini','dengan','untuk','pada','adalah',
  'akan','tidak','juga','sudah','saya','anda','dia','mereka','kita','kami','para',
  'oleh','karena','sebagai','dalam','telah','bahwa','agar','seorang','sebuah','suatu',
  'ada','saat','setelah','sebelum','hingga','sampai','antara','tentang','namun','tetapi',
  'lebih','sangat','hanya','masih','bisa','dapat','harus','akibat','membuat',
  'bagaimana','bersangkutan','adanya','tersebut','ialah','yaitu'
]);
function tokenize(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)
    .filter(w => w.length > 2 && !ID_STOPWORDS.has(w));
}
function lexicalCosine(text1, text2) {
  const tf = (toks) => { const m = new Map(); toks.forEach(t => m.set(t,(m.get(t)||0)+1)); return m; };
  const a = tf(tokenize(text1)), b = tf(tokenize(text2));
  const terms = new Set([...a.keys(), ...b.keys()]);
  let dot=0, m1=0, m2=0;
  for (const t of terms) { const x=a.get(t)||0, y=b.get(t)||0; dot+=x*y; m1+=x*x; m2+=y*y; }
  return (m1 && m2) ? dot/(Math.sqrt(m1)*Math.sqrt(m2)) : 0;
}
function vectorCosine(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return 0;
  let dot=0, m1=0, m2=0;
  for (let i=0; i<a.length; i++) { dot+=a[i]*b[i]; m1+=a[i]*a[i]; m2+=b[i]*b[i]; }
  return (m1 && m2) ? dot/(Math.sqrt(m1)*Math.sqrt(m2)) : 0;
}
async function embedTexts(texts) {
  try {
    const res = await axios.post(`${process.env.BERT_API_URL}/embed`, { texts }, {
      headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 20000
    });
    const emb = res.data?.embeddings;
    if (Array.isArray(emb) && emb.length === texts.length && Array.isArray(emb[0])) return emb;
    return null;
  } catch (err) { console.error('[embedTexts] fallback ke leksikal:', err.message); return null; }
}
async function similarityScores(summary, serpResults) {
  const articleTexts = serpResults.map(r => `${r.judul} ${r.snippet}`);
  const emb = await embedTexts([summary, ...articleTexts]);
  if (emb) {
    const [sVec, ...aVecs] = emb;
    return { method: 'embedding', scores: aVecs.map(v => vectorCosine(sVec, v)) };
  }
  return { method: 'lexical', scores: articleTexts.map(t => lexicalCosine(summary, t)) };
}

// ── NER / NLI calls ───────────────────────────────────────────
const CHECKABLE_NER_CATEGORIES = new Set(['TIM','DATE','MON','PRC','PERCENT','QTY','QUANTITY']);
function meaningfulMismatches(nerResult) {
  if (!nerResult || !Array.isArray(nerResult.mismatches)) return [];
  return nerResult.mismatches.filter(m => {
    const cat = String(m.category || '').toUpperCase();
    if (!CHECKABLE_NER_CATEGORIES.has(cat)) return false;
    const ev = Array.isArray(m.evidence_values) ? m.evidence_values : [];
    return ev.length > 0;
  });
}
async function nerCheck(summary, serpResults) {
  try {
    const articles = serpResults.map(r => r.judul + ' ' + r.snippet);
    const res = await axios.post(`${process.env.BERT_API_URL}/ner`, { summary, articles }, {
      headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 15000
    });
    return res.data;
  } catch (err) { console.error('[nerCheck] error:', err.message); return null; }
}
async function nliCheck(summary, serpResults) {
  try {
    const evidences = serpResults.map(r => `${r.judul} ${r.snippet}`);
    const res = await axios.post(`${process.env.BERT_API_URL}/nli`, { claim: summary, evidences }, {
      headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 20000
    });
    return res.data;
  } catch (err) { console.error('[nliCheck] error:', err.message); return null; }
}
// Cari index artikel dgn skor kontradiksi tertinggi
function pickContradictingIdx(nli) {
  if (!nli || !Array.isArray(nli.per_evidence)) return -1;
  let best = -1, bestIdx = -1;
  nli.per_evidence.forEach((pe, i) => {
    const s = pe.scores || {};
    const c = Math.max(0, ...Object.entries(s)
      .filter(([k]) => /contra|kontradik/i.test(k) || String(k).toUpperCase() === 'LABEL_2')
      .map(([, v]) => Number(v)));
    if (c > best) { best = c; bestIdx = i; }
  });
  return bestIdx;
}
// Util: bentuk objek artikel-bukti utk dilaporkan ke user
function asEvidence(a, role) {
  return { judul: a.judul, url: a.url, sim: a.sim != null ? Number(a.sim.toFixed(3)) : null, role };
}

// ── Google Fact Check: jadikan PUTUSAN, bukan sekadar tampilan ──
// Rating dari lembaga cek-fakta (TurnBackHoax, Tirto, dsb) bersifat otoritatif.
// Kosakata rating beragam (id & en), jadi klasifikasi terpisah dari pola artikel.
const _FC_FALSE = [
  /\bsalah\b/i, /\bhoaks?\b/i, /\bhoax\b/i, /\bfalse\b/i, /\bmisleading\b/i,
  /\bmenyesatkan\b/i, /\btidak\s+benar\b/i, /\bkeliru\b/i, /\bpalsu\b/i,
  /\bbohong\b/i, /\bdisinformasi\b/i, /\bmisinformasi\b/i, /\bfabricated\b/i,
  /\bparod(?:i|y)\b/i, /\bscam\b/i, /\btipuan?\b/i,
];
const _FC_TRUE = [
  /\bbenar\b/i, /\btrue\b/i, /\bfakta\b/i, /\bvalid\b/i, /\bakurat\b/i, /\bterbukti\b/i,
];
function classifyFactRating(rating) {
  const r = String(rating || '').toLowerCase().trim();
  if (!r) return null;
  // "sebagian benar/salah", "partly", "mixed", "missing context" = ambigu -> abstain
  if (/\bsebagian\b|\bpartly\b|\bhalf\b|\bcampuran\b|\bmixed\b|missing context/.test(r)) return null;
  if (_FC_FALSE.some(p => p.test(r))) return 'HOAKS';   // cek FALSE dulu ("tidak benar" memuat "benar")
  if (_FC_TRUE.some(p => p.test(r)))  return 'BENAR';
  return null;
}
function classifyFactResults(factResults) {
  const items = Array.isArray(factResults) ? factResults : [];
  let hoax = 0, fakta = 0;
  const evidence = [];
  for (const f of items) {
    const v = classifyFactRating(f.rating);
    if (!v) continue;
    if (v === 'HOAKS') hoax++; else fakta++;
    evidence.push({ judul: f.klaim || f.sumber || 'Hasil cek fakta', url: f.url || '',
                    rating: f.rating || '', sumber: f.sumber || '', role: v === 'HOAKS' ? 'debunk' : 'confirm' });
  }
  if (hoax === 0 && fakta === 0) return { label: null, hoax, fakta, evidence };
  if (hoax > fakta) return { label: 'HOAKS', hoax, fakta, evidence };
  if (fakta > hoax) return { label: 'BENAR', hoax, fakta, evidence };
  // Seri antar-lembaga -> jangan paksakan verdict keras
  return { label: null, hoax, fakta, evidence };
}

// Apakah ringkasan memuat entitas terverifikasi (tanggal/angka) yg TIDAK punya
// pembanding sekategori di bukti? -> klaim spesifik yg belum terkoroborasi.
const _CHECKABLE_CATS = new Set(['DATE','DAT','TIM','MON','PRC','PERCENT','QTY','QUANTITY']);
function hasUncorroboratedCheckable(nerRes) {
  if (!nerRes) return false;
  const norm = c => String(c || '').toUpperCase();
  const userCats = (nerRes.user_entities || [])
    .map(e => norm(e.category)).filter(c => _CHECKABLE_CATS.has(c));
  if (userCats.length === 0) return false;
  const evCats = new Set((nerRes.evidence_entities || []).map(e => norm(e.category)));
  return userCats.some(c => !evCats.has(c));
}

// ── Decision Fusion ───────────────────────────────────────────
async function decisionFusion(serpResults, bertResult, factResults = [], opts = {}) {
  const { keywordThreshold = 1, embThreshold = 0.65, lexThreshold = 0.20,
          nerHardVerdict = true, contraThreshold = 0.7, topK = 5,
          // Eskalasi ke full-artikel hanya saat verdict "ragu". Default ON.
          enableEscalation = true,
          // Ambang "ragu": NLI 0.4–0.7 (kontradiksi sedang) ATAU
          // similarity sedang (0.4–0.65) yg mau jadi BENAR begitu saja.
          escalationContraMin = 0.40, escalationSimMax = 0.65,
          articleFetchTimeoutMs = 8000,
          articleParagraphCount = 3, articleMaxChars = 1500 } = opts;

  // ── Flag tahap (default semua ON) ──
  const { stages = {} } = opts;
  const S = {
    factcheck:  stages.factcheck  !== false,
    keyword:    stages.keyword    !== false,
    similarity: stages.similarity !== false,
    nli:        stages.nli        !== false,
    ner:        stages.ner        !== false,
    ...stages,
  };

  const summary = bertResult?._summary || '';

  // ── Step 0: Google Fact Check = sumber paling otoritatif ──
  // Dijalankan SEBELUM apa pun, dan tetap berlaku walau SerpAPI kosong.
  if (S.factcheck) {
    const fc = classifyFactResults(factResults);
    if (fc.label) {
      return { label: fc.label, source: 'factcheck', E: fc.hoax, T: fc.fakta,
               diff: Math.abs(fc.hoax - fc.fakta),
               simMethod: null, avgSim: null, maxSim: null,
               displayArticles: [], evidenceArticles: [], factEvidence: fc.evidence };
    }
  }

  if (!serpResults || serpResults.length === 0) {
    return { label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
             source: 'bert_only', lowConfidence: true, E: 0, T: 0, diff: 0,
             simMethod: null, avgSim: null, maxSim: null,
             displayArticles: [], evidenceArticles: [] };
  }

  // Similarity SEKALI utk ranking + pelaporan
  const { method, scores } = await similarityScores(summary, serpResults);
  const ranked = serpResults.map((r, i) => ({ ...r, sim: scores[i] })).sort((a, b) => b.sim - a.sim);
  const topArticles = ranked.slice(0, Math.min(topK, ranked.length));
  const displayArticles = topArticles.map(a => ({
    judul: a.judul, snippet: a.snippet, url: a.url, sim: Number(a.sim.toFixed(3))
  }));
  const maxSim = ranked.length ? ranked[0].sim : 0;
  const avgSim = scores.reduce((a, b) => a + b, 0) / scores.length;
  const simThreshold = method === 'embedding' ? embThreshold : lexThreshold;

  // ── Step 1: Keyword (atas SEMUA hasil) ──
  let E = 0, T = 0;
  if (S.keyword) {
    const hoaxKeywordArticles = [], factKeywordArticles = [];
    ranked.forEach(r => {
      if (articleHasHoaxKeyword(r)) { E++; hoaxKeywordArticles.push(r); }
      else if (articleHasFactKeyword(r)) { T++; factKeywordArticles.push(r); }
    });
    if (E > 0 || T > 0) {
      const diff = Math.abs(E - T);
      if (diff >= keywordThreshold) {
        const winners = E > T ? hoaxKeywordArticles : factKeywordArticles;
        return { label: E > T ? 'HOAKS' : 'BENAR', source: 'external', E, T, diff,
                 simMethod: method, avgSim: avgSim.toFixed(3), maxSim: maxSim.toFixed(3),
                 displayArticles, evidenceArticles: winners.slice(0, 3).map(a => asEvidence(a, 'keyword-cek-fakta')) };
      }
      return { label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
               source: 'bert', lowConfidence: true, E, T, diff,
               simMethod: method, avgSim: avgSim.toFixed(3), maxSim: maxSim.toFixed(3),
               displayArticles, evidenceArticles: [] };
    }
  }

  // ── Step 2: Similarity gate ──
  // Kalau similarity dimatikan, tak ada gerbang relevansi -> tak ada dasar
  // untuk NLI/NER, sehingga langsung pakai BERT.
  if (!S.similarity) {
    return { label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
             source: 'bert_only', lowConfidence: true, E, T, diff: 0,
             simMethod: method, avgSim: avgSim.toFixed(3), maxSim: maxSim.toFixed(3),
             displayArticles, evidenceArticles: [] };
  }
  if (maxSim < simThreshold) {
    return { label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
             source: 'bert_only', lowConfidence: true, E, T, diff: 0,
             simMethod: method, avgSim: avgSim.toFixed(3), maxSim: maxSim.toFixed(3),
             displayArticles, evidenceArticles: [] };
  }

  const relevantArticles = topArticles.filter(a => a.sim >= simThreshold);
  const corroboratingEv = (relevantArticles.length ? relevantArticles : topArticles.slice(0, 1))
    .map(a => asEvidence(a, 'corroborates'));

  // ─────────────────────────────────────────────────────────────
  // Helper: jalankan NLI + NER atas serangkaian artikel.
  // nliCheck/nerCheck hanya dipanggil jika tahapnya aktif (lebih cepat).
  // ─────────────────────────────────────────────────────────────
  async function analyzeArticles(articles) {
    const nliRes = S.nli ? await nliCheck(summary, articles) : null;
    const nerRes = S.ner ? await nerCheck(summary, articles) : null;
    let detail = '';
    let nerSrcIdxs = [];
    if (nerRes) {
      const mm = meaningfulMismatches(nerRes);
      if (mm.length > 0) {
        detail = mm.map(m =>
          `${m.category}: ${m.user_text} (user: ${formatValueDisplay(m.category, m.user_value)}, bukti: ${m.evidence_values.join('/')})`
        ).join(' | ');
        const set = new Set();
        mm.forEach(m => (m.evidence_source_idxs || []).forEach(i => set.add(i)));
        nerSrcIdxs = [...set];
      }
    } else if (S.ner) {
      // Fallback regex hanya relevan kalau NER aktif tapi /ner gagal.
      const dn = detectNumberMismatch(summary, articles);
      if (dn.mismatch) detail = (dn.mismatchedNums || []).join(', ');
    }
    return { nli: nliRes, ner: nerRes, nerDetail: detail, nerSrcIdxs };
  }

  // ── Step 3 (tahap-1): analisis pakai SNIPPET dulu ──
  let analysis = await analyzeArticles(topArticles);
  let usedFullArticles = false;
  let fetchStats = null;       // { tried, succeeded, failed: [{url, error}] }

  // Cek apakah verdict "ragu" -> eskalasi
  const nliContraNow = analysis.nli?.max_contradiction ?? 0;
  const nliEntailNow = analysis.nli?.max_entailment   ?? 0;
  const verdictWouldBe =
        (nliContraNow >= contraThreshold && nliContraNow > nliEntailNow) ? 'HOAKS-nli'
      : (analysis.nerDetail ? 'HOAKS-ner' : 'BENAR');

  // BENAR yg punya entitas spesifik (tanggal/angka) TANPA pembanding di snippet
  // = klaim yg belum terverifikasi. Buka artikel penuh.
  const uncorroborated = (verdictWouldBe === 'BENAR')
                         && hasUncorroboratedCheckable(analysis.ner);

  const isAmbiguous = (
    // NLI ragu: skor kontradiksi di "zona abu-abu" (sedang) -> eskalasi
    (nliContraNow >= escalationContraMin && nliContraNow < contraThreshold)
    // BENAR tapi klaim memuat tanggal/angka yg tak ada di snippet -> eskalasi.
    || uncorroborated
  );

  if (enableEscalation && isAmbiguous) {
    const urls = topArticles.map(a => a.url).filter(Boolean);
    const fetched = await fetchArticlesParallel(urls, articleFetchTimeoutMs);
    const failedList = fetched.filter(f => f.error).map(f => ({ url: f.url, error: f.error }));
    fetchStats = { tried: urls.length, succeeded: fetched.length - failedList.length, failed: failedList };
    // Bentuk ulang topArticles dgn snippet diganti paragraf paling relevan
    // (HANYA utk yg sukses; yg gagal tetap pakai snippet lama).
    const enrichedArticles = topArticles.map((a, i) => {
      const f = fetched[i];
      if (!f || f.error || !Array.isArray(f.paragraphs)) return a;
      const enriched = selectRelevantParagraphs(f.paragraphs, summary,
                                                articleParagraphCount, articleMaxChars);
      if (!enriched) return a;
      // judul tetap; snippet = paragraf relevan; ini yg dipakai utk NLI+NER
      return { ...a, snippet: enriched };
    });
    if (fetchStats.succeeded > 0) {
      analysis = await analyzeArticles(enrichedArticles);
      usedFullArticles = true;
    }
  }

  // ── Putuskan akhir dari analisis (snippet atau full) ──
  const evidenceMeta = { simMethod: method, avgSim: avgSim.toFixed(3), maxSim: maxSim.toFixed(3),
                         analyzedFrom: usedFullArticles ? 'full_article' : 'snippet',
                         fetchStats };

  if (S.nli && analysis.nli && analysis.nli.max_contradiction >= contraThreshold
      && analysis.nli.max_contradiction > analysis.nli.max_entailment) {
    const idx = pickContradictingIdx(analysis.nli);
    const contraEv = (idx >= 0 && topArticles[idx]) ? [asEvidence(topArticles[idx], 'contradicts')] : corroboratingEv;
    return { label: 'HOAKS', source: 'nli', E, T, diff: 0, ...evidenceMeta,
             nliContra: analysis.nli.max_contradiction.toFixed(2),
             nliEntail: analysis.nli.max_entailment.toFixed(2),
             displayArticles, evidenceArticles: contraEv };
  }

  if (S.ner && analysis.nerDetail) {
    const nerEv = analysis.nerSrcIdxs.length
      ? analysis.nerSrcIdxs.map(i => topArticles[i]).filter(Boolean).map(a => asEvidence(a, 'contradicts-fact'))
      : corroboratingEv;
    if (nerHardVerdict) {
      return { label: 'HOAKS', source: 'ner', E, T, diff: 0, ...evidenceMeta,
               nerDetail: analysis.nerDetail, displayArticles, evidenceArticles: nerEv };
    }
    return { label: bertResult ? bertResult.label : 'TIDAK DAPAT DIPASTIKAN',
             source: 'ner_soft', lowConfidence: true, E, T, diff: 0, ...evidenceMeta,
             nerDetail: analysis.nerDetail, displayArticles, evidenceArticles: nerEv };
  }

  return { label: 'BENAR', source: 'similarity', E, T, diff: 0, ...evidenceMeta,
           displayArticles, evidenceArticles: corroboratingEv };
}

// ── Format hasil ──────────────────────────────────────────────
function stripMd(text) {
  return String(text).replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1')
    .replace(/__(.*?)__/g, '$1').replace(/_(.*?)_/g, '$1').replace(/[_*`\[\]]/g, '');
}
function evidenceLines(evidenceArticles, displayArticles) {
  if (!Array.isArray(evidenceArticles) || evidenceArticles.length === 0) return '';
  // Cari nomor referensi yg ditampilkan di bawah (cocokkan via URL)
  return evidenceArticles.map(ev => {
    const idx = displayArticles.findIndex(d => d.url === ev.url);
    const numTag = idx >= 0 ? `[#${idx + 1}]` : `[?]`;
    return `${numTag} ${stripMd(ev.judul).slice(0, 80)}`;
  }).join('\n   ');
}
function formatResult(bertResult, factResults, serpResults, fusion = null) {
  const conf = bertResult.confidence != null ? parseFloat(bertResult.confidence) : null;
  const isHoaks = bertResult.label?.toUpperCase().includes('HOAKS') || bertResult.label?.toUpperCase().includes('HOAX');
  const hoaxPct = bertResult.hoax_probability != null
    ? (bertResult.hoax_probability * 100).toFixed(1)
    : conf != null ? (isHoaks ? conf.toFixed(1) : (100 - conf).toFixed(1)) : null;
  const validPct = bertResult.valid_probability != null
    ? (bertResult.valid_probability * 100).toFixed(1)
    : conf != null ? (isHoaks ? (100 - conf).toFixed(1) : conf.toFixed(1)) : null;
  const finalLabel = fusion ? fusion.label : bertResult.label;
  const lowConf = fusion && fusion.lowConfidence;
  const fusionOverride = fusion && (fusion.source === 'external' || fusion.source === 'factcheck')
                         && fusion.label !== bertResult.label;
  let verdictEmoji, verdictText;
  if (lowConf) { verdictEmoji = '❓'; verdictText = `PERLU VERIFIKASI (indikasi model: ${finalLabel})`; }
  else if (finalLabel === 'HOAKS') { verdictEmoji = '⛔'; verdictText = 'HOAKS'; }
  else if (finalLabel === 'BENAR') { verdictEmoji = '✅'; verdictText = 'KEMUNGKINAN FAKTA'; }
  else { verdictEmoji = '❓'; verdictText = 'TIDAK DAPAT DIPASTIKAN'; }

  let msg = `🔍 *Hasil Analisis Berita*\n\n${verdictEmoji} *VERDICT: ${verdictText}*\n`;

  // Badge eskalasi: kasih tahu user analisis pakai snippet atau full artikel
  if (fusion?.analyzedFrom === 'full_article') {
    const fs = fusion.fetchStats;
    const detail = fs ? ` (${fs.succeeded}/${fs.tried} artikel berhasil dibaca)` : '';
    msg += `🔬 _Analisis mendalam: isi artikel lengkap${detail}_\n`;
  } else if (fusion?.analyzedFrom === 'snippet') {
    msg += `📄 _Analisis berdasarkan snippet pencarian_\n`;
  }

  // Penjelasan + sumber bukti (selalu dengan referensi nomor [#n])
  const displayArticles = fusion?.displayArticles || [];
  const evLines = fusion ? evidenceLines(fusion.evidenceArticles, displayArticles) : '';
  if (!fusion) {
    msg += `_Diputuskan berdasarkan model BERT (tanpa pencarian eksternal)_\n`;
  } else if (fusion.source === 'factcheck') {
    msg += `_Diputuskan berdasarkan ${fusion.E + fusion.T} putusan lembaga cek fakta (Google Fact Check)._\n`;
    msg += `_(Rincian & tautan ada di bagian "Hasil Google Fact Check" di bawah.)_\n`;
  } else if (fusion.source === 'external') {
    msg += `_Diputuskan berdasarkan ${fusion.E + fusion.T} artikel cek-fakta eksternal_\n`;
    if (evLines) msg += `🧾 *Sumber putusan:*\n   ${evLines}\n`;
  } else if (fusion.source === 'similarity') {
    msg += `_Topik cocok dengan artikel tepercaya — tidak ditemukan kontradiksi atau pemalsuan angka._\n`;
    msg += `_(${fusion.simMethod}, kemiripan max=${fusion.maxSim})_\n`;
    if (evLines) msg += `🧾 *Diperkuat oleh artikel:*\n   ${evLines}\n`;
  } else if (fusion.source === 'ner') {
    msg += `_Berita mirip topik, TETAPI angka/tanggal tidak cocok dengan artikel._\n`;
    if (fusion.nerDetail) msg += `_Detil: ${stripMd(fusion.nerDetail)}_\n`;
    if (evLines) msg += `🧾 *Bukti pembanding angka/tanggal:*\n   ${evLines}\n`;
  } else if (fusion.source === 'nli') {
    msg += `_Klaim BERTENTANGAN dengan isi artikel tepercaya (kontradiksi makna)._\n`;
    if (fusion.nliContra) msg += `_Skor kontradiksi: ${fusion.nliContra} (entailment: ${fusion.nliEntail})_\n`;
    if (evLines) msg += `🧾 *Artikel yang berkontradiksi:*\n   ${evLines}\n`;
  } else if (fusion.source === 'ner_soft') {
    msg += `_Berita mirip; ada angka/tanggal yang belum cocok — perlu verifikasi manual._\n`;
    if (fusion.nerDetail) msg += `_Detil: ${stripMd(fusion.nerDetail)}_\n`;
    if (evLines) msg += `🧾 *Bukti pembanding:*\n   ${evLines}\n`;
  } else if (fusion.source === 'bert_only') {
    const why = fusion.maxSim != null
      ? `tidak ada artikel cukup relevan (sim ${fusion.simMethod}: max=${fusion.maxSim})`
      : `tidak ada hasil pencarian`;
    msg += `_⚠️ Tidak dapat dikoroborasi: ${why}. Label di atas hanya dugaan model BERT — BUKAN bukti. Verifikasi manual._\n`;
  } else {
    msg += `_Bukti eksternal seri → mengikuti model BERT (perlu verifikasi)_\n`;
  }

  if (fusionOverride) {
    const srcLabel = fusion.source === 'factcheck' ? 'putusan lembaga cek fakta' : 'artikel eksternal';
    msg += `\n⚠️ _Catatan: Model BERT mendeteksi ${bertResult.label} (${hoaxPct}%), namun ${fusion.E + fusion.T} ${srcLabel} menunjukkan sebaliknya._\n`;
  }

  msg += `\n${'─'.repeat(28)}\n📊 *Detail Model BERT:*\n`;
  msg += `├─ 🚨 Hoax  : *${hoaxPct ?? '?'}%*\n└─ ✅ Fakta : *${validPct ?? '?'}%*\n`;
  if (fusion) {
    msg += `\n📊 *Bukti Eksternal:*\n├─ 🚨 Artikel hoaks : ${fusion.E}\n└─ ✅ Artikel fakta : ${fusion.T}\n`;
  }

  if (factResults && factResults.length > 0) {
    msg += `\n${'─'.repeat(28)}\n🗂️ *Hasil Google Fact Check:*\n\n`;
    factResults.forEach((r, i) => {
      msg += `*${i + 1}.* ${stripMd(r.klaim)}\n   📌 Rating: ${stripMd(r.rating)}\n   🏛️ Sumber: ${stripMd(r.sumber)}\n`;
      if (r.url) msg += `   [🔗 Lihat Sumber](${r.url})\n`;
      msg += '\n';
    });
  }

  const refs = displayArticles.length ? displayArticles : (serpResults || []).slice(0, 5);
  if (refs.length > 0) {
    msg += `${'─'.repeat(28)}\n📎 *Referensi paling relevan (${refs.length}):*\n_Diurutkan dari yang paling mirip topik._\n\n`;
    refs.forEach((r, i) => {
      const rel = r.sim != null ? ` _(relevansi ${r.sim})_` : '';
      msg += `*[#${i + 1}]* ${stripMd(r.judul)}${rel}\n📄 _Snippet:_ ${stripMd(r.snippet)}\n`;
      if (r.url) msg += `[🔗 Baca Selengkapnya](${r.url})\n`;
      msg += '\n';
    });
  } else {
    msg += `\n📭 _Tidak ditemukan referensi (atau semua hasil dari media sosial & dibuang)._\n`;
  }

  return msg;
}

module.exports = { sendMessage, summarize, classifyBert, factCheck, searchSerpApi, formatResult, decisionFusion, classifyFactResults };