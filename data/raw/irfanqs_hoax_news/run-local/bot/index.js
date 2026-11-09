require('dotenv').config();
const express = require('express');
const { sendMessage, summarize, classifyBert, factCheck, searchSerpApi, formatResult, decisionFusion } = require('./api');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;

// ── Health check ──────────────────────────────────────────────

app.get('/', (req, res) => res.json({ status: 'ok', service: 'hoax-checker-waha' }));

// ── WAHA Webhook ──────────────────────────────────────────────

app.post('/webhook', async (req, res) => {
  // Balas 200 segera agar WAHA tidak retry
  res.sendStatus(200);

  const { event, payload } = req.body || {};

  // Hanya proses event "message" dengan isi teks
  if (event !== 'message') return;
  const chatId = payload?.from;
  const text = payload?.body?.trim();
  if (!chatId || !text) return;

  // Abaikan pesan dari diri sendiri (status update, dll.)
  if (payload?.fromMe) return;

  console.log(`[webhook] from=${chatId} text="${text.slice(0, 60)}..."`);

  try {
    await sendMessage(chatId, '⏳ Memproses teks kamu...');

    // ── Step 1: BERT ─────────────────────────────────────────
    await sendMessage(chatId, '🧠 Step 1/3: Analisis Semantik (BERT)...');

    let bertResult = null;
    try {
      bertResult = await classifyBert(text);
      const emoji = bertResult.label === 'BENAR' ? '✅' : '❌';
      await sendMessage(chatId,
        `${emoji} Hasil Semantik BERT: ${bertResult.label}\nKepercayaan: ${bertResult.confidence}%`
      );
    } catch (bertErr) {
      console.error('[bert] error:', bertErr.message);
      await sendMessage(chatId, '⚠️ BERT tidak tersedia, melanjutkan ke langkah berikutnya...');
    }

    // ── Step 2: BART Summarize ────────────────────────────────
    await sendMessage(chatId, '📝 Step 2/3: Meringkas Teks (BART)...');

    let summary = text;
    try {
      summary = await summarize(text);
      await sendMessage(chatId, `Ringkasan:\n${summary}`);
    } catch (bartErr) {
      console.error('[bart] error:', bartErr.message);
      await sendMessage(chatId, '⚠️ BART tidak tersedia, menggunakan teks asli untuk fact check...');
    }

    // ── Step 3: Fact Check + AI Verdict ─────────────────────
    await sendMessage(chatId, '🔎 Step 3/3: Pengecekan Fakta...');

    const factResults = await factCheck(summary);
    let serpResults = factResults.length === 0 ? await searchSerpApi(summary) : [];

    if (factResults.length === 0 && serpResults.length === 0) {
      await sendMessage(chatId, 'Tidak ditemukan informasi terkait klaim ini di internet.');
    } else {
      const fusion = serpResults.length > 0
        ? await decisionFusion(serpResults, { ...bertResult, _summary: summary })
        : null;

      const resultMsg = formatResult(bertResult, factResults, serpResults, fusion);
      await sendMessage(chatId, resultMsg);
    }

    await sendMessage(chatId, 'Ingin mengecek teks lain? Kirimkan teks berikutnya kapan saja.');

  } catch (err) {
    console.error('[error]', err.message);
    try {
      await sendMessage(chatId,
        '❌ Terjadi kesalahan saat memproses. Pastikan backend sudah berjalan.\n\nKirim teks lagi untuk mencoba ulang.'
      );
    } catch (_) { /* ignore send error */ }
  }
});

app.listen(PORT, () => {
  console.log(`Hoax Checker WAHA bot running on port ${PORT}`);
  console.log(`Webhook URL: http://localhost:${PORT}/webhook`);
});
