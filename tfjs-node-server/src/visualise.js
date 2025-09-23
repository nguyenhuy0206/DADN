const { createCanvas, loadImage } = require('canvas');
const path = require('path');
const fs = require('fs');
const { OUTPUT_DIR, PATCH_SIZE } = require('./config');
const logger = require('./logger');

async function visualizeOutput(imagePath, importantTiles, epsilon, lambdaSet, scaledBBox) {
  try {
    const canvas = createCanvas(512, 512);
    const ctx = canvas.getContext('2d');

    const img = await loadImage(imagePath);
    ctx.drawImage(img, 0, 0, 512, 512);

    // draw GT box
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.strokeRect(scaledBBox.x, scaledBBox.y, scaledBBox.width, scaledBBox.height);
    ctx.font = '14px Arial';
    ctx.fillStyle = 'blue';
    ctx.fillText('GT Box', scaledBBox.x + 6, scaledBBox.y + 16);

    // draw top tiles (red translucent)
    const top = importantTiles.slice(0, 6);
    for (const t of top) {
      const { x, y } = t;
      const att = typeof t.attentionEMA !== 'undefined' ? t.attentionEMA : t.combinedScore || 0;
      ctx.fillStyle = 'rgba(255,0,0,0.3)';
      ctx.fillRect(x, y, PATCH_SIZE, PATCH_SIZE);
      ctx.fillStyle = 'black';
      ctx.fillText(`A=${att.toFixed(3)}`, x + 6, y + 18);
    }

    // ensure output dir exists
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    const base = path.basename(imagePath, path.extname(imagePath));
    const lambdaStr = lambdaSet ? lambdaSet.map(l => l.toFixed(2)).join('-') : 'na';
    const outPath = path.join(OUTPUT_DIR, `${base}_lam${lambdaStr}_eps${epsilon}.png`);
    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(outPath, buffer);
    logger.info('Visualization saved to', outPath);
    return outPath;
  } catch (err) {
    logger.error('visualizeOutput error:', err.message);
    throw err;
  }
}

module.exports = { visualizeOutput };
