import puppeteer from 'puppeteer';
import fs from 'fs';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Set viewport large enough
  await page.setViewport({ width: 1200, height: 800 });
  
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', err => console.log('ERR:', err.toString()));
  
  await page.goto('http://localhost:5173');
  await new Promise(r => setTimeout(r, 2000));
  
  // Click run
  console.log("Clicking Run...");
  await page.click('.run-btn');
  
  await new Promise(r => setTimeout(r, 3000));
  
  // Save screenshot
  await page.screenshot({ path: '/tmp/screenshot.png' });
  
  // Save DOM
  const html = await page.content();
  fs.writeFileSync('/tmp/dom.html', html);
  
  console.log("Screenshot & DOM saved to /tmp");
  
  await browser.close();
})();
