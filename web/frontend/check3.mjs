import puppeteer from 'puppeteer';
import fs from 'fs';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  await page.setViewport({ width: 1400, height: 900 });
  
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', err => console.log('ERR:', err.toString()));
  
  await page.goto('http://localhost:5173');
  await new Promise(r => setTimeout(r, 2000));
  
  // Click the 3rd lesson (Sparse MoE Router)
  console.log("Clicking Lesson 3...");
  const lessons = await page.$$('.lesson-item');
  await lessons[2].click();
  
  await new Promise(r => setTimeout(r, 1000));
  
  // Click run
  console.log("Clicking Run...");
  await page.click('.run-btn');
  
  await new Promise(r => setTimeout(r, 3000));
  
  // Save screenshot
  await page.screenshot({ path: '/tmp/screenshot_sidebar.png' });
  console.log("Screenshot saved to /tmp/screenshot_sidebar.png");
  
  await browser.close();
})();
