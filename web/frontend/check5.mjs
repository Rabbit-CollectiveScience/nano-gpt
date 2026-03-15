import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  await page.setViewport({ width: 1400, height: 900 });
  
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', err => console.log('ERR:', err.toString()));
  
  await page.goto('http://localhost:5173');
  await new Promise(r => setTimeout(r, 2000));
  
  // Select mistaken model
  console.log("Selecting Mistral from dropdown...");
  await page.select('select', 'mistral');
  
  await new Promise(r => setTimeout(r, 1000));
  
  // Save screenshot
  await page.screenshot({ path: '/tmp/screenshot_dropdown.png' });
  console.log("Screenshot saved to /tmp/screenshot_dropdown.png");
  
  await browser.close();
})();
