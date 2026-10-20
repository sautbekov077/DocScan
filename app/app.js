const DB_NAME = 'DocScannerDB';
const STORE_NAME = 'pages';
let dbInstance = null;

function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = (e) => {
      dbInstance = e.target.result;
      resolve(dbInstance);
    };
    request.onerror = (e) => reject(e.target.error);
  });
}

function savePageToDB(imageDataUrl) {
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.add({ dataUrl: imageDataUrl, createdAt: Date.now() });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function getAllPagesFromDB() {
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function deletePageFromDB(id) {
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}


let isOpenCvReady = false;

window.onOpenCvReady = function () {
  cv['onRuntimeInitialized'] = () => {
    isOpenCvReady = true;
    hideLoader();
  };
};

function processScannedImage(imageElement) {
  
  let src = cv.imread(imageElement);
  let processed = new cv.Mat();

  
  let maxDim = 1000;
  let scale = Math.min(1.0, maxDim / Math.max(src.rows, src.cols));
  let dsize = new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale));
  cv.resize(src, processed, dsize, 0, 0, cv.INTER_AREA);

 
  let gray = new cv.Mat();
  cv.cvtColor(processed, gray, cv.COLOR_RGBA2GRAY);
  let blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);


  let edges = new cv.Mat();
  cv.Canny(blurred, edges, 75, 200);
  let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.dilate(edges, edges, kernel);

 
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  let maxArea = 0;
  let docContour = null;

  for (let i = 0; i < contours.size(); ++i) {
    let cnt = contours.get(i);
    let area = cv.contourArea(cnt);
    if (area > 5000) {
      let peri = cv.arcLength(cnt, true);
      let approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
      
      if (approx.rows === 4 && area > maxArea) {
        maxArea = area;
        if (docContour) docContour.delete();
        docContour = approx;
      } else {
        approx.delete();
      }
    }
    cnt.delete();
  }

  
  let warped = new cv.Mat();
  if (docContour) {
  
    let pts = [];
    for (let i = 0; i < 4; i++) {
      pts.push({
        x: docContour.data32S[i * 2] / scale,
        y: docContour.data32S[i * 2 + 1] / scale
      });
    }
    pts = orderPoints(pts);

    
    let widthA = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    let widthB = Math.hypot(pts[2].x - pts[3].x, pts[2].y - pts[3].y);
    let targetWidth = Math.max(widthA, widthB);

    let heightA = Math.hypot(pts[3].x - pts[0].x, pts[3].y - pts[0].y);
    let heightB = Math.hypot(pts[2].x - pts[1].x, pts[2].y - pts[1].y);
    let targetHeight = Math.max(heightA, heightB);

    let srcCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
      pts[0].x, pts[0].y,
      pts[1].x, pts[1].y,
      pts[2].x, pts[2].y,
      pts[3].x, pts[3].y
    ]);
    let dstCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      targetWidth, 0,
      targetWidth, targetHeight,
      0, targetHeight
    ]);

    let M = cv.getPerspectiveTransform(srcCoords, dstCoords);
    let finalSize = new cv.Size(targetWidth, targetHeight);
    cv.warpPerspective(src, warped, M, finalSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

    srcCoords.delete(); dstCoords.delete(); M.delete(); docContour.delete();
  } else {
    
    warped = src.clone();
  }

  
  let warpedGray = new cv.Mat();
  cv.cvtColor(warped, warpedGray, cv.COLOR_RGBA2GRAY);
  let enhanced = new cv.Mat();
  
 
  cv.adaptiveThreshold(
    warpedGray, 
    enhanced, 
    255, 
    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv.THRESH_BINARY, 
    21, 
    11
  );

  
  let canvas = document.getElementById('hidden-canvas');
  cv.imshow(canvas, enhanced);
  let finalDataUrl = canvas.toDataURL('image/jpeg', 0.92);

 
  src.delete(); processed.delete(); gray.delete(); blurred.delete();
  edges.delete(); kernel.delete(); contours.delete(); hierarchy.delete();
  warped.delete(); warpedGray.delete(); enhanced.delete();

  return finalDataUrl;
}


function orderPoints(pts) {
  pts.sort((a, b) => (a.x + a.y) - (b.x + b.y));
  let tl = pts[0];
  let br = pts[3];

  let remaining = [pts[1], pts[2]];
  remaining.sort((a, b) => (a.y - a.x) - (b.y - b.x));
  let tr = remaining[0];
  let bl = remaining[1];

  return [tl, tr, br, bl];
}


const cameraInput = document.getElementById('camera-input');
const galleryInput = document.getElementById('gallery-input');
const pagesGrid = document.getElementById('pages-grid');
const emptyState = document.getElementById('empty-state');
const pageCountBadge = document.getElementById('page-count');
const exportPdfBtn = document.getElementById('export-pdf-btn');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');

function showLoader(text = 'Обработка...') {
  loaderText.innerText = text;
  loader.classList.remove('hidden');
}

function hideLoader() {
  loader.classList.add('hidden');
}

async function handleFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;

  showLoader('Выравнивание и фильтрация...');
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = async () => {
      try {
        const processedDataUrl = processScannedImage(img);
        await savePageToDB(processedDataUrl);
        await renderPages();
      } catch (err) {
        console.error('Ошибка обработки:', err);
        alert('Не удалось обработать документ.');
      } finally {
        hideLoader();
        event.target.value = '';
      }
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

cameraInput.addEventListener('change', handleFileSelected);
galleryInput.addEventListener('change', handleFileSelected);

async function renderPages() {
  const pages = await getAllPagesFromDB();
  pagesGrid.innerHTML = '';

  if (pages.length === 0) {
    emptyState.classList.remove('hidden');
    pagesGrid.classList.add('hidden');
    exportPdfBtn.disabled = true;
    pageCountBadge.innerText = `0 стр.`;
    return;
  }

  emptyState.classList.add('hidden');
  pagesGrid.classList.remove('hidden');
  exportPdfBtn.disabled = false;
  pageCountBadge.innerText = `${pages.length} стр.`;

  pages.forEach((page, index) => {
    const card = document.createElement('div');
    card.className = 'page-card';
    card.innerHTML = `
      <img src="${page.dataUrl}" alt="Стр. ${index + 1}">
      <div class="page-footer">
        <span class="page-num">Стр. ${index + 1}</span>
        <button class="delete-btn" onclick="removePage(${page.id})">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
        </button>
      </div>
    `;
    pagesGrid.appendChild(card);
  });
}

window.removePage = async function(id) {
  if (confirm('Удалить эту страницу?')) {
    await deletePageFromDB(id);
    await renderPages();
  }
};

exportPdfBtn.addEventListener('click', async () => {
  const pages = await getAllPagesFromDB();
  if (pages.length === 0) return;

  showLoader('Формирование PDF...');
  
  try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    for (let i = 0; i < pages.length; i++) {
      if (i > 0) doc.addPage();
      doc.addImage(pages[i].dataUrl, 'JPEG', 0, 0, pageWidth, pageHeight, undefined, 'FAST');
    }

    doc.save(`Scan_${new Date().toISOString().slice(0,10)}.pdf`);
  } catch (error) {
    console.error(error);
    alert('Ошибка генерации PDF.');
  } finally {
    hideLoader();
  }
});


if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js')
      .then(reg => console.log('Service Worker зарегистрирован:', reg.scope))
      .catch(err => console.error('Ошибка регистрации SW:', err));
  });
}


initDB().then(() => renderPages());
