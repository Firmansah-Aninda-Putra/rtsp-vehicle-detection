function adjustHeaderHeight() {
    const header = document.getElementById('header');
    const bgImg = new Image();
    bgImg.src = '/static/images/kota madiun.jpeg';
    bgImg.onload = function() {
      // Hitung aspect ratio (tinggi / lebar) gambar
      const ratio = bgImg.naturalHeight / bgImg.naturalWidth;
      // Dapatkan lebar container header
      const headerWidth = header.offsetWidth;
      // Set tinggi header berdasarkan aspect ratio
      header.style.height = (headerWidth * ratio) + 'px';
    };
  }
  
  window.addEventListener('load', adjustHeaderHeight);
  window.addEventListener('resize', adjustHeaderHeight);
  