/* ============================================================
   HepatoAI — Main JavaScript
   File: frontend/static/js/main.js
   ============================================================ */

// Simple page load animation
document.addEventListener('DOMContentLoaded', () => {
  // Animate hero content in
  const heroContent = document.querySelector('.hero-content');
  const heroVisual = document.querySelector('.hero-visual');

  if (heroContent) {
    heroContent.style.opacity = '0';
    heroContent.style.transform = 'translateY(20px)';
    setTimeout(() => {
      heroContent.style.transition = 'all 0.7s ease';
      heroContent.style.opacity = '1';
      heroContent.style.transform = 'translateY(0)';
    }, 100);
  }

  if (heroVisual) {
    heroVisual.style.opacity = '0';
    heroVisual.style.transform = 'translateY(20px)';
    setTimeout(() => {
      heroVisual.style.transition = 'all 0.7s ease';
      heroVisual.style.opacity = '1';
      heroVisual.style.transform = 'translateY(0)';
    }, 300);
  }

  // Animate stat cards
  const statCards = document.querySelectorAll('.stat-card, .feature-card');
  statCards.forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(15px)';
    setTimeout(() => {
      card.style.transition = 'all 0.5s ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 400 + i * 80);
  });
});
