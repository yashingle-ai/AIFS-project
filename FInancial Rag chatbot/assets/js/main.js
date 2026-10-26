/* ═══════════════════════════════════════════════════════════
   QuantX — interaction layer
   Zero dependencies. Everything degrades without JS.
   ═══════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Nav: opacity on scroll + mobile menu ───────────────── */
  var nav = document.getElementById('nav');
  var navToggle = document.getElementById('navToggle');
  var navLinks = document.querySelector('.nav__links');

  navToggle.addEventListener('click', function () {
    var open = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!open));
    navLinks.classList.toggle('is-open', !open);
  });
  navLinks.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') {
      navToggle.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('is-open');
    }
  });

  /* ── Scroll progress ────────────────────────────────────── */
  var progress = document.getElementById('scrollProgress');

  function onScroll() {
    var y = window.scrollY || window.pageYOffset;
    nav.classList.toggle('is-stuck', y > 40);

    var max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (max > 0 ? (y / max) * 100 : 0) + '%';

    updateJourney();
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });

  /* ── Reveal on enter ────────────────────────────────────── */
  var revealables = document.querySelectorAll('.reveal, .jstep, .tier');

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-in');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    revealables.forEach(function (el) { io.observe(el); });
  } else {
    revealables.forEach(function (el) { el.classList.add('is-in'); });
  }

  /* ── Count-up numbers ───────────────────────────────────── */
  var counters = document.querySelectorAll('[data-count]');

  function countUp(el) {
    var target = parseInt(el.getAttribute('data-count'), 10);
    if (reduced) { el.textContent = target; return; }
    var start = performance.now();
    var dur = 1300;
    (function tick(now) {
      var p = Math.min((now - start) / dur, 1);
      // easeOutExpo
      var eased = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
      el.textContent = Math.round(target * eased);
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }

  if ('IntersectionObserver' in window) {
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) { countUp(entry.target); cio.unobserve(entry.target); }
      });
    }, { threshold: 0.6 });
    counters.forEach(function (el) { cio.observe(el); });
  } else {
    counters.forEach(function (el) { el.textContent = el.getAttribute('data-count'); });
  }

  /* ── Hero query trace animation ─────────────────────────── */
  var traceSteps = document.querySelectorAll('.trace__step');
  var traceAnswer = document.querySelector('.trace__answer');

  function runTrace() {
    if (reduced) {
      traceSteps.forEach(function (s) { s.classList.add('is-on'); });
      traceAnswer.classList.add('is-on');
      return;
    }
    traceSteps.forEach(function (s) { s.classList.remove('is-on'); });
    traceAnswer.classList.remove('is-on');

    traceSteps.forEach(function (step, i) {
      setTimeout(function () { step.classList.add('is-on'); }, 520 + i * 430);
    });
    setTimeout(function () {
      traceAnswer.classList.add('is-on');
    }, 520 + traceSteps.length * 430 + 220);
  }

  runTrace();
  if (!reduced) {
    setInterval(runTrace, 520 + traceSteps.length * 430 + 4200);
  }

  /* ── Journey rail fill ──────────────────────────────────── */
  var journey = document.getElementById('journey');
  var journeyFill = document.getElementById('journeyFill');

  function updateJourney() {
    if (!journey || !journeyFill) return;
    var rect = journey.getBoundingClientRect();
    var vh = window.innerHeight;
    var total = rect.height;
    // 0 when the section top hits 70% of viewport, 1 when its bottom passes 40%
    var travelled = (vh * 0.7) - rect.top;
    var pct = Math.max(0, Math.min(1, travelled / total));
    journeyFill.style.height = (pct * 100) + '%';
  }

  /* ── Screenshot tabs ────────────────────────────────────── */
  var tabs = document.querySelectorAll('.shots__tabs button');
  var shots = document.querySelectorAll('.shot');
  var frameUrl = document.getElementById('frameUrl');

  var urls = [
    'localhost:8000 — DocChat',
    'localhost:8000 — grounded answer',
    'localhost:3000 — RAG workspace',
    'localhost:3000 — landing',
    'localhost:3000 — the problem',
    'localhost:3000 — the solution'
  ];

  tabs.forEach(function (tab, i) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) {
        t.classList.remove('is-active');
        t.setAttribute('aria-selected', 'false');
      });
      shots.forEach(function (s) {
        s.classList.remove('is-active');
        s.hidden = true;
      });
      tab.classList.add('is-active');
      tab.setAttribute('aria-selected', 'true');
      shots[i].classList.add('is-active');
      shots[i].hidden = false;
      frameUrl.textContent = urls[i] || urls[0];
    });

    // arrow-key navigation across the tablist
    tab.addEventListener('keydown', function (e) {
      var dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      var next = (i + dir + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
    });
  });

  /* ── Lightbox ───────────────────────────────────────────── */
  var lightbox = document.getElementById('lightbox');
  var lightboxImg = document.getElementById('lightboxImg');
  var lightboxClose = document.getElementById('lightboxClose');
  var lastFocused = null;

  function openLightbox(src, alt) {
    lastFocused = document.activeElement;
    lightboxImg.src = src;
    lightboxImg.alt = alt || '';
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    lightboxClose.focus();
  }
  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImg.src = '';
    document.body.style.overflow = '';
    if (lastFocused) lastFocused.focus();
  }

  document.querySelectorAll('.shot, .zoomable').forEach(function (el) {
    el.addEventListener('click', function () {
      var src = el.getAttribute('data-full');
      var img = el.querySelector('img');
      if (src) openLightbox(src, img ? img.alt : '');
    });
  });

  lightboxClose.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', function (e) {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !lightbox.hidden) closeLightbox();
    // keep focus inside the dialog
    if (e.key === 'Tab' && !lightbox.hidden) {
      e.preventDefault();
      lightboxClose.focus();
    }
  });

  /* ── Cursor glow (pointer devices only) ─────────────────── */
  var glow = document.getElementById('cursorGlow');

  if (!reduced && window.matchMedia('(pointer:fine)').matches) {
    var gx = 0, gy = 0, cx = 0, cy = 0, raf = null;

    window.addEventListener('mousemove', function (e) {
      gx = e.clientX; gy = e.clientY;
      glow.style.opacity = '1';
      if (!raf) raf = requestAnimationFrame(follow);
    }, { passive: true });

    document.addEventListener('mouseleave', function () { glow.style.opacity = '0'; });

    function follow() {
      cx += (gx - cx) * 0.09;
      cy += (gy - cy) * 0.09;
      glow.style.transform = 'translate(' + cx + 'px,' + cy + 'px) translate(-50%,-50%)';
      raf = (Math.abs(gx - cx) > 0.5 || Math.abs(gy - cy) > 0.5)
        ? requestAnimationFrame(follow)
        : null;
    }
  }

  /* ── Magnetic buttons ───────────────────────────────────── */
  if (!reduced && window.matchMedia('(pointer:fine)').matches) {
    document.querySelectorAll('.btn').forEach(function (btn) {
      btn.addEventListener('mousemove', function (e) {
        var r = btn.getBoundingClientRect();
        var mx = e.clientX - r.left - r.width / 2;
        var my = e.clientY - r.top - r.height / 2;
        btn.style.transform = 'translate(' + mx * 0.16 + 'px,' + my * 0.24 + 'px)';
      });
      btn.addEventListener('mouseleave', function () { btn.style.transform = ''; });
    });
  }

  /* ── Active section in nav ──────────────────────────────── */
  var sections = document.querySelectorAll('section[id]');
  var navAnchors = document.querySelectorAll('.nav__links a');

  if ('IntersectionObserver' in window) {
    var sio = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var id = entry.target.getAttribute('id');
        navAnchors.forEach(function (a) {
          a.style.color = a.getAttribute('href') === '#' + id ? 'var(--acc)' : '';
        });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    sections.forEach(function (s) { sio.observe(s); });
  }

  onScroll();
})();
