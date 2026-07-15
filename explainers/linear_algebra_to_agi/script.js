(() => {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- KaTeX ---------------- */
  document.addEventListener('DOMContentLoaded', () => {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [{ left: '$$', right: '$$', display: true }],
        throwOnError: false
      });
    }
    document.querySelectorAll('.tex').forEach((el) => {
      if (window.katex) {
        try {
          katex.render(el.textContent.trim(), el, { displayMode: el.dataset.display === 'true', throwOnError: false });
        } catch (e) { /* leave raw text as fallback */ }
      }
    });
  });

  /* ---------------- Progress rail + hue + reveal ---------------- */

  const acts = Array.from(document.querySelectorAll('.act'));
  const dots = Array.from(document.querySelectorAll('.progress-rail .dot'));
  const root = document.documentElement;

  const setActive = (actIndex, hue) => {
    dots.forEach((d) => d.classList.toggle('is-active', d.dataset.act === String(actIndex)));
    if (hue !== undefined) root.style.setProperty('--accent-h', hue);
  };

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const inner = entry.target.querySelector('.act-inner');
        if (inner) inner.classList.add('is-revealed');
        initDiagramsFor(entry.target);
      }
    });
  }, { threshold: 0.18 });

  const activeObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        setActive(entry.target.dataset.act, entry.target.dataset.hue);
      }
    });
  }, { threshold: 0, rootMargin: '-45% 0px -45% 0px' });

  acts.forEach((act) => {
    revealObserver.observe(act);
    activeObserver.observe(act);
  });

  /* ---------------- Diagram init (runs once per act, on first reveal) ---------------- */

  const initialized = new WeakSet();

  function initDiagramsFor(act) {
    if (initialized.has(act)) return;
    initialized.add(act);
    const diagram = act.querySelector('[data-diagram]');
    switch (act.dataset.act) {
      case '1':
        animateBayesBars(act);
        break;
      case '2':
        animateDecisionBoundary(act);
        animateGDBall(act);
        break;
      case '3':
        runForwardPass(act);
        break;
      case '4':
        buildGradientRows(act);
        break;
      case '5':
        animateAttentionArcs(act);
        break;
      case '6':
        runSoftmax(act);
        break;
      case '7':
        runAgentLoop(act);
        break;
      case '8':
        animateBranchPaths(act);
        break;
      default:
        break;
    }
  }

  /* ---------------- Act 1: dot product mini-compute + Bayes bars ---------------- */

  const TOY_VECTORS = {
    cat: [0.20, 0.85],
    mat: [0.15, 0.60],
    grass: [0.80, 0.30]
  };

  document.querySelectorAll('[data-compute="dot-product"] .compute-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const [a, b] = btn.dataset.pair.split('-');
      const va = TOY_VECTORS[a], vb = TOY_VECTORS[b];
      const dot = (va[0] * vb[0] + va[1] * vb[1]).toFixed(3);
      const out = btn.closest('.mini-compute').querySelector('.compute-result');
      out.textContent = `${a} · ${b} = (${va[0]}×${vb[0]}) + (${va[1]}×${vb[1]}) = ${dot}`;
    });
  });

  function animateBayesBars(act) {
    const bars = act.querySelectorAll('.bayes-bar');
    bars.forEach((bar, i) => {
      const target = Number(bar.dataset.target);
      const fill = bar.querySelector('.bar-fill');
      const delay = reduceMotion ? 0 : i * 250;
      setTimeout(() => { fill.setAttribute('width', target); }, delay);
    });
  }

  /* ---------------- Act 2: decision boundary + gradient descent ball ---------------- */

  function animateDecisionBoundary(act) {
    const line = act.querySelector('.decision-line');
    if (!line) return;
    requestAnimationFrame(() => { line.style.strokeDashoffset = '0'; });
  }

  function animateGDBall(act) {
    const ball = act.querySelector('.gd-ball');
    if (!ball) return;
    requestAnimationFrame(() => { ball.style.offsetDistance = '100%'; });
  }

  /* ---------------- Act 3: forward pass / backprop network ---------------- */

  function networkSequence(act, reverse) {
    const nodes = Array.from(act.querySelectorAll('.node'));
    const edges = Array.from(act.querySelectorAll('.edges line'));
    nodes.forEach((n) => n.classList.remove('is-lit'));
    edges.forEach((e) => e.classList.remove('is-active'));

    const layerOrder = reverse
      ? [['output'], ['hidden'], ['input']]
      : [['input'], ['hidden'], ['output']];

    const step = (layerClasses, delay) => {
      setTimeout(() => {
        layerClasses.forEach((cls) => {
          nodes.filter((n) => n.classList.contains(`node--${cls}`)).forEach((n) => n.classList.add('is-lit'));
        });
        edges.forEach((edge) => {
          const [x1] = [Number(edge.getAttribute('x1'))];
          // light edges leaving the layer we just lit (rough heuristic by x position)
        });
      }, delay);
    };

    if (reduceMotion) {
      nodes.forEach((n) => n.classList.add('is-lit'));
      edges.forEach((e) => e.classList.add('is-active'));
      return;
    }

    const stepDelay = 550;
    layerOrder.forEach((layer, i) => step(layer, i * stepDelay));

    // edges: input<->hidden group, hidden<->output group
    const inputHiddenEdges = edges.slice(0, 6);
    const hiddenOutputEdges = edges.slice(6);
    if (reverse) {
      setTimeout(() => hiddenOutputEdges.forEach((e) => e.classList.add('is-active')), stepDelay * 0.6);
      setTimeout(() => inputHiddenEdges.forEach((e) => e.classList.add('is-active')), stepDelay * 1.6);
    } else {
      setTimeout(() => inputHiddenEdges.forEach((e) => e.classList.add('is-active')), stepDelay * 0.6);
      setTimeout(() => hiddenOutputEdges.forEach((e) => e.classList.add('is-active')), stepDelay * 1.6);
    }
  }

  function runForwardPass(act) {
    networkSequence(act, false);
    act.querySelectorAll('[data-replay="network"]').forEach((btn) => {
      btn.addEventListener('click', () => networkSequence(act, false));
    });
    act.querySelectorAll('[data-replay="backprop"]').forEach((btn) => {
      btn.addEventListener('click', () => networkSequence(act, true));
    });
  }

  /* ---------------- Act 4: vanishing gradient rows ---------------- */

  function buildGradientRows(act) {
    const rows = act.querySelectorAll('.grad-row');
    const n = 8;
    rows.forEach((row) => {
      const isSigmoid = row.classList.contains('grad-row--sigmoid');
      const g = row.querySelector('.grad-layers');
      const y = isSigmoid ? 12 : 82;
      const size = 26;
      const startX = 70, gap = 48;
      const cells = [];
      for (let i = 0; i < n; i++) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', startX + i * gap);
        rect.setAttribute('y', y);
        rect.setAttribute('width', size);
        rect.setAttribute('height', size);
        rect.setAttribute('rx', 4);
        rect.setAttribute('fill', isSigmoid ? 'var(--rust)' : 'var(--accent)');
        rect.classList.add('grad-cell');
        const t = i / (n - 1);
        const targetOpacity = isSigmoid ? 0.06 + 0.85 * Math.pow(t, 2.4) : 0.5 + 0.4 * t;
        rect.style.opacity = reduceMotion ? targetOpacity : 0;
        g.appendChild(rect);
        cells.push({ el: rect, targetOpacity, i });
      }
      if (!reduceMotion) {
        cells.forEach(({ el, targetOpacity, i }) => {
          const delay = (n - 1 - i) * 90;
          setTimeout(() => { el.style.transition = 'opacity 400ms ease'; el.style.opacity = targetOpacity; }, delay);
        });
      }
    });
  }

  /* ---------------- Act 5: attention arcs ---------------- */

  function animateAttentionArcs(act) {
    const arcs = Array.from(act.querySelectorAll('.attn-arc'));
    arcs.forEach((arc, i) => {
      const w = Number(arc.dataset.weight);
      arc.style.strokeWidth = String(1 + w * 12);
      const delay = reduceMotion ? 0 : i * 180;
      setTimeout(() => arc.classList.add('is-visible'), delay);
    });
  }

  /* ---------------- Act 6: softmax bars ---------------- */

  function runSoftmax(act) {
    const run = () => {
      const bars = act.querySelectorAll('.sm-bar');
      bars.forEach((bar, i) => {
        const target = Number(bar.dataset.target);
        const fill = bar.querySelector('.bar-fill');
        const valueLabel = bar.querySelector('.sm-value');
        const trackY = 190, trackTop = 10, maxH = trackY - trackTop;
        const h = target * maxH * 1.8; // exaggerate for readability
        const cappedH = Math.min(h, maxH);
        const delay = reduceMotion ? 0 : i * 150;
        setTimeout(() => {
          fill.setAttribute('y', trackY - cappedH);
          fill.setAttribute('height', cappedH);
          const x = Number(fill.getAttribute('x')) + Number(fill.getAttribute('width')) / 2;
          valueLabel.setAttribute('x', x);
          valueLabel.setAttribute('y', trackY - cappedH - 8);
          valueLabel.classList.add('is-visible');
        }, delay);
      });
    };
    run();
    act.querySelectorAll('[data-replay="softmax"]').forEach((btn) => btn.addEventListener('click', run));
  }

  /* ---------------- Act 7: agent loop ---------------- */

  function runAgentLoop(act) {
    const run = () => {
      const nodes = Array.from(act.querySelectorAll('.loop-node'));
      nodes.forEach((n) => n.classList.remove('is-active'));
      nodes.forEach((n, i) => {
        const delay = reduceMotion ? 0 : i * 550;
        setTimeout(() => {
          nodes.forEach((m) => m.classList.remove('is-active'));
          n.classList.add('is-active');
        }, delay);
      });
    };
    run();
    act.querySelectorAll('[data-replay="agent-loop"]').forEach((btn) => btn.addEventListener('click', run));
  }

  /* ---------------- Act 8: branching paths ---------------- */

  function animateBranchPaths(act) {
    const lines = act.querySelectorAll('.branch-line');
    lines.forEach((line, i) => {
      const delay = reduceMotion ? 0 : i * 220;
      setTimeout(() => { line.style.strokeDashoffset = '0'; }, delay);
    });
  }

})();
