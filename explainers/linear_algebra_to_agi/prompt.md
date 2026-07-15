# Prompt for Claude Code: "From Matrices to Minds" — An Interactive Explainer

## High-Level Goal

Build a minimal, visually distinctive, single-scroll (or lightly paginated) interactive
web experience that explains the evolutionary path:

**Linear Algebra & Probability → Machine Learning → Deep Learning → Neural Networks →
Transformers → GPTs → Agentic Systems → (speculative) AGI**

This is not a slide deck and not a Wikipedia page. It's a *guided intuition-builder*:
someone with high-school math should leave understanding *why* each step was necessary,
*what broke* in the previous paradigm, and *what one clean idea* unlocked the next stage.
Someone with an ML background should still learn something from the historical framing,
references, and the way the running example evolves.

## Core Design Constraints

- **Minimal, unique visual identity.** Not a generic Bootstrap/Tailwind SaaS look. Pick a
  restrained palette (2-3 colors + a dark or light neutral base), one distinctive
  typeface pairing (a serif or grotesk for headers, clean mono for math/code), and stick
  to it. Favor whitespace and typographic hierarchy over decoration. Think: Stripe's old
  patient capital essays, Bret Victor's explorable explanations, or Pudding.cool essays —
  not a startup landing page.
- **Smooth, intuitive animation** tied to scroll position or explicit "next" interactions
  — not decorative, but *explanatory*: animations should show a matrix multiplying, a
  gradient descending, a perceptron combining into a network, attention weights lighting
  up between tokens, etc. Prefer CSS transitions + a scroll-driven or step-driven state
  machine over heavy animation libraries. Respect `prefers-reduced-motion`.
- **One running example, followed end-to-end.** Pick something dead simple and concrete —
  e.g. predicting whether a short sentence expresses positive/negative sentiment, or
  predicting the next word after "The cat sat on the ___". Use this *same* toy example at
  every stage of the story so the reader feels the scale-up:
  1. Linear algebra/probability stage: represent 2-3 words as vectors, show a dot
     product, show a simple probability table (Naive Bayes-style).
  2. Classic ML stage: same example as logistic regression — draw the decision boundary.
  3. Neural network stage: same example through a 2-layer perceptron.
  4. Deep learning stage: stack more layers, mention vanishing gradients, ReLU, backprop.
  5. Transformer stage: same tiny sentence, show self-attention weights between its words.
  6. GPT stage: same sentence, now predicting the next token via a decoder-only model,
     show softmax over a tiny vocabulary.
  7. Agentic stage: the same underlying model now used in a loop with tools/memory to
     accomplish a small task (e.g. "look up the weather and text me if it's raining").
  8. AGI speculation: zoom out — what's missing, what current research directions claim
     to address it (world models, continual learning, reasoning/search at inference time,
     multi-agent self-improvement), framed as open questions, not certainties.
- **Key formulae included, but broken down**, not just dropped in. For each formula:
  a plain-English one-liner, the formula itself (rendered with KaTeX), and a labeled
  breakdown of each symbol. Suggested minimal formula set (don't overload — pick the
  smallest set that tells the story):
  - Dot product / cosine similarity: `a · b = Σ aᵢbᵢ`
  - Bayes' theorem: `P(A|B) = P(B|A)P(A)/P(B)`
  - Linear regression: `ŷ = Wx + b`
  - Logistic/sigmoid: `σ(z) = 1/(1+e^-z)`
  - Loss function (cross-entropy): `L = -Σ yᵢlog(ŷᵢ)`
  - Gradient descent update: `θ := θ - α∇L(θ)`
  - Perceptron/neuron: `y = f(Σ wᵢxᵢ + b)`
  - Backpropagation (conceptual, chain rule): `∂L/∂w = ∂L/∂y · ∂y/∂w`
  - Softmax: `softmax(z)ᵢ = e^zᵢ / Σⱼe^zⱼ`
  - Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`
  - (Optional, if scope allows) next-token prediction objective:
    `P(x_t | x_<t)` autoregressive factorization.
- **Historical milestones and references woven into the narrative**, not a separate
  timeline dump. As each math/architecture idea is introduced, attach it to who/when/why.
  Suggested milestones to include (verify dates/names before publishing — do not assume
  training data is authoritative on exact dates without a quick check):
  - 1763/1812 — Bayes' theorem (Thomas Bayes, published posthumously; Laplace's
    generalization)
  - Early 1800s — Least squares (Gauss/Legendre)
  - 1943 — McCulloch & Pitts artificial neuron
  - 1950 — Turing's "Computing Machinery and Intelligence" / Turing Test
  - 1956 — Dartmouth Workshop, term "Artificial Intelligence" coined
  - 1957 — Rosenblatt's Perceptron
  - 1969 — Minsky & Papert's "Perceptrons" critique, first AI winter
  - 1986 — Rumelhart, Hinton, Williams — backpropagation popularized
  - 1989 — LeCun — convolutional networks for digit recognition
  - 1997 — Hochreiter & Schmidhuber — LSTM
  - 1997 — Deep Blue beats Kasparov
  - 2006 — Hinton — deep belief nets, "deep learning" re-branding
  - 2012 — AlexNet / ImageNet moment (Krizhevsky, Sutskever, Hinton) — GPU-driven deep
    learning takeoff
  - 2013 — Word2Vec (Mikolov et al.) — word embeddings
  - 2014 — GANs (Goodfellow), Seq2Seq (Sutskever et al.), Adam optimizer
  - 2015 — ResNet (He et al.) — very deep networks via residual connections
  - 2017 — "Attention Is All You Need" (Vaswani et al.) — the Transformer
  - 2018 — BERT (Google), GPT-1 (OpenAI)
  - 2019 — GPT-2
  - 2020 — GPT-3, scaling laws (Kaplan et al.)
  - 2021-2022 — Codex, DALL·E, Chinchilla scaling laws, InstructGPT/RLHF
  - Nov 2022 — ChatGPT public launch — mainstream inflection point
  - 2023 — GPT-4, open-weight models (LLaMA), multimodal models
  - 2023-2024 — Tool use, function calling, RAG, early agent frameworks
  - 2024-2025 — Reasoning/inference-time compute models, agentic coding tools, multi-agent
    systems
  - Present — frontier labs pursuing longer-horizon autonomous agents, world models,
    continual learning as candidate paths toward AGI
  Use Claude's web search (in the Claude Code environment, or ask the user) to verify any
  date/name before finalizing copy — do not silently guess.
- **A clear, honest "path to AGI" closing section**, framed as informed speculation, not
  prophecy: present 2-3 competing hypotheses (scale is enough / scale plus new
  architecture is needed / symbolic-neural hybrid needed / embodiment is needed), name
  the researchers or labs associated with each view where fair to do so, and end with
  open questions rather than a confident timeline.

## Structure / Information Architecture

Suggest a single-page scroll experience broken into clearly delineated acts, each with
its own color accent or background shift so the reader always knows which era they're
in. Include a persistent minimal progress indicator (e.g. a thin vertical timeline dot
that moves) rather than a traditional nav bar. Suggested acts:

0. **Cold open** — the running example stated in one sentence, and the promise: "by the
   end, you'll see this same sentence handled by 70 years of math."
1. **Foundations** — linear algebra + probability primitives, via the toy example.
2. **Classical ML** — regression/classification, loss, gradient descent.
3. **Neural Networks** — perceptron → multilayer → backprop → why "deep."
4. **Deep Learning at Scale** — CNNs/RNNs, ImageNet moment, GPU compute, data scale.
5. **The Transformer** — attention mechanism, why RNNs hit a wall, parallelization.
6. **GPTs** — pretraining, scaling laws, RLHF/instruction tuning, emergent behavior.
7. **Agentic Systems** — tool use, memory, planning loops, multi-agent orchestration.
8. **Toward AGI?** — open hypotheses, honest uncertainty, closing reflection.

Each act should follow a consistent internal rhythm: **intuition → formula → toy example
visualization → milestone callouts → transition hook to next act.**

## Technical Implementation Notes for Claude Code

- Plain HTML/CSS/JS (or a minimal framework if truly helpful) — avoid heavy dependencies;
  this should load fast and feel crafted, not templated.
- Use KaTeX (via CDN) for formula rendering.
- Use SVG + CSS transitions for diagrams (vectors, decision boundaries, network graphs,
  attention heatmaps) — animate with the Web Animations API or IntersectionObserver-
  triggered classes for scroll-linked reveals. Keep everything under 60fps-friendly,
  GPU-accelerated transforms (translate/opacity/scale, avoid layout thrashing).
- Fully responsive; test narrow mobile widths where the multi-column diagrams will need
  to stack.
- Accessibility: semantic headings, alt text/aria-labels on SVG diagrams, honor
  `prefers-reduced-motion` by disabling non-essential motion, sufficient color contrast.
- Milestone references should appear as unobtrusive marginalia or a toggleable
  "sources & dates" panel per act, not inline citations that break flow — cite original
  papers/sources (arXiv links, original publication) where possible.
- Keep total scope buildable: aim for a coherent, polished 8-act single page rather than
  an overstuffed 20-section site. Depth of clarity over breadth of coverage.

## Tone

Confident, precise, a little bit in awe of the material — but never hype-y or vague.
Every claim should be the kind of thing a careful ML researcher would nod at. Avoid
marketing language like "revolutionary" or "game-changing"; let the ideas and the
30x-scale-jumps speak for themselves.