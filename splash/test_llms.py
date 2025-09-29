from __future__ import annotations
import os
import math
import numpy as np
import pytest

# Optional: make CPU-only runs deterministic and a bit faster
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Soft dependencies (skip if not available) --------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


# --- Splash imports -----------------------------------------------------------
from .types import EvalKnobs, Measures
from .geometry import one_tick_measures
from .scg.invariants import energy_flux

def _load_gpt2_medium():
    """Load gpt2-medium with hidden states on (skip tests if unavailable)."""
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        pytest.skip("transformers/torch not available")
    try:
        tok = AutoTokenizer.from_pretrained("gpt2-medium")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tok, model, device
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not load gpt2-medium: {e}")



def _layer_energies(layer_arrays):
    """Simple per-layer energy: sum of squared token norms (A^2)."""
    energies = []
    for E in layer_arrays:
        A = np.linalg.norm(E, axis=1)  # (T,)
        energies.append(float(np.sum(A * A)))
    return energies


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

import os


# keep tokenizer quiet on CI
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- soft deps: skip if not available ----------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


# ---------- helpers -----------------------------------------------------------
def _hidden_states(tok, model, device, text: str, max_tokens: int = 64):
    """Return list of per-layer hidden states as numpy arrays (T,D), skipping the embedding layer."""
    with torch.no_grad():
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        if max_tokens:
            input_ids = input_ids[:, :max_tokens]
        out = model(input_ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple: [embeddings, layer1, layer2, ...]
        return [h.squeeze(0).detach().cpu().numpy() for h in hs[1:]]  # list[(T,D)]


# ---------- session-scoped model/tokenizer -----------------------------------
@pytest.fixture(scope="session")
def gpt2_handles():
    """Load gpt2-medium once for all tests; skip if unavailable."""
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        pytest.skip("transformers/torch not available")

    try:
        tok = AutoTokenizer.from_pretrained("gpt2-medium")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not load gpt2-medium: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return tok, model, device


# ---------- requested fixtures -----------------------------------------------
@pytest.fixture
def gpt2_layers_mid(gpt2_handles):
    """A single mid-stack layer matrix E=(T,D) for a stable reference prompt."""
    tok, model, device = gpt2_handles
    text = "The 2012 Nobel Prize in Literature was awarded to Mo Yan."
    layers = _hidden_states(tok, model, device, text, max_tokens=48)
    L = len(layers)
    return layers[min(8, L - 1)]  # pick a middle layer


@pytest.fixture
def gpt2_layers_prompt_A(gpt2_handles):
    """List of early/mid hidden-state layers for prompt A."""
    tok, model, device = gpt2_handles
    textA = "The 2012 Nobel Prize in Literature was awarded to Mo Yan."
    layers = _hidden_states(tok, model, device, textA, max_tokens=64)
    return layers  # list[(T,D)]


@pytest.fixture
def gpt2_layers_prompt_B(gpt2_handles):
    """List of early/mid hidden-state layers for prompt B (different semantics)."""
    tok, model, device = gpt2_handles
    textB = "In 1984, a small personal computer changed how people write code."
    layers = _hidden_states(tok, model, device, textB, max_tokens=64)
    return layers  # list[(T,D)]


@pytest.fixture
def gpt2_layers_early(gpt2_handles):
    """
    Early hidden-state layers for a stable reference prompt (used for φ_E tests).
    Returns the first N layers as a list of (T, D) numpy arrays.
    """
    tok, model, device = gpt2_handles
    text = "The 2012 Nobel Prize in Literature was awarded to Mo Yan."
    layers = _hidden_states(tok, model, device, text, max_tokens=64)
    N = min(12, len(layers))   # keep runtime reasonable
    return layers[:N]


def test_smoke_measures_not_trivial(gpt2_layers_mid):  # use your fixture or inline loader
    E = gpt2_layers_mid
    ek = EvalKnobs(Ns=(8,), k_neighbors=8, settle_steps=16)
    m: Measures = one_tick_measures(E, eval_knobs=ek)
    assert 0.0 <= m.alignment_score <= 1.0
    assert 0.0 <= m.alignment_xy    <= 1.0
    assert 0.0 <= m.entropy         <= 1.0      # clamp guards tiny >1 overshoot
    assert math.isfinite(m.kappa_var) and math.isfinite(m.rho_phi)


def test_alignment_increases_with_settle(gpt2_layers_mid):
    E = gpt2_layers_mid
    m1 = one_tick_measures(E, eval_knobs=EvalKnobs(settle_steps=1,  k_neighbors=8))
    m2 = one_tick_measures(E, eval_knobs=EvalKnobs(settle_steps=32, k_neighbors=8))

    # scale the expected improvement by headroom to 1.0
    headroom = max(0.0, 1.0 - m1.alignment_score)
    expected_delta = max(0.002, 0.2 * headroom)   # at least +0.002, more if headroom exists
    assert m2.alignment_score >= m1.alignment_score + expected_delta
    assert m2.alignment_xy    >= m1.alignment_xy    + max(0.002, 0.2 * max(0.0, 1.0 - m1.alignment_xy))


def test_energy_flux_varies_across_layers(gpt2_layers_early):
    layers = gpt2_layers_early
    energies = _layer_energies(layers)          # your helper: mean/sum of A^2 per layer
    phi = energy_flux(energies, beta=0.5)       # bounded by tanh
    assert np.any(np.abs(phi) > 1e-8), f"φ_E all zeros: {phi}"
    assert np.max(np.abs(phi)) < 0.9            # compressed reasonable bound


def test_signatures_differ_across_prompts(gpt2_layers_prompt_A, gpt2_layers_prompt_B):
    layersA = gpt2_layers_prompt_A
    layersB = gpt2_layers_prompt_B

    def sig(layers):
        R = []; K = []; Rxy = []; Rho = []
        for E in layers[:8]:
            m = one_tick_measures(E, eval_knobs=EvalKnobs(settle_steps=8, k_neighbors=8))
            R.append(m.alignment_score); Rxy.append(m.alignment_xy); K.append(m.kappa_var); Rho.append(m.rho_phi)
        return np.array([np.mean(R), np.mean(Rxy), np.mean(K), np.mean(Rho)])

    a = sig(layersA)
    b = sig(layersB)

    # Require the overall signature vectors to differ by > 1e-5 in L2 (very small but robust)
    assert not np.allclose(a, b, rtol=0, atol=1e-5), (a, b)

def test_curvature_variance_nonzero_for_real_layers():
    tok, model, device = _load_gpt2_medium()
    text = "The 2012 Nobel Prize in Literature was awarded to Mo Yan."
    layers = _hidden_states(tok, model, device, text, max_tokens=48)

    kv = []
    for E in layers[:12]:  # first dozen layers
        m = one_tick_measures(E, eval_knobs=EvalKnobs(settle_steps=8, k_neighbors=8))
        kv.append(m.kappa_var)
    kv = np.asarray(kv)

    # Not all zero and shows variation across depth
    assert np.any(kv > 1e-6), f"kappa_var all ~0: {kv}"
    assert np.std(kv) > 1e-6, f"kappa_var flat: {kv}"


def test_rho_phi_is_reasonable_on_clean_prompt():
    tok, model, device = _load_gpt2_medium()
    text = "A short clean sentence that should not produce large phase jumps."
    layers = _hidden_states(tok, model, device, text, max_tokens=40)
    E = layers[min(6, len(layers) - 1)]

    m = one_tick_measures(E, eval_knobs=EvalKnobs(settle_steps=8, k_neighbors=6))
    # On clean text we expect low dislocation density (but > 0 is fine)
    assert 0.0 <= m.rho_phi < 0.5
