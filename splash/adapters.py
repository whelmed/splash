from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List, Tuple
import numpy as np

from .types import SequenceBatch, ModelOutputs, HiddenStates, AttentionMaps

class ModelAdapter(Protocol):
    """
    Minimal surface needed by the evaluator.
    Implementations must be deterministic under a fixed RNG seed.
    """
    def tokenize(self, texts: List[str], *, max_length: Optional[int] = None) -> SequenceBatch: ...
    def forward(self, batch: SequenceBatch, *, capture_attention: bool = False) -> ModelOutputs: ...
    def get_config(self) -> Dict[str, Any]: ...

class HFAdapter:
    """
    HuggingFace transformer adapter.
    - supports capturing per-layer hidden states and optional attention
    - respects attention_mask and truncation
    """
    def __init__(self, model, tokenizer, *, device: Optional[str] = None, seed: int = 42):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = int(seed)
        try:
            import torch
            self._torch = torch
        except Exception as e:
            raise ImportError("HFAdapter requires PyTorch and transformers installed.") from e

        # put model in eval mode & device
        if device is not None:
            self.model.to(device)
        self.model.eval()

    def tokenize(self, texts: List[str], *, max_length: Optional[int] = None) -> SequenceBatch:
        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = toks["input_ids"].cpu().numpy()
        attn = toks.get("attention_mask", None)
        attention_mask = attn.cpu().numpy() if attn is not None else None
        return SequenceBatch(input_ids=input_ids, attention_mask=attention_mask, texts=texts)

    def forward(self, batch: SequenceBatch, *, capture_attention: bool = False) -> ModelOutputs:
        torch = self._torch
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            input_ids = torch.tensor(batch.input_ids)
            attention_mask = torch.tensor(batch.attention_mask) if batch.attention_mask is not None else None
            if self.device is not None:
                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=capture_attention,
                use_cache=False,
            )

            # If a DynamicManager is present, apply it to torch tensors before numpy conversion
            if self.has_dynamic_manager():
                with torch.no_grad():
                    hts = [h.detach() for h in out.hidden_states]   # list of (B,T,d) torch tensors
                    hts = self._dyn_mgr.apply_layers(hts)
                hs = [ht.detach().cpu().numpy() for ht in hts]
            else:
                hs = [h.detach().cpu().numpy() for h in out.hidden_states]

            final = hs[-1]
            layer_names = [f"layer_{i}" for i in range(len(hs))]
            hidden = HiddenStates(layers=hs, final=final, layer_names=layer_names)

            attn_maps = None
            if capture_attention and hasattr(out, "attentions") and out.attentions is not None:
                per_layer = [a.detach().cpu().numpy() for a in out.attentions]
                # averaged per head (B,T,T)
                averaged = [a.mean(axis=1) for a in per_layer] if per_layer[0].ndim == 4 else None
                attn_maps = AttentionMaps(per_layer=per_layer, averaged=averaged)

            # token embeddings (if accessible via model.get_input_embeddings())
            tok_emb = None
            if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
                emb_w = self.model.get_input_embeddings().weight.detach().cpu().numpy()
                tok_emb = emb_w

            # positional embeddings best-effort (varies across arch)
            pos_emb = None
            if hasattr(self.model, "model") and hasattr(self.model.model, "embed_positions"):
                pe = self.model.model.embed_positions.weight.detach().cpu().numpy()
                pos_emb = pe

            return ModelOutputs(
                batch=batch,
                hidden=hidden,
                attention=attn_maps,
                token_embeddings=tok_emb,
                pos_embeddings=pos_emb,
            )

    def get_config(self) -> Dict[str, Any]:
        cfg = {"adapter": "HFAdapter"}
        if hasattr(self.model, "config"):
            cfg["model"] = getattr(self.model.config, "name_or_path", type(self.model).__name__)
            cfg["n_layer"] = getattr(self.model.config, "num_hidden_layers", None)
            cfg["d_model"] = getattr(self.model.config, "hidden_size", None)
        return cfg

    def set_dynamic_manager(self, mgr):
        """
        Attach an optional DynamicManager (splash.dynamic) to transform hidden states
        before they are converted to numpy. Purely optional; no effect if None.
        """
        self._dyn_mgr = mgr

    def has_dynamic_manager(self) -> bool:
        return hasattr(self, "_dyn_mgr") and (self._dyn_mgr is not None)


class APIAdapter:
    """
    Lightweight adapter for API-based LLMs where only final hidden or logits are exposed.
    This is a template; users supply a client with .embed(texts) -> (layers or final).
    """
    def __init__(self, client, model_id: str, *, seed: int = 42):
        self.client = client
        self.model_id = model_id
        self.seed = int(seed)

    def tokenize(self, texts: List[str], *, max_length: Optional[int] = None) -> SequenceBatch:
        # Assume client does its own tokenization; we keep texts only.
        # For reproducibility, we keep a placeholder (B, T) array by splitting on spaces, bounded by max_length.
        ids = []
        for t in texts:
            toks = t.split()
            if max_length is not None:
                toks = toks[:max_length]
            ids.append([hash(tok) % 50000 for tok in toks])  # pseudo-ids; not used downstream
        maxT = max((len(x) for x in ids), default=1)
        arr = np.zeros((len(ids), maxT), dtype=np.int64)
        for i, row in enumerate(ids):
            arr[i, :len(row)] = row
        mask = (arr != 0).astype(np.int64)
        return SequenceBatch(input_ids=arr, attention_mask=mask, texts=texts)

    def forward(self, batch: SequenceBatch, *, capture_attention: bool = False) -> ModelOutputs:
        # Client must return numpy arrays. Expected:
        # - either per-layer: List[(B,T,d)] or a single (B,T,d)
        out = self.client.embed(batch.texts, model_id=self.model_id, seed=self.seed, return_layers=True)
        if isinstance(out, list):
            layers = out
            final = layers[-1]
        else:
            layers = [out]
            final = out
        hidden = HiddenStates(layers=layers, final=final, layer_names=[f"layer_{i}" for i in range(len(layers))])
        return ModelOutputs(batch=batch, hidden=hidden, attention=None, token_embeddings=None, pos_embeddings=None)

    def get_config(self) -> Dict[str, Any]:
        return {"adapter": "APIAdapter", "model": self.model_id}
