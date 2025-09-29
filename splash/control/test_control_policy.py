from .policy import RulePolicy, PolicyPack
from .types import InvariantFrame, BandLabel

def make_frame(align=0.7, tension=0.2, asym=0.1, label="warn"):
    m = {
        "alignment_score": align, "tension": tension, "asymmetry": asym,
        "bend_spread": 0.1, "ledger": 0.0, "memory_mean":0.0,
        "shed_rate":0.0, "shed_total":0.0, "effective_degree":10.0, "cap_fraction":0.0
    }
    return InvariantFrame(t=5, window=(0,5), measures=m, label=BandLabel(label=label))

def test_policy_fail_when_alignment_low():
    pol = RulePolicy(PolicyPack.balanced(), policy_id="balanced")
    d = pol.decide(make_frame(align=0.6, label="fail"))
    assert d.severity == "fail"
    assert d.op in ("temp_clamp",)

def test_policy_near_triggers_align_preview():
    pol = RulePolicy(PolicyPack.balanced(), policy_id="balanced")
    d = pol.decide(make_frame(align=0.7, label="warn"))
    assert d.severity in ("info","warn")
