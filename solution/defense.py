"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    batch_id = payload.get("batch_id")
    profile = ctx.tools.batch_profile(batch_id)
    if "error" in profile:
        return Verdict(alert=False, pillar="checks")

    row_count = profile.get("row_count", 0)
    null_rate = profile.get("null_rate", {}).get("customer_id", 0.0)
    mean_amount = profile.get("mean_amount", 0.0)
    staleness_min = profile.get("staleness_min", 0.0)
    b = ctx.baseline

    # 1. Direct single-threshold bounds (calibrated 3-sigma limits)
    if row_count < b.get("row_count_min", 435.4732) or row_count > b.get("row_count_max", 561.2948):
        return Verdict(alert=True, pillar="checks", reason="row_count out of bounds")

    if null_rate > b.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason="null_rate exceeded")

    if mean_amount < b.get("mean_amount_min", 72.7645) or mean_amount > b.get("mean_amount_max", 90.6053):
        return Verdict(alert=True, pillar="checks", reason="mean_amount out of bounds")

    if staleness_min > b.get("staleness_min_max", 8.418):
        return Verdict(alert=True, pillar="checks", reason="staleness_min exceeded")

    # 2. Combined signal check for subtle distribution shifts
    row_mean = (b.get("row_count_min", 435.4732) + b.get("row_count_max", 561.2948)) / 2.0
    row_std = (b.get("row_count_max", 561.2948) - b.get("row_count_min", 435.4732)) / 6.0
    amt_mean = (b.get("mean_amount_min", 72.7645) + b.get("mean_amount_max", 90.6053)) / 2.0
    amt_std = (b.get("mean_amount_max", 90.6053) - b.get("mean_amount_min", 72.7645)) / 6.0

    z_row = abs(row_count - row_mean) / row_std
    z_amt = abs(mean_amount - amt_mean) / amt_std

    # Multi-signal deviation check: both row_count and mean_amount deviating together
    if (z_row > 2.0 and z_amt > 2.0) or (z_row > 1.8 and z_amt > 1.8 and (z_row**2 + z_amt**2) > 10.0):
        return Verdict(alert=True, pillar="checks", reason="combined distribution shift")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    contract_id = payload.get("contract_id", "")
    checkpoint_batch_id = payload.get("checkpoint_batch_id", "")
    diff = ctx.tools.contract_diff(contract_id, checkpoint_batch_id)
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", reason=f"contract violations: {violations}")

    freshness_delay = diff.get("freshness_delay_min", 0.0)
    sla_freshness = payload.get("declared_sla", {}).get("freshness_min", 15.0)
    if freshness_delay > sla_freshness:
        return Verdict(alert=True, pillar="contracts", reason="freshness SLA violation")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    run_id = payload.get("run_id", "")
    slice_info = ctx.tools.lineage_graph_slice(run_id)
    if "error" in slice_info:
        return Verdict(alert=False, pillar="lineage")

    dur_max = ctx.baseline.get("lineage_duration_ms_max", 5134.9804)
    if slice_info.get("duration_ms", 0.0) > dur_max:
        return Verdict(alert=True, pillar="lineage", reason="lineage runtime anomaly")

    if slice_info.get("actual_downstream_count", 1) < 1:
        return Verdict(alert=True, pillar="lineage", reason="orphan output")

    actual_upstream = slice_info.get("actual_upstream", [])
    if "raw.customers" not in actual_upstream or "raw.orders" not in actual_upstream or len(actual_upstream) < 2:
        return Verdict(alert=True, pillar="lineage", reason="missing upstream dependency")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    feature_view = payload.get("feature_view", "")
    batch_id = payload.get("batch_id", "")
    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    shift_sigma = drift.get("mean_shift_sigma", 0.0)
    if shift_sigma >= 0.55:
        return Verdict(alert=True, pillar="ai_infra", reason="feature skew detected")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    corpus = payload.get("corpus", "")
    batch_id = payload.get("chunk_batch_id", "")
    drift = ctx.tools.embedding_drift(corpus, batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    centroid_shift = drift.get("centroid_shift", 0.0)
    avg_age = drift.get("avg_doc_age_days", 0.0)

    if centroid_shift >= 0.0400:
        return Verdict(alert=True, pillar="ai_infra", reason="embedding drift detected")

    if avg_age >= 46.0:
        return Verdict(alert=True, pillar="ai_infra", reason="corpus staleness detected")

    return Verdict(alert=False, pillar="ai_infra")
