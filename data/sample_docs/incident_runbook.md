# Production Incident Response Runbook

## 1. Severity Classification

Every production incident must be assigned a severity level within 5 minutes of detection:

- **SEV-1 (Critical):** Complete outage affecting all users, active data loss, or a confirmed security breach. Initial acknowledgement SLA is **5 minutes**, and status updates are required every **15 minutes** in `#inc-war-room`.
- **SEV-2 (Major):** Partial degradation affecting more than 10% of requests or core checkout workflows with a workaround available. Initial acknowledgement SLA is **15 minutes**, with updates every **30 minutes**.
- **SEV-3 (Minor):** Non-critical bug or single-tenant issue with minimal customer impact. Triaged during normal business hours within **4 business hours**.

## 2. Database Connection Pool Exhaustion (`ERR_DB_POOL_EXHAUSTED`)

When API pods emit `ERR_DB_POOL_EXHAUSTED` and p99 latency spikes above 3,000 ms:

1. Check active connections in the primary PostgreSQL cluster using `SELECT count(*), state FROM pg_stat_activity GROUP BY state;`.
2. Inspect recent deploys from the last 60 minutes. If a release introduced an unclosed transaction or missing timeout, **roll back immediately** via `argocd app rollback payments-api`.
3. As a temporary mitigation, raise `PG_POOL_MAX_SIZE` from `25` to `40` only if CPU utilization on the primary database is below 65%.
4. Never restart the primary database instance without Incident Commander sign-off.

## 3. Escalation & Post-Mortem Policy

If a SEV-1 incident is not mitigated within **30 minutes**, the on-call engineer must page the Secondary On-Call and the Engineering Manager on Duty.

A blameless post-mortem document must be opened within **24 hours** of resolution and reviewed in the weekly reliability meeting. Action items from SEV-1 post-mortems are P0 sprint commitments due within 14 days.
