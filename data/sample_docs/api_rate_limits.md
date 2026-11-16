# Public API Rate Limits & Authentication Guide

## 1. Authentication & Key Rotation

All requests to `/v2/*` endpoints require a Bearer token passed in the `Authorization` header. API keys are scoped per environment (`sandbox` or `production`).

- Production API keys must be rotated every **90 days**.
- Up to **two active keys** can coexist during a 48-hour zero-downtime rotation window.
- Keys leaked in public repositories are automatically revoked within 60 seconds by our secret-scanning webhook.

## 2. Tier Quotas

Rate limits are enforced using a sliding 60-second window per organization ID:

- **Free Tier:** 60 requests/minute, burst cap of 15 concurrent requests, maximum payload size 2 MB.
- **Pro Tier:** 600 requests/minute, burst cap of 100 concurrent requests, maximum payload size 10 MB.
- **Enterprise Tier:** 6,000 requests/minute (customizable via SLA), dedicated regional edge endpoints, maximum payload size 50 MB.

## 3. Handling `ERR_RATE_429`

When a client exceeds its quota, the gateway returns HTTP `429 Too Many Requests` with error code `ERR_RATE_429` and the following headers:

- `X-RateLimit-Limit`: The organization's configured requests-per-minute ceiling.
- `X-RateLimit-Remaining`: Requests left in the current window (`0` when throttled).
- `Retry-After`: Integer seconds the client must wait before retrying.

Clients should implement **exponential backoff with full jitter** starting at 500 ms and capped at 32 seconds, respecting `Retry-After` whenever present. Tight retry loops without backoff may trigger an automatic 15-minute IP cooldown.
