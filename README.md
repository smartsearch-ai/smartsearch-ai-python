# SmartSearch AI SDK for Python

## Registering users

`users().submit_job(...)` registers users through an integration that an administrator set up in SmartSearch
(this SDK cannot create one). Users are created in SmartSearch's sign-in realm either way:

- **Invite** (`KEYCLOAK_INVITE`): each user gets an email to set a SmartSearch password. Send `EnsureItem` without
  `external_identity`.
- **Federated** (`KEYCLOAK_FEDERATED`): each user is linked to their existing account at your identity provider.
  Every `EnsureItem` must carry `external_identity=ExternalIdentity(subject=..., username=...)`.

Named codes (`SmartSearchError.code`, and the code of a failed job item): `INTEGRATION_NOT_FOUND`,
`EXTERNAL_IDENTITY_REQUIRED`, `EXTERNAL_IDENTITY_NOT_SUPPORTED`, importable from `smartsearch_ai`.
