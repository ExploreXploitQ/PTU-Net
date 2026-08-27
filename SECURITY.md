# Security policy

## Supported versions

PTU-Net has not made a stable release. Security fixes are applied to the current default branch when maintainers can reproduce and address the report. No maintenance window or response-time guarantee is currently offered.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability that could expose data, credentials, or host systems. Use the repository's private vulnerability reporting page:

<https://github.com/ExploreXploitQ/PTU-Net/security/advisories/new>

Include the affected commit, environment, reproduction steps, likely impact, and any proposed mitigation. Remove private data, access tokens, machine names, and absolute user paths from logs.

If private reporting is unavailable, open a minimal issue asking for a private contact method. Do not include exploit details in that issue.

## Research software considerations

Model checkpoints and binary array files are not trusted inputs. Load only artifacts from sources you trust. PyTorch checkpoint loading can execute code when unsafe serialized objects are used. Run third-party data preparation and training configurations with the same care as other code.
