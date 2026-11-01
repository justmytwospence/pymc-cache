# Changelog

## 0.1.0 (unreleased)

- Extract independent PyMC model-aware sampling cache from cloudposterior.
- Add versioned, fail-closed graph and call identity; bounded memory and atomic NetCDF disk backends.
- Add a preflight session API, safe disk-management CLI, and PyMC 5/6 and ArviZ 0/1 support.
- Encode nutpie divergence messages losslessly; support nullable diagnostic warning text and floating NaN without dropping data.
- Pre-release identity and disk schema 3 invalidate schema-1/2 entries; old files are misses, left untouched and never overwritten. Existing caches must be sampled again.

Old cloudposterior keys are deliberately not trusted or imported; old files are not deleted.
