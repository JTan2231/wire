# Wire Testing Ethos
- `wire` treats each provider contract as a source of truth. The tests exercise
  the deterministic pieces of that contract—request shape, authentication, and
  response decoding—so regressions show up before a network call ever fires.
- Scenarios focus on resilient defaults: explicit headers, stable role mapping,
  and graceful parsing when fields disappear.

## Coverage Overview
- **Request building – OpenAI**
  - Verifies `build_request` keeps the HTTPS target (`api.openai.com` on 443)
    and attaches the bearer token pulled out of the request params.
  - Ensures the JSON envelope mirrors chat history ordering, re-labels
    tool-call turns with the `assistant` role, and threads tool call IDs into
    `tool` responses.
  - Confirms tool metadata is translated into the OpenAI function schema.
- **Request building – Anthropic**
  - Asserts API-key and version headers are present for every call.
  - Exercises the message formatter that compresses consecutive
    `FunctionCallOutput` messages into a single Anthropics `tool_result` block.
  - Checks that assistant turns emit `tool_use` entries whenever tool calls are
    queued.
- **Request building – Gemini**
  - Validates the query-string key injection and path selection between
    `generateContent` and `streamGenerateContent`.
  - Ensures role remapping (`user` → `user`, `assistant` → `model`) and system
    instruction propagation into the request body.
- **Raw HTTP envelope**
  - Guards the handwritten streaming path by asserting `build_request_raw`
    emits the expected HTTP start line, host header, bearer auth, and a
    `Content-Length` that matches the serialized JSON payload.
- **Response parsing**
  - Exercises `read_json_response` paths for OpenAI, Anthropic, and Gemini so
    that each provider’s happy-path selector returns the expected text.
  - Includes a missing-field case to confirm the parser reports errors instead
    of silently succeeding with empty content.

## Test Harness Notes
- Environment-dependent secrets are injected with `temp-env` inside each test,
  keeping global state isolated while emulating production configuration.
- Tests rely on pure transformations; no sockets are opened, so suites run fast
  and deterministically inside CI.
- Future expansion targets include exercising stream processing helpers and the
  tool-execution loop once they mature and expose injectable surfaces.
