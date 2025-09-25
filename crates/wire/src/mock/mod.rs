//! Facilities for running a lightweight mock HTTP server that mimics
//! LLM provider endpoints. Useful for integration-style tests or
//! applications that want to exercise clients without contacting real
//! services.

mod server;

pub use server::*;
