//! Serves Cursor's RunInference protocol: the agent loop runs inside Cursor and BYOK only
//! answers its model invocations with the configured provider.
mod metadata;
mod session;
mod translate;

pub use session::{decode_start, InferenceSession, LocalModel};
