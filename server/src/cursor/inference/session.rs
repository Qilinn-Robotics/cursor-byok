//! Serves one locally routed RunInference stream: RunReady, model invocations, and shutdown.
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use bytes::Bytes;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cursor::{
        compile::requested_spec,
        protocol::{
            connect,
            proto::{agent::v1 as agent, inference as pb},
        },
    },
    model::{ModelInvocation, ModelSpec},
    provider::Provider,
    Error, Result,
};

use super::{metadata, translate};
use pb::{
    run_inference_client_message::Message as Client,
    run_inference_server_message::Message as Server,
};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// Connect codes reported in `RunInferenceInvocationError.code`.
const CODE_CANCELED: i32 = 1;
const CODE_UNKNOWN: i32 = 2;

/// What BYOK knows about the selected model before the stream opens.
pub struct LocalModel {
    pub display_name: String,
    /// Window configured in BYOK; Cursor's own `context` choice takes precedence.
    pub context_window_tokens: Option<u64>,
}

pub struct InferenceSession {
    request_id: String,
    conversation_id: String,
    model: ModelSpec,
    provider: Arc<dyn Provider>,
    output: mpsc::UnboundedSender<Bytes>,
    receiver: Mutex<Option<mpsc::UnboundedReceiver<Bytes>>>,
    invocations: Mutex<HashMap<String, CancellationToken>>,
    closed: CancellationToken,
    calls: AtomicU64,
}

impl InferenceSession {
    /// Opens the stream by answering the client's RunRequest with RunReady.
    pub fn start(
        request_id: &str,
        run: &pb::RunInferenceRunRequest,
        local: LocalModel,
        provider: Arc<dyn Provider>,
    ) -> Result<Arc<Self>> {
        let requested = run
            .requested_model
            .clone()
            .ok_or_else(|| Error::Protocol("RunInference RunRequest has no model".into()))?;
        let mut model = model_spec(&requested)?;
        // Same window the Agent Run compaction uses: Cursor's context choice, else the BYOK config.
        model.context_window_tokens = model.context_window_tokens.or(local.context_window_tokens);
        let ready =
            metadata::run_ready(&requested, &local.display_name, model.context_window_tokens)?;
        let (output, receiver) = mpsc::unbounded_channel();
        let session = Arc::new(Self {
            request_id: request_id.to_owned(),
            conversation_id: run.conversation_id.clone(),
            model,
            provider,
            output,
            receiver: Mutex::new(Some(receiver)),
            invocations: Mutex::new(HashMap::new()),
            closed: CancellationToken::new(),
            calls: AtomicU64::new(0),
        });
        session.emit(Server::RunReady(ready));
        session.emit(heartbeat());
        let heartbeats = session.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = heartbeats.closed.cancelled() => return,
                    () = tokio::time::sleep(HEARTBEAT_INTERVAL) => heartbeats.emit(heartbeat()),
                }
            }
        });
        Ok(session)
    }

    pub fn model_id(&self) -> &str {
        &self.model.model_id
    }

    /// Hands the server-message stream to the RunInferenceSSE response; only one subscriber exists.
    pub fn take_output(&self) -> Option<mpsc::UnboundedReceiver<Bytes>> {
        self.receiver
            .lock()
            .expect("inference receiver lock")
            .take()
    }

    pub fn append(self: &Arc<Self>, message: pb::RunInferenceClientMessage) -> Result<()> {
        match message.message {
            Some(Client::InvokeModel(invoke)) => self.invoke(invoke),
            Some(Client::CancelInvocation(cancel)) => {
                if let Some(token) = self.invocation(&cancel.invocation_id) {
                    token.cancel();
                }
            }
            Some(Client::FinishRun(finish)) => {
                if finish.aborted {
                    self.cancel_invocations();
                }
                let _ = self.output.send(connect::encode_end_stream());
                self.closed.cancel();
            }
            Some(Client::RunRequest(_)) => {
                return Err(Error::Protocol(
                    "RunInference stream received a second RunRequest".into(),
                ))
            }
            // Heartbeats, telemetry and timings need no answer; RunReady does not advertise them.
            _ => {}
        }
        Ok(())
    }

    /// Stops every invocation once the client stream is gone.
    pub fn close(&self) {
        self.cancel_invocations();
        self.closed.cancel();
    }

    fn invoke(self: &Arc<Self>, invoke: pb::RunInferenceInvokeModel) {
        let token = self.closed.child_token();
        self.invocations
            .lock()
            .expect("inference invocations lock")
            .insert(invoke.invocation_id.clone(), token.clone());
        let session = self.clone();
        tokio::spawn(async move {
            let invocation_id = invoke.invocation_id.clone();
            let result = session.run_invocation(invoke, token.clone()).await;
            session
                .invocations
                .lock()
                .expect("inference invocations lock")
                .remove(&invocation_id);
            let error = match result {
                Ok(()) => None,
                Err(error) => {
                    let code = if token.is_cancelled() {
                        CODE_CANCELED
                    } else {
                        tracing::warn!(
                            request_id = session.request_id,
                            invocation_id,
                            %error,
                            "RunInference invocation failed"
                        );
                        CODE_UNKNOWN
                    };
                    Some(pb::RunInferenceInvocationError {
                        code,
                        message: error.to_string(),
                        ..Default::default()
                    })
                }
            };
            session.emit(Server::InvocationEnd(pb::RunInferenceInvocationEnd {
                invocation_id,
                error,
                delta_refused: None,
            }));
        });
    }

    async fn run_invocation(
        &self,
        invoke: pb::RunInferenceInvokeModel,
        token: CancellationToken,
    ) -> Result<()> {
        let request = invoke.request.ok_or_else(|| {
            Error::Protocol("RunInference invocation carries no plain request".into())
        })?;
        let model = match request.requested_model.as_ref() {
            Some(requested) if requested.model_id == self.model.model_id => {
                let mut model = model_spec(requested)?;
                model.context_window_tokens = model
                    .context_window_tokens
                    .or(self.model.context_window_tokens);
                model
            }
            _ => self.model.clone(),
        };
        let model_id = model.model_id.clone();
        let (request, names) = translate::model_request(&request, model)?;
        let invocation = ModelInvocation {
            call_id: format!("{}:{}", self.request_id, invoke.invocation_id),
            run_id: self.request_id.clone(),
            conversation_id: self.conversation_id.clone(),
            provider_call_index: self.calls.fetch_add(1, Ordering::Relaxed),
            request,
        };
        let mut encoder = translate::StreamEncoder::new(names, model_id);
        let mut stream = self.provider.stream(invocation, token.clone());
        loop {
            let event = tokio::select! {
                () = token.cancelled() => return Err(Error::Cancelled),
                event = stream.next() => event,
            };
            let Some(event) = event else {
                return Ok(());
            };
            for response in encoder.event(event?) {
                self.emit(Server::InvocationResponse(
                    pb::RunInferenceInvocationResponse {
                        invocation_id: invoke.invocation_id.clone(),
                        response: Some(response),
                    },
                ));
            }
        }
    }

    fn invocation(&self, invocation_id: &str) -> Option<CancellationToken> {
        self.invocations
            .lock()
            .expect("inference invocations lock")
            .get(invocation_id)
            .cloned()
    }

    fn cancel_invocations(&self) {
        for token in self
            .invocations
            .lock()
            .expect("inference invocations lock")
            .values()
        {
            token.cancel();
        }
    }

    fn emit(&self, message: Server) {
        match connect::encode_message(&pb::RunInferenceServerMessage {
            message: Some(message),
        }) {
            Ok(frame) => {
                let _ = self.output.send(frame);
            }
            Err(error) => tracing::warn!(%error, "cannot encode RunInference server message"),
        }
    }
}

impl Drop for InferenceSession {
    fn drop(&mut self) {
        self.close();
    }
}

fn heartbeat() -> Server {
    Server::Heartbeat(pb::RunInferenceHeartbeat::default())
}

fn model_spec(requested: &pb::InferenceRequestedModel) -> Result<ModelSpec> {
    requested_spec(&agent::RequestedModel {
        model_id: requested.model_id.clone(),
        max_mode: requested.max_mode,
        parameters: requested
            .parameters
            .iter()
            .map(|parameter| agent::requested_model::ModelParameterValue {
                id: parameter.id.clone(),
                value: parameter.value.clone(),
            })
            .collect(),
        ..Default::default()
    })
}

/// Recognizes the first append of a RunInference stream. Agent Run appends start with an
/// `AgentRunRequest` whose field 1 is an encoded message, never a bare conversation UUID.
pub fn decode_start(payload: &[u8]) -> Option<pb::RunInferenceRunRequest> {
    let message = <pb::RunInferenceClientMessage as prost::Message>::decode(payload).ok()?;
    let Some(Client::RunRequest(run)) = message.message else {
        return None;
    };
    (uuid::Uuid::parse_str(&run.conversation_id).is_ok() && run.requested_model.is_some())
        .then_some(run)
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::{
        model::Usage,
        provider::{FinishReason, ModelEvent, ProviderStream},
    };

    struct ScriptedProvider;

    impl Provider for ScriptedProvider {
        fn stream(
            &self,
            _invocation: ModelInvocation,
            _cancel: CancellationToken,
        ) -> ProviderStream {
            Box::pin(futures_util::stream::iter(
                [
                    ModelEvent::Start {
                        model_call_id: "resp".into(),
                    },
                    ModelEvent::TextDelta("hi".into()),
                    ModelEvent::Usage(Usage::default()),
                    ModelEvent::Done(FinishReason::Stop),
                ]
                .into_iter()
                .map(Ok),
            ))
        }
    }

    fn local() -> LocalModel {
        LocalModel {
            display_name: "BYOK".into(),
            context_window_tokens: Some(200_000),
        }
    }

    fn run_request() -> pb::RunInferenceRunRequest {
        pb::RunInferenceRunRequest {
            conversation_id: uuid::Uuid::new_v4().to_string(),
            requested_model: Some(pb::InferenceRequestedModel {
                model_id: "byok".into(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn server_messages(frames: &[Bytes]) -> Vec<Server> {
        frames
            .iter()
            .filter(|frame| frame[0] & connect::END_STREAM_FLAG == 0)
            .map(|frame| {
                pb::RunInferenceServerMessage::decode(&frame[5..])
                    .unwrap()
                    .message
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn recognizes_run_inference_start_but_not_agent_run() {
        let start = pb::RunInferenceClientMessage {
            message: Some(Client::RunRequest(run_request())),
        }
        .encode_to_vec();
        assert!(decode_start(&start).is_some());

        let agent_run = agent::AgentClientMessage {
            message: Some(agent::agent_client_message::Message::RunRequest(
                agent::AgentRunRequest {
                    conversation_state: Some(Default::default()),
                    conversation_id: Some(uuid::Uuid::new_v4().to_string()),
                    ..Default::default()
                },
            )),
        }
        .encode_to_vec();
        assert!(decode_start(&agent_run).is_none());
    }

    #[tokio::test]
    async fn serves_run_ready_invocation_and_finish() {
        let session =
            InferenceSession::start("req", &run_request(), local(), Arc::new(ScriptedProvider))
                .unwrap();
        let mut output = session.take_output().unwrap();
        session
            .append(pb::RunInferenceClientMessage {
                message: Some(Client::InvokeModel(pb::RunInferenceInvokeModel {
                    invocation_id: "inv-1".into(),
                    request: Some(pb::InferenceStreamRequest::default()),
                    ..Default::default()
                })),
            })
            .unwrap();

        let mut frames = Vec::new();
        while let Some(frame) = output.recv().await {
            frames.push(frame);
            if matches!(
                server_messages(&frames).last(),
                Some(Server::InvocationEnd(_))
            ) {
                break;
            }
        }
        session
            .append(pb::RunInferenceClientMessage {
                message: Some(Client::FinishRun(pb::RunInferenceFinishRun::default())),
            })
            .unwrap();
        while let Some(frame) = output.recv().await {
            let end = frame[0] & connect::END_STREAM_FLAG != 0;
            frames.push(frame);
            if end {
                break;
            }
        }

        let messages = server_messages(&frames);
        assert!(matches!(messages[0], Server::RunReady(_)));
        assert!(messages.iter().any(|message| matches!(
            message,
            Server::InvocationResponse(pb::RunInferenceInvocationResponse { response: Some(pb::InferenceStreamResponse { response: Some(pb::inference_stream_response::Response::TextPart(part)), .. }), .. }) if part.text == "hi"
        )));
        let Some(Server::InvocationEnd(end)) = messages
            .iter()
            .find(|message| matches!(message, Server::InvocationEnd(_)))
        else {
            panic!("missing invocation end");
        };
        assert!(end.error.is_none());
        assert!(frames.last().unwrap()[0] & connect::END_STREAM_FLAG != 0);
    }
}
