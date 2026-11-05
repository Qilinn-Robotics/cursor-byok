//! Converts RunInference model calls to provider requests and provider events back to stream parts.
use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine};
use prost_types::{value::Kind, ListValue, Struct, Value as PbValue};
use serde_json::{json, Map, Value};

use crate::{
    cursor::protocol::proto::inference as pb,
    model::{
        normalize_tool_name, ContentPart, ModelRequest, ModelSpec, ProjectedContent,
        ProjectedMessage, PromptSpec, Role, ToolCallContent, ToolDefinition, ToolResultContent,
        Usage,
    },
    provider::ModelEvent,
    Error, Result,
};

use pb::{
    inference_content_part::Part, inference_core_message::Content,
    inference_stream_response::Response as StreamPart, InferenceMessageRole,
};

/// Provider-safe tool names mapped back to the names Cursor advertised.
#[derive(Clone, Debug, Default)]
pub struct ToolNames(HashMap<String, String>);

impl ToolNames {
    fn provider(&mut self, original: &str) -> String {
        let normalized = normalize_tool_name(original);
        self.0
            .entry(normalized.clone())
            .or_insert_with(|| original.to_owned());
        normalized
    }

    pub fn original(&self, provider: &str) -> String {
        self.0
            .get(provider)
            .cloned()
            .unwrap_or_else(|| provider.to_owned())
    }
}

pub fn model_request(
    request: &pb::InferenceStreamRequest,
    mut model: ModelSpec,
) -> Result<(ModelRequest, ToolNames)> {
    let mut names = ToolNames::default();
    let tools = request
        .tools
        .iter()
        .map(|tool| ToolDefinition {
            name: names.provider(&tool.name),
            description: tool.description.clone(),
            parameters: tool
                .parameters
                .as_ref()
                .map(struct_to_json)
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        })
        .collect();
    let mut instructions = Vec::new();
    let mut history = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        let id = format!("inference-{index}");
        match message.role() {
            InferenceMessageRole::System if history.is_empty() => {
                instructions.push(content_text(message));
            }
            InferenceMessageRole::System => history.push(ProjectedMessage {
                message_id: id,
                role: Role::System,
                content: ProjectedContent::Parts(vec![ContentPart::Text {
                    text: content_text(message),
                }]),
            }),
            InferenceMessageRole::User | InferenceMessageRole::Unspecified => {
                history.push(ProjectedMessage {
                    message_id: id,
                    role: Role::User,
                    content: ProjectedContent::Parts(user_parts(message)),
                })
            }
            InferenceMessageRole::Assistant => {
                history.push(assistant_message(id, message, &mut names)?)
            }
            InferenceMessageRole::Tool => {
                let Some(Content::ToolContent(results)) = &message.content else {
                    return Err(Error::Protocol(
                        "RunInference tool message has no tool results".into(),
                    ));
                };
                for (part_index, result) in results.parts.iter().enumerate() {
                    history.push(ProjectedMessage {
                        message_id: format!("{id}-{part_index}"),
                        role: Role::Tool,
                        content: ProjectedContent::ToolResult(tool_result(result, &mut names)),
                    });
                }
            }
        }
    }
    if let Some(max_tokens) = request
        .model_config
        .as_ref()
        .and_then(|config| config.max_tokens)
        .filter(|tokens| *tokens > 0)
    {
        model.max_output_tokens = Some(max_tokens as u64);
    }
    Ok((
        ModelRequest {
            prompt: PromptSpec {
                instructions: instructions.join("\n\n"),
                tools,
            },
            model,
            history,
        },
        names,
    ))
}

fn content_text(message: &pb::InferenceCoreMessage) -> String {
    match &message.content {
        Some(Content::Text(text)) => text.clone(),
        Some(Content::Parts(parts)) => parts
            .parts
            .iter()
            .filter_map(|part| match &part.part {
                Some(Part::Text(text)) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(Content::ToolContent(_)) | None => String::new(),
    }
}

fn user_parts(message: &pb::InferenceCoreMessage) -> Vec<ContentPart> {
    match &message.content {
        Some(Content::Parts(parts)) => parts.parts.iter().filter_map(content_part).collect(),
        _ => vec![ContentPart::Text {
            text: content_text(message),
        }],
    }
}

fn content_part(part: &pb::InferenceContentPart) -> Option<ContentPart> {
    match part.part.as_ref()? {
        Part::Text(text) => Some(ContentPart::Text {
            text: text.text.clone(),
        }),
        Part::Image(image) => Some(image_part(&image.data, image.mime_type.as_deref())),
        Part::File(file) => Some(ContentPart::Text {
            text: format!("[File: {}]", file.filename.as_deref().unwrap_or("unnamed")),
        }),
    }
}

/// Cursor sends images as data URLs or bare base64; anything else is kept as a textual reference.
fn image_part(data: &str, mime_type: Option<&str>) -> ContentPart {
    let (mime, encoded) = match data
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
    {
        Some((header, encoded)) => (
            header.trim_end_matches(";base64").to_owned(),
            encoded.to_owned(),
        ),
        None => (mime_type.unwrap_or("image/png").to_owned(), data.to_owned()),
    };
    match STANDARD.decode(encoded.trim()) {
        Ok(bytes) => ContentPart::Image {
            mime_type: mime,
            data: bytes,
        },
        Err(_) => ContentPart::Text {
            text: format!("[Image: {data}]"),
        },
    }
}

fn assistant_message(
    id: String,
    message: &pb::InferenceCoreMessage,
    names: &mut ToolNames,
) -> Result<ProjectedMessage> {
    let text = match &message.content {
        Some(Content::Text(text)) => text.clone(),
        _ if !message.assistant_text_parts.is_empty() => message.assistant_text_parts.join(""),
        _ => content_text(message),
    };
    let thinking = message
        .reasoning_parts
        .iter()
        .filter(|part| !part.is_redacted)
        .map(|part| part.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    let calls = message
        .tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| ToolCallContent {
            index,
            call_id: call.tool_call_id.clone(),
            name: names.provider(&call.tool_name),
            arguments: call
                .args
                .as_ref()
                .map(struct_to_json)
                .or_else(|| {
                    call.raw_tool_call_args
                        .as_deref()
                        .and_then(|raw| serde_json::from_str(raw).ok())
                })
                .unwrap_or_else(|| json!({})),
        })
        .collect();
    Ok(ProjectedMessage {
        message_id: id,
        role: Role::Assistant,
        content: ProjectedContent::Assistant {
            text,
            thinking,
            replay_state: None,
            calls,
        },
    })
}

fn tool_result(result: &pb::InferenceToolResultPart, names: &mut ToolNames) -> ToolResultContent {
    let content = match result.result.as_ref().map(value_to_json) {
        Some(Value::String(text)) => text,
        Some(Value::Null) | None => String::new(),
        Some(value) => value.to_string(),
    };
    let images: Vec<_> = result
        .experimental_content
        .iter()
        .filter_map(content_part)
        .filter(|part| matches!(part, ContentPart::Image { .. }))
        .collect();
    let provider_parts = if images.is_empty() {
        Vec::new()
    } else {
        std::iter::once(ContentPart::Text {
            text: content.clone(),
        })
        .chain(images)
        .collect()
    };
    ToolResultContent {
        call_id: result.tool_call_id.clone(),
        name: names.provider(&result.tool_name),
        content,
        is_error: result.is_error || result.cursor_tool_call_is_error == Some(true),
        image: None,
        provider_parts,
    }
}

/// Turns one provider call into RunInference stream parts and the final response summary.
pub struct StreamEncoder {
    names: ToolNames,
    model: String,
    response_id: String,
    text: String,
    thinking: String,
    calls: Vec<StreamedCall>,
    usage: Usage,
}

struct StreamedCall {
    index: usize,
    call_id: String,
    name: String,
    args: String,
}

impl StreamEncoder {
    pub fn new(names: ToolNames, model: String) -> Self {
        Self {
            names,
            model,
            response_id: String::new(),
            text: String::new(),
            thinking: String::new(),
            calls: Vec::new(),
            usage: Usage::default(),
        }
    }

    /// Returns the parts to stream for one provider event; `Done` yields the closing parts.
    pub fn event(&mut self, event: ModelEvent) -> Vec<pb::InferenceStreamResponse> {
        let part = match event {
            ModelEvent::Start { model_call_id } => {
                self.response_id = model_call_id;
                return Vec::new();
            }
            ModelEvent::TextDelta(delta) => {
                self.text.push_str(&delta);
                StreamPart::TextPart(pb::InferenceTextStreamPart {
                    text: delta,
                    is_final: false,
                })
            }
            ModelEvent::ThinkingDelta(delta) => {
                self.thinking.push_str(&delta);
                StreamPart::ThinkingPart(pb::InferenceThinkingStreamPart {
                    text: delta,
                    ..Default::default()
                })
            }
            ModelEvent::ToolCallStart {
                index,
                call_id,
                name,
            } => {
                let name = self.names.original(&name);
                self.calls.push(StreamedCall {
                    index,
                    call_id: call_id.clone(),
                    name: name.clone(),
                    args: String::new(),
                });
                StreamPart::ToolCallPart(pb::InferenceToolCallStreamPart {
                    tool_call_id: call_id,
                    tool_name: name,
                    tool_index: Some(index as i32),
                    ..Default::default()
                })
            }
            ModelEvent::ToolCallArgumentsDelta { index, delta } => {
                let Some(call) = self.call(index) else {
                    return Vec::new();
                };
                call.args.push_str(&delta);
                StreamPart::ToolCallPart(pb::InferenceToolCallStreamPart {
                    tool_call_id: call.call_id.clone(),
                    args: delta,
                    tool_index: Some(index as i32),
                    ..Default::default()
                })
            }
            ModelEvent::ToolCallEnd { index } => {
                let Some(call) = self.call(index) else {
                    return Vec::new();
                };
                StreamPart::ToolCallPart(pb::InferenceToolCallStreamPart {
                    tool_call_id: call.call_id.clone(),
                    tool_name: call.name.clone(),
                    args: complete_args(&call.args),
                    is_complete: true,
                    tool_index: Some(index as i32),
                    ..Default::default()
                })
            }
            ModelEvent::Usage(usage) => {
                self.usage += usage;
                return Vec::new();
            }
            ModelEvent::Done(_) => return self.finish(),
            ModelEvent::TextStart
            | ModelEvent::TextEnd
            | ModelEvent::ThinkingStart
            | ModelEvent::ThinkingEnd
            | ModelEvent::ProviderReplayState(_) => return Vec::new(),
        };
        vec![stream_part(part)]
    }

    fn call(&mut self, index: usize) -> Option<&mut StreamedCall> {
        self.calls.iter_mut().find(|call| call.index == index)
    }

    fn finish(&mut self) -> Vec<pb::InferenceStreamResponse> {
        let input = token_count(self.usage.input_tokens);
        let output = token_count(self.usage.output_tokens);
        let message = pb::InferenceResponseMessage {
            id: self.response_id.clone(),
            role: InferenceMessageRole::Assistant as i32,
            content: Some(self.text.clone()),
            tool_calls: self
                .calls
                .iter()
                .map(|call| pb::InferenceToolCall {
                    tool_call_id: call.call_id.clone(),
                    tool_name: call.name.clone(),
                    args: Some(json_to_struct(&parse_args(&call.args))),
                    raw_tool_call_args: Some(complete_args(&call.args)),
                })
                .collect(),
            reasoning_parts: (!self.thinking.is_empty())
                .then(|| pb::InferenceReasoningPart {
                    text: self.thinking.clone(),
                    ..Default::default()
                })
                .into_iter()
                .collect(),
            ..Default::default()
        };
        vec![
            stream_part(StreamPart::Usage(pb::InferenceUsageInfo {
                prompt_tokens: input,
                completion_tokens: output,
                total_tokens: Some(input.saturating_add(output)),
            })),
            stream_part(StreamPart::ExtendedUsage(pb::InferenceExtendedUsageInfo {
                input_tokens: input,
                output_tokens: output,
                cache_read_tokens: token_count(self.usage.cache_read_tokens),
                cache_write_tokens: token_count(self.usage.cache_write_tokens),
                max_tokens: 0,
                reasoning_tokens: self.usage.reasoning_tokens.map(|tokens| tokens as i32),
            })),
            stream_part(StreamPart::ResponseInfo(pb::InferenceResponseInfo {
                id: self.response_id.clone(),
                model: self.model.clone(),
                created_at: chrono::Utc::now().timestamp_millis(),
                messages: vec![message],
                ..Default::default()
            })),
            stream_part(StreamPart::TextPart(pb::InferenceTextStreamPart {
                text: String::new(),
                is_final: true,
            })),
        ]
    }
}

fn stream_part(part: StreamPart) -> pb::InferenceStreamResponse {
    pb::InferenceStreamResponse {
        response: Some(part),
        ..Default::default()
    }
}

fn token_count(value: Option<u64>) -> i32 {
    value.unwrap_or_default().min(i32::MAX as u64) as i32
}

fn complete_args(args: &str) -> String {
    if args.trim().is_empty() {
        "{}".into()
    } else {
        args.to_owned()
    }
}

fn parse_args(args: &str) -> Value {
    match serde_json::from_str(args) {
        Ok(Value::Object(object)) => Value::Object(object),
        _ => json!({}),
    }
}

pub fn struct_to_json(value: &Struct) -> Value {
    Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), value_to_json(value)))
            .collect(),
    )
}

pub fn value_to_json(value: &PbValue) -> Value {
    match &value.kind {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::NumberValue(number)) => serde_json::Number::from_f64(*number)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::StringValue(text)) => Value::String(text.clone()),
        Some(Kind::BoolValue(flag)) => Value::Bool(*flag),
        Some(Kind::StructValue(object)) => struct_to_json(object),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(value_to_json).collect())
        }
    }
}

pub fn json_to_struct(value: &Value) -> Struct {
    match value {
        Value::Object(object) => object_to_struct(object),
        _ => Struct::default(),
    }
}

fn object_to_struct(object: &Map<String, Value>) -> Struct {
    Struct {
        fields: object
            .iter()
            .map(|(key, value)| (key.clone(), json_to_value(value)))
            .collect(),
    }
}

fn json_to_value(value: &Value) -> PbValue {
    let kind = match value {
        Value::Null => Kind::NullValue(0),
        Value::Bool(flag) => Kind::BoolValue(*flag),
        Value::Number(number) => Kind::NumberValue(number.as_f64().unwrap_or_default()),
        Value::String(text) => Kind::StringValue(text.clone()),
        Value::Array(items) => Kind::ListValue(ListValue {
            values: items.iter().map(json_to_value).collect(),
        }),
        Value::Object(object) => Kind::StructValue(object_to_struct(object)),
    };
    PbValue { kind: Some(kind) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_message(role: InferenceMessageRole, text: &str) -> pb::InferenceCoreMessage {
        pb::InferenceCoreMessage {
            role: role as i32,
            content: Some(Content::Text(text.into())),
            ..Default::default()
        }
    }

    #[test]
    fn converts_conversation_with_tool_round() {
        let request = pb::InferenceStreamRequest {
            messages: vec![
                text_message(InferenceMessageRole::System, "be helpful"),
                text_message(InferenceMessageRole::User, "read a.txt"),
                pb::InferenceCoreMessage {
                    role: InferenceMessageRole::Assistant as i32,
                    content: Some(Content::Text("reading".into())),
                    tool_calls: vec![pb::InferenceToolCall {
                        tool_call_id: "call-1".into(),
                        tool_name: "Read.File".into(),
                        args: Some(json_to_struct(&json!({"path": "a.txt"}))),
                        raw_tool_call_args: None,
                    }],
                    ..Default::default()
                },
                pb::InferenceCoreMessage {
                    role: InferenceMessageRole::Tool as i32,
                    content: Some(Content::ToolContent(pb::InferenceToolResultContent {
                        parts: vec![pb::InferenceToolResultPart {
                            tool_call_id: "call-1".into(),
                            tool_name: "Read.File".into(),
                            result: Some(json_to_value(&json!("hello"))),
                            ..Default::default()
                        }],
                    })),
                    ..Default::default()
                },
            ],
            tools: vec![pb::InferenceAgentTool {
                name: "Read.File".into(),
                description: "read".into(),
                parameters: Some(json_to_struct(&json!({"type": "object"}))),
                custom_tool_format: None,
            }],
            model_config: Some(pb::InferenceModelConfig {
                max_tokens: Some(1024),
                ..Default::default()
            }),
            ..Default::default()
        };

        let (converted, names) = model_request(&request, ModelSpec::new("m")).unwrap();

        assert_eq!(converted.prompt.instructions, "be helpful");
        assert_eq!(converted.prompt.tools[0].name, "Read_File");
        assert_eq!(converted.model.max_output_tokens, Some(1024));
        assert_eq!(converted.history.len(), 3);
        let ProjectedContent::Assistant { calls, text, .. } = &converted.history[1].content else {
            panic!("expected assistant message");
        };
        assert_eq!(text, "reading");
        assert_eq!(calls[0].name, "Read_File");
        assert_eq!(calls[0].arguments, json!({"path": "a.txt"}));
        let ProjectedContent::ToolResult(result) = &converted.history[2].content else {
            panic!("expected tool result");
        };
        assert_eq!(result.content, "hello");
        assert_eq!(names.original("Read_File"), "Read.File");
    }

    #[test]
    fn decodes_data_url_images() {
        let ContentPart::Image { mime_type, data } =
            image_part("data:image/jpeg;base64,aGk=", None)
        else {
            panic!("expected image");
        };
        assert_eq!(mime_type, "image/jpeg");
        assert_eq!(data, b"hi");
    }

    #[test]
    fn encodes_tool_call_stream_and_summary() {
        let mut names = ToolNames::default();
        names.provider("Read.File");
        let mut encoder = StreamEncoder::new(names, "m".into());
        encoder.event(ModelEvent::Start {
            model_call_id: "resp-1".into(),
        });
        let start = encoder.event(ModelEvent::ToolCallStart {
            index: 0,
            call_id: "call-1".into(),
            name: "Read_File".into(),
        });
        let delta = encoder.event(ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"path\":\"a\"}".into(),
        });
        let end = encoder.event(ModelEvent::ToolCallEnd { index: 0 });
        let done = encoder.event(ModelEvent::Done(crate::provider::FinishReason::ToolUse));

        let Some(StreamPart::ToolCallPart(start)) = &start[0].response else {
            panic!("expected tool call start");
        };
        assert_eq!(start.tool_name, "Read.File");
        assert!(!start.is_complete);
        let Some(StreamPart::ToolCallPart(delta)) = &delta[0].response else {
            panic!("expected tool call delta");
        };
        assert!(delta.tool_name.is_empty());
        let Some(StreamPart::ToolCallPart(end)) = &end[0].response else {
            panic!("expected tool call end");
        };
        assert!(end.is_complete);
        assert_eq!(end.args, "{\"path\":\"a\"}");
        let Some(StreamPart::ResponseInfo(info)) = &done[2].response else {
            panic!("expected response info");
        };
        assert_eq!(info.messages[0].tool_calls[0].tool_name, "Read.File");
        assert!(matches!(
            done.last().and_then(|part| part.response.as_ref()),
            Some(StreamPart::TextPart(pb::InferenceTextStreamPart {
                is_final: true,
                ..
            }))
        ));
    }
}
